from __future__ import annotations

import hashlib
import json
import re
import requests

from config import OLLAMA_MODEL, OLLAMA_TAGS_URL, OLLAMA_URL, QUESTIONS_PER_VIVA
from utils.text import clean, concept_terms, parse_json_object

_EXPERIMENT_QUESTION_CACHE: dict[str, list[dict]] = {}

_EVAL_STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "up", "about", "into", "over", "after", "is", "am", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "must", "can", "could", "what", "which",
    "who", "whom", "this", "that", "these", "those", "how", "why", "where", "when",
    "you", "your", "we", "our", "i", "me", "my", "it", "its", "or", "and", "but",
    "if", "not", "no", "yes", "experiment", "question", "answer", "explain", "describe",
}


def _extract_content_words(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9_+#.-]{2,}", (text or "").lower())
    return {w for w in words if w not in _EVAL_STOP_WORDS}


def _is_repetition_or_keyword_only(question: str, answer: str) -> bool:
    q_words = _extract_content_words(question)
    a_words = _extract_content_words(answer)

    if not a_words:
        return True

    clean_ans = clean(answer).lower()
    non_answers = {
        "idk", "dont know", "don't know", "no idea", "pass", "skip", "none",
        "na", "n/a", "nothing", "dunno", "abc", "xyz", "asdf", "test", "whatever",
        "i don't know", "i dont know", "no answer", "nil"
    }
    if clean_ans in non_answers:
        return True

    if a_words.issubset(q_words) and len(a_words) <= len(q_words):
        return True

    return False


def ollama_available() -> bool:
    try:
        return requests.get(OLLAMA_TAGS_URL, timeout=3).ok
    except Exception:
        return False


def ask_ollama(prompt: str, temperature: float = 0.2, timeout: int = 120) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_ctx": 2048,
                    "num_predict": 450,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        text = response.json().get("response") or ""
        if not text.strip():
            raise RuntimeError("Ollama returned an empty response.")
        return text.strip()
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Ollama request timed out after {timeout} seconds.")
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Ollama connection error: {exc}")


def clean_question_text(raw_q: str) -> str:
    text = clean(raw_q)
    if not text:
        return ""
    # Strip leading bullet points, numbers, or "Question 1:" prefixes
    import re
    text = re.sub(r"^(?:q(?:uestion)?\s*\d*[:.-]?|\d+[:.-]?|\*|-)\s*", "", text, flags=re.I)
    text = clean(text)
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if not text.endswith("?"):
        text += "?"
    return text


def _prepare_experiment_material(content: str, max_chars: int = 1800) -> str:
    cleaned = clean(content)
    if len(cleaned) <= max_chars:
        return cleaned

    keywords = (
        "algorithm",
        "method",
        "model",
        "parameter",
        "feature",
        "variable",
        "formula",
        "split",
        "metric",
        "train",
        "test",
        "evaluate",
        "fit",
        "cluster",
        "class",
    )
    sentences = cleaned.split(". ")
    selected = []
    current_len = 0
    for s in sentences:
        line = s.strip()
        if not line:
            continue
        if any(kw in line.lower() for kw in keywords) or current_len < max_chars // 2:
            selected.append(line)
            current_len += len(line)
            if current_len >= max_chars:
                break

    res = ". ".join(selected)
    if len(res) < 200:
        res = cleaned[:max_chars]
    return res[:max_chars]


def generate_viva_questions(
    experiment_content: str,
    count: int = QUESTIONS_PER_VIVA,
) -> list[dict]:
    """
    Generate exactly 5 easy-to-moderate undergraduate conceptual/application viva questions.
    Questions test practical understanding, scenarios, troubleshooting, interpretation, and trade-offs.
    No generic recall headings ('aim', 'procedure', 'what is X').
    Cached by experiment content hash.
    """
    raw_material = clean(experiment_content)
    if len(raw_material) < 40:
        raise RuntimeError(
            "This experiment does not contain enough manual content to generate viva questions."
        )

    # Check cache (Req 6 & 10)
    cache_key = hashlib.md5(raw_material.encode("utf-8")).hexdigest()
    if cache_key in _EXPERIMENT_QUESTION_CACHE:
        cached = _EXPERIMENT_QUESTION_CACHE[cache_key]
        if len(cached) >= count:
            return cached[:count]

    target_count = count
    truncated_material = _prepare_experiment_material(raw_material, max_chars=1800)
    concepts = concept_terms(truncated_material, 12)

    def _request_questions(num_needed: int, existing_questions: list[str]) -> list[dict]:
        avoid_str = ""
        if existing_questions:
            avoid_str = "\nDo NOT ask about: " + "; ".join(existing_questions)

        prompt = f"""You are a friendly, encouraging university laboratory viva examiner.

Generate EXACTLY {num_needed} clear, natural, easy-to-moderate undergraduate viva questions based ONLY on this experiment.{avoid_str}

STRICT EXAMINER GUIDELINES:
1. Questions MUST be easy-to-moderate undergraduate viva level. Keep them clear, natural, grammatically correct, with NO typos or spelling errors.
2. Questions MUST test practical understanding, scenarios, troubleshooting, parameter choices, model interpretation, or applications.
3. NEVER ask generic recall questions like "What is the aim?", "Explain the procedure", "What is the objective?", or simple "What is X?" definitions.
4. Do NOT ask overly complex theoretical proofs. Keep questions approachable and conversational for an undergraduate student.

Return STRICT JSON ONLY matching format:
{{"questions": [{{"question": "Clear natural viva question?", "expected_concepts": ["concept1", "concept2"], "difficulty": "medium"}}]}}

EXPERIMENT CONTENT:
{truncated_material}"""

        raw = ask_ollama(prompt, temperature=0.2, timeout=120)
        payload = parse_json_object(raw)
        if not payload or not isinstance(payload.get("questions"), list):
            return []

        results = []
        for item in payload["questions"]:
            if not isinstance(item, dict):
                continue
            q_text = clean_question_text(item.get("question") or "")
            if not q_text:
                continue

            # Reject generic heading or simple definition questions if generated
            lower_q = q_text.lower()
            if any(
                phrase in lower_q
                for phrase in (
                    "what is the objective",
                    "what is the aim",
                    "explain the procedure",
                    "what are the steps",
                    "what is the title",
                    "what is the description",
                    "what is linear regression?",
                    "what is decision tree?",
                )
            ):
                continue

            exp_concepts = item.get("expected_concepts") or []
            if not isinstance(exp_concepts, list):
                exp_concepts = [str(exp_concepts)]
            exp_concepts = [clean(str(t)) for t in exp_concepts if clean(str(t))][:8]
            if not exp_concepts:
                exp_concepts = concepts[:4]

            diff = str(item.get("difficulty") or "medium").lower()
            if diff not in {"easy", "medium", "hard"}:
                diff = "medium"

            results.append(
                {
                    "question": q_text,
                    "expected_concepts": exp_concepts,
                    "difficulty": diff,
                }
            )
        return results

    # Attempt 1
    valid_questions: list[dict] = []
    seen_texts: set[str] = set()

    try:
        first_batch = _request_questions(target_count, [])
        for q in first_batch:
            key = q["question"].lower()
            if key not in seen_texts:
                seen_texts.add(key)
                valid_questions.append(q)
    except Exception as err:
        first_error = str(err)
    else:
        first_error = ""

    # Fallback retry - MAX 1 RETRY for ONLY missing questions
    if len(valid_questions) < target_count:
        missing_count = target_count - len(valid_questions)
        existing_texts = [q["question"] for q in valid_questions]
        try:
            fallback_batch = _request_questions(missing_count, existing_texts)
            for q in fallback_batch:
                key = q["question"].lower()
                if key not in seen_texts:
                    seen_texts.add(key)
                    valid_questions.append(q)
        except Exception:
            pass

    # Complete to exactly 5 using clean, easy-to-moderate undergraduate viva questions if needed
    if len(valid_questions) < target_count:
        topic_terms = concepts if len(concepts) >= 5 else (concepts + ["algorithm", "model", "parameter", "feature", "evaluation"])
        pattern_templates = [
            ("In what real-world scenario would you apply {c} in this experiment?", ["application", "scenario"]),
            ("If your model shows poor performance with {c}, what parameters or data adjustments would you inspect first?", ["troubleshooting", "tuning"]),
            ("How does changing the dataset split or hyper-parameters affect the predictions of {c}?", ["parameters", "evaluation"]),
            ("What key assumptions does {c} rely on, and what happens if those assumptions are violated?", ["assumptions", "limitations"]),
            ("How would you interpret and explain the evaluation metrics obtained for {c} in this experiment?", ["interpretation", "metrics"]),
        ]
        for idx, (tmpl, tmpl_concepts) in enumerate(pattern_templates):
            if len(valid_questions) >= target_count:
                break
            term = topic_terms[idx % len(topic_terms)].replace("_", " ")
            formatted_q = clean_question_text(tmpl.format(c=term))
            key = formatted_q.lower()
            if key not in seen_texts:
                seen_texts.add(key)
                valid_questions.append(
                    {
                        "question": formatted_q,
                        "expected_concepts": tmpl_concepts + [term],
                        "difficulty": "medium",
                    }
                )

    final_5 = valid_questions[:target_count]
    if len(final_5) < target_count:
        err_msg = f" (Ollama error: {first_error})" if first_error else ""
        raise RuntimeError(f"Unable to obtain 5 valid viva questions.{err_msg}")

    # Store in cache (Req 6 & 10)
    _EXPERIMENT_QUESTION_CACHE[cache_key] = final_5
    return final_5


def evaluate_answer(
    question: str,
    transcript: str,
    experiment_content: str,
    expected_concepts: list[str],
    max_marks: int = 10,
) -> dict:
    """
    Evaluate student's answer ONLY against current question and selected experiment material.
    Evidence-based scoring with strict post-validation rules:
    - Completely correct -> high/full marks
    - Partially correct -> proportional partial marks
    - Incorrect / unrelated / empty / repetition -> 0 marks
    """
    material = clean(experiment_content)[:4000]
    answer = clean(transcript)
    question_text = clean(question)
    concepts = [clean(str(c)) for c in (expected_concepts or []) if clean(str(c))]
    if not concepts:
        concepts = concept_terms(material, 10)

    # 1. Pre-check: Empty / whitespace answer
    if not answer or len(answer) < 2:
        return {
            "score": 0.0,
            "matched_concepts": [],
            "missing_concepts": concepts,
            "incorrect_claims": [],
            "is_correct": False,
            "feedback": "No answer was provided.",
        }

    # 2. Pre-check: Repetition of question words only or non-answer
    if _is_repetition_or_keyword_only(question_text, answer):
        return {
            "score": 0.0,
            "matched_concepts": [],
            "missing_concepts": concepts,
            "incorrect_claims": [],
            "is_correct": False,
            "feedback": "The answer merely repeats words from the question or contains no genuine explanation.",
        }

    # 3. Pre-check: Zero domain word overlap between answer and experiment material + question + expected concepts
    a_words = _extract_content_words(answer)
    exp_words = (
        _extract_content_words(material)
        | _extract_content_words(question_text)
        | _extract_content_words(" ".join(concepts))
    )
    if not a_words.intersection(exp_words):
        return {
            "score": 0.0,
            "matched_concepts": [],
            "missing_concepts": concepts,
            "incorrect_claims": [],
            "is_correct": False,
            "feedback": "The answer is completely unrelated to the question or experiment material.",
        }

    # 4. Strict LLM Evaluation Prompt
    prompt = f"""You are a strict, evidence-based university laboratory viva evaluator.

EVALUATION RULES:
1. Compare the student's answer ONLY against the specific QUESTION, EXPECTED CONCEPTS, and EXPERIMENT MATERIAL below.
2. EVIDENCE-BASED SCORING:
   - Completely correct answer demonstrating expected concepts → high/full marks (7.0 to {max_marks}).
   - Partially correct answer demonstrating SOME expected concepts with genuine explanation → proportional partial marks (1.0 to 6.0).
   - Completely wrong, incorrect, or scientifically false answer → score = 0.0.
   - Irrelevant, off-topic, or gibberish answer → score = 0.0, matched_concepts = [], is_correct = false.
   - Answer mentioning keywords without explanation when explanation is needed → score = 0.0, matched_concepts = [], is_correct = false.
   - Do NOT award marks simply because the answer is grammatically valid or contains keywords.
3. MATCHED CONCEPTS:
   - Only list a concept in "matched_concepts" if the student's answer GENUINELY demonstrates understanding of that concept.
   - If no expected concepts are genuinely demonstrated, "matched_concepts" MUST be [].
4. IS_CORRECT:
   - Set "is_correct": true ONLY if at least one expected concept is correctly demonstrated.
   - Set "is_correct": false if the answer is incorrect, irrelevant, or fails to address the question.
5. HARD RULE:
   - If "matched_concepts" is [] OR "is_correct" is false, "score" MUST BE 0.0.

QUESTION:
{question_text}

EXPECTED CONCEPTS:
{json.dumps(concepts)}

EXPERIMENT MATERIAL:
{material}

STUDENT ANSWER:
{answer}

Return STRICT JSON ONLY with exact keys:
{{
  "score": 0.0,
  "matched_concepts": ["concept1"],
  "missing_concepts": ["concept2"],
  "incorrect_claims": [],
  "is_correct": true,
  "feedback": "Evidence-based explanation of marks awarded or why 0 marks were given."
}}"""

    def _attempt_eval(attempt_idx: int) -> dict:
        temp = 0.0 if attempt_idx == 0 else 0.1
        raw = ask_ollama(prompt, temperature=temp, timeout=120)
        payload = parse_json_object(raw)
        if not payload or not isinstance(payload, dict):
            raise ValueError("Invalid JSON response from Ollama evaluation.")
        return payload

    try:
        payload = _attempt_eval(0)
    except Exception:
        try:
            payload = _attempt_eval(1)
        except Exception as exc:
            raise RuntimeError(f"Ollama answer evaluation failed: {exc}") from exc

    # 5. Post-validation & Hard Rule Enforcements
    try:
        score_val = float(payload.get("score", 0))
    except (TypeError, ValueError):
        score_val = 0.0

    is_correct_val = bool(payload.get("is_correct", False))

    raw_matched = payload.get("matched_concepts") or []
    raw_missing = payload.get("missing_concepts") or []
    raw_incorrect = payload.get("incorrect_claims") or []

    if not isinstance(raw_matched, list):
        raw_matched = [str(raw_matched)]
    if not isinstance(raw_missing, list):
        raw_missing = [str(raw_missing)]
    if not isinstance(raw_incorrect, list):
        raw_incorrect = [str(raw_incorrect)]

    matched = [clean(str(item)) for item in raw_matched if clean(str(item))]
    missing = [clean(str(item)) for item in raw_missing if clean(str(item))]
    incorrect = [clean(str(item)) for item in raw_incorrect if clean(str(item))]
    feedback_str = clean(payload.get("feedback") or "")

    # Validate matched_concepts against answer text & expected concepts/material
    validated_matched = []
    ans_lower = answer.lower()
    for item in matched:
        item_words = _extract_content_words(item)
        if item_words and item_words.intersection(a_words):
            validated_matched.append(item)
        elif any(c.lower() in item.lower() or item.lower() in c.lower() for c in concepts) and any(w in ans_lower for w in item_words):
            validated_matched.append(item)

    matched = validated_matched

    # Check feedback sentiment for rejection keywords
    rejection_keywords = (
        "irrelevant", "unrelated", "incorrect", "completely wrong",
        "does not address", "no relevant", "0 marks", "does not answer",
        "off-topic", "wrong answer", "fails to address"
    )
    fb_lower = feedback_str.lower()
    if any(kw in fb_lower for kw in rejection_keywords) and not matched:
        is_correct_val = False

    # Enforce Hard Scoring Rules (Req 4 & 6)
    if not matched or not is_correct_val or score_val <= 0.0:
        score_val = 0.0
        matched = []
        is_correct_val = False
        if not feedback_str or feedback_str == "No additional evaluation feedback was provided.":
            feedback_str = "The answer does not correctly address the question or cover any expected concepts."

    score_val = max(0.0, min(float(max_marks), round(score_val, 1)))

    # Re-calculate missing_concepts to include all expected concepts not in matched
    all_missing_set = set(missing)
    for c in concepts:
        if not any(c.lower() in m.lower() or m.lower() in c.lower() for m in matched):
            all_missing_set.add(c)
    missing = [clean(str(x)) for x in all_missing_set if clean(str(x))][:12]

    return {
        "score": score_val,
        "matched_concepts": matched[:12],
        "missing_concepts": missing,
        "incorrect_claims": incorrect[:12],
        "is_correct": is_correct_val,
        "feedback": feedback_str or "No additional evaluation feedback was provided.",
    }
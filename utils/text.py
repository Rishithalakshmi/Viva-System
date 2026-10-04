import json
import re
import unicodedata


def clean(text: str | None) -> str:
    if not text:
        return ""
    # Normalize unicode characters
    try:
        text = unicodedata.normalize("NFKC", text)
    except Exception:
        pass
    # Replace common corruptions like Na\ufffdve / Na?ve with Naive
    text = re.sub(r"na[\ufffd\?\uFFFD\xa0]ve", "Naive", text, flags=re.IGNORECASE)
    # Strip non-ascii private glyphs (like \uf0b7, \uf020, \ufffd) and unicode bullets
    text = text.replace("\ufffd", "").replace("", "")
    text = re.sub(r"[\uf000-\uf8ff]", " ", text)
    text = re.sub(r"[•·■▪►✔✕✖★\u2022\u2023\u25E6\u2043\u2219]", " ", text)
    # Acronym followed by lowercase e.g. DBSCANclustering -> DBSCAN clustering, SVMbased -> SVM based
    text = re.sub(r"([A-Z]{2,})([a-z]{2,})", r"\1 \2", text)
    # Acronym followed by TitleCase e.g. DBSCANClustering -> DBSCAN Clustering
    text = re.sub(r"([A-Z]{2,})([A-Z][a-z])", r"\1 \2", text)
    # lower/digit followed by Upper e.g. SimpleLinear -> Simple Linear, basedClassification -> based Classification
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    # Common merged words from PDF extractions
    text = re.sub(r"(?i)\bmultiplelinear\b", "Multiple Linear", text)
    text = re.sub(r"(?i)decisiontree", "Decision Tree ", text)
    text = re.sub(r"(?i)treelearning", "Tree Learning", text)
    text = re.sub(r"(?i)clusteringalgorithm", "Clustering Algorithm", text)
    text = re.sub(r"(?i)basedclassification", "Based Classification", text)
    # Insert space after colon or punctuation merged with text (e.g. Statement:Taking -> Statement: Taking)
    text = re.sub(r"([:;,])([A-Za-z])", r"\1 \2", text)
    # Normalize consecutive spaces/newlines to single space
    return re.sub(r"\s+", " ", text).strip()


def clean_multiline(text: str | None) -> str:
    if not text:
        return ""
    try:
        text = unicodedata.normalize("NFKC", text)
    except Exception:
        pass
    text = re.sub(r"na[\ufffd\?\uFFFD\xa0]ve", "Naive", text, flags=re.IGNORECASE)
    text = text.replace("\ufffd", "").replace("", "")
    text = re.sub(r"[\uf000-\uf8ff]", " ", text)
    text = re.sub(r"[•·■▪►✔✕✖★\u2022\u2023\u25E6\u2043\u2219]", " ", text)
    text = re.sub(r"([A-Z]{2,})([a-z]{2,})", r"\1 \2", text)
    text = re.sub(r"([A-Z]{2,})([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([:;,])([A-Za-z])", r"\1 \2", text)
    # Normalize whitespace per line while preserving line breaks
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    # Remove excessive consecutive empty lines (max 2)
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if not line:
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append("")
        else:
            blank_count = 0
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def format_experiment_title(raw_title: str | None, number: int) -> str:
    if not raw_title:
        return f"Experiment {number}"

    title = clean(str(raw_title))
    # Remove leading experiment markers if present in raw title
    title = re.sub(r"(?i)^(?:experiment|expt|exp|ex\.no|practical|lab\s*exercise)\s*[:#.\-—–\s\?\d]*", "", title).strip()
    # Remove trailing/leading punctuation
    title = re.sub(r"^[-:#.\s]+", "", title).strip()
    title = re.sub(r"[-:#.\s]+$", "", title).strip()
    # Normalize common abbreviations
    title = re.sub(r"(?i)\bres\s*net\b", "ResNet", title)
    title = re.sub(r"(?i)\bk\s*means\b", "k-Means", title)
    title = re.sub(r"(?i)\bk\s*medoid\b", "k-Medoid", title)
    title = re.sub(r"(?i)\bk\s*nearest\b", "k-Nearest", title)
    title = re.sub(r"(?i)\bdecision\s*tree\s*learning\b", "Decision Tree Learning", title)
    title = re.sub(r"(?i)\bsvm\s*based\s*classification\b", "SVM Based Classification", title)
    title = re.sub(r"\s+", " ", title).strip()

    if not title or title.lower() in {f"experiment {number}".lower(), f"exp {number}".lower(), "laboratory manual"}:
        return f"Experiment {number}"

    return title


def parse_json_object(text: str | None):
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            return None
    return None


def concept_terms(text: str, limit: int = 15) -> list[str]:
    cleaned = clean(text)
    raw_words = re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{2,}\b", cleaned.lower())
    stop = {
        "the", "and", "for", "with", "from", "this", "that", "using", "into", "have",
        "has", "are", "was", "were", "will", "then", "than", "their", "there", "which",
        "where", "when", "what", "your", "you", "they", "them", "can", "also", "such",
        "one", "two", "three", "step", "program", "problem", "statement", "description",
        "output", "import", "data", "aim", "theory", "procedure", "result", "conclusion",
        "experiment", "laboratory", "manual", "student", "faculty", "write", "given",
        "example", "examples", "taking", "table", "following", "none", "each", "used",
        "need", "show", "find", "make", "same", "well", "like", "call", "apply", "code",
        "python", "pandas", "numpy", "matplotlib", "sklearn", "fit", "plot", "print",
        "def", "return", "self", "class", "true", "false", "file", "line", "read",
    }
    freq: dict[str, int] = {}
    for word in raw_words:
        w = word.strip("._-")
        if w not in stop and len(w) > 3 and not w.isdigit() and len(w) < 22:
            freq[w] = freq.get(w, 0) + 1
    return [word for word, _ in sorted(freq.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def load_json_list(raw: str | None) -> list:
    try:
        data = json.loads(raw or "[]")
        return data if isinstance(data, list) else []
    except Exception:
        return []

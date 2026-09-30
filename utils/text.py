import json
import re


def clean(text: str | None) -> str:
    if not text:
        return ""
    # Separate camel-cased concatenated words from PDF extractions (e.g. SimpleLinearRegression -> Simple Linear Regression)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    # Insert space after colon or punctuation merged with text (e.g. Statement:Taking -> Statement: Taking)
    text = re.sub(r"([:;,])([A-Za-z])", r"\1 \2", text)
    # Normalize consecutive spaces/newlines to single space
    return re.sub(r"\s+", " ", text).strip()


def format_experiment_title(raw_title: str | None, number: int) -> str:
    if not raw_title:
        return f"Experiment {number}"

    normalized = re.sub(r"\s+", "", raw_title.lower())

    topic_map = [
        (r"simple.*linear.*regression", "Simple Linear Regression"),
        (r"multiple.*linear.*regression", "Multiple Linear Regression"),
        (r"k.*medoid", "K-Medoids Clustering"),
        (r"k.*means", "K-Means Clustering"),
        (r"k.*nearest|knn", "k-Nearest Neighbors (KNN)"),
        (r"decision.*tree|id3", "Decision Tree"),
        (r"naive.*bayes", "Naive Bayes Classification"),
        (r"dbscan", "DBSCAN Clustering"),
        (r"svm|support.*vector", "SVM Classification"),
        (r"principal.*component|pca", "Principal Component Analysis (PCA)"),
    ]
    for pattern, clean_name in topic_map:
        if re.search(pattern, normalized):
            return clean_name

    # Fallback cleaning
    title = re.split(
        r"(?i)\b(?:problem\s*statement|description|aim|objective)\b", raw_title
    )[0]
    title = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", title)
    title = re.sub(r"[^\w\s+#.-]", "", title)
    title = clean(title)
    if len(title) > 45:
        title = title[:45].rsplit(" ", 1)[0]
    return title.title() if title else f"Experiment {number}"


def parse_json_object(text: str | None):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.I)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    match = re.search(r"\{.*\}", text or "", re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            return None
    return None


def concept_terms(text: str, limit: int = 15) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_+#.-]{2,}", (text or "").lower())
    stop = {
        "the", "and", "for", "with", "from", "this", "that", "using", "into", "have",
        "has", "are", "was", "were", "will", "then", "than", "their", "there", "which",
        "where", "when", "what", "your", "you", "they", "them", "can", "also", "such",
        "one", "two", "three", "step", "program", "problem", "statement", "description",
        "output", "import", "data", "aim", "theory", "procedure", "result", "conclusion",
        "experiment", "laboratory", "manual", "student", "faculty", "write", "given",
    }
    freq: dict[str, int] = {}
    for word in words:
        if word not in stop and len(word) > 3:
            freq[word] = freq.get(word, 0) + 1
    return [word for word, _ in sorted(freq.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def load_json_list(raw: str | None) -> list:
    try:
        data = json.loads(raw or "[]")
        return data if isinstance(data, list) else []
    except Exception:
        return []

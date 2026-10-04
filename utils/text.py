import json
import re


def clean(text: str | None) -> str:
    if not text:
        return ""
    # Strip non-ascii private glyphs (like \uf0b7, \uf020) and unicode bullets
    text = re.sub(r"[\uf000-\uf8ff]", " ", text)
    text = re.sub(r"[•·■▪►✔✕✖★\u2022\u2023\u25E6\u2043\u2219]", " ", text)
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

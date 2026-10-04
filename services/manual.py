from __future__ import annotations

import re
from pathlib import Path

import pymupdf
from docx import Document

from utils.text import clean, clean_multiline, format_experiment_title

ROMAN_NUMERALS = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
    "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15, "xvi": 16, "xvii": 17, "xviii": 18, "xix": 19, "xx": 20,
}

EXPLICIT_EXP_HEADER_RE = re.compile(
    r"(?i)^[ \t]*(?:"
    r"(?:ex(?:pt)?\.?\s*n[o0]\.?|experiment(?:\s*(?:n[o0]\.?|number|#))?|exp(?:\.|\b)|lab(?:oratory)?\s*(?:exercise|experiment)|practical(?:\s*(?:exercise|n[o0]\.?))?)\s*[:#.\-—–\s\?\uf000-\uf8ff\ufffd]*\s*(\d+|[ivx]+)"
    r"|"
    r"(?:ex(?:pt)?\.?\s*n[o0]\.?|experiment|exp|practical)\s*(\d+|[ivx]+)"
    r")\b[ \t]*(.*)$",
    re.MULTILINE,
)

NUMBER_FALLBACK_RE = re.compile(
    r"(?i)^[ \t]*(\d+)[.\)\-:][ \t]+([A-Z][^\n]{3,90})$",
    re.MULTILINE,
)

STOP_WORDS_SECTION = re.compile(
    r"(?i)^\s*(?:problem\s*(?:statement|understanding|description)|aim\b|objective\b|title\b|description\b|theory\b|procedure\b|dataset\b|dataset\s*description|program\b|algorithm\b|code\b|apparatus\b|result\b|output\b|viva\s*questions?|overview\b|introduction\b|prelab\b|postlab\b)"
)


def extract_manual_text(path: Path, raw: bytes | None = None) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        data = raw if raw is not None else path.read_bytes()
        document = pymupdf.open(stream=data, filetype="pdf")
        try:
            # sort=True ensures correct reading order and preserves vertical text/spans
            text = "\n".join(page.get_text("text", sort=True) for page in document)
        finally:
            document.close()
        return clean_multiline(text)
    if suffix in {".docx"}:
        document = Document(str(path))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        return clean_multiline(text)
    if suffix in {".txt", ".md"}:
        data = raw if raw is not None else path.read_bytes()
        return clean_multiline(data.decode("utf-8", errors="replace"))
    raise ValueError("Upload a PDF, DOCX or TXT laboratory manual.")


def _clean_title_part(s: str) -> str:
    s = clean(s)
    # Remove leading experiment markers
    s = re.sub(
        r"(?i)^(?:experiment|expt|exp|ex\.no|practical|lab\s*exercise)\s*[:#.\-—–\s\?\d]*",
        "",
        s,
    ).strip()
    s = re.sub(r"^[-:#.\s]+", "", s).strip()
    return s


def detect_experiments(text: str) -> list[dict]:
    text = clean_multiline(text) or ""
    if not text:
        return []

    # 1. First try matching explicit experiment headers
    raw_matches = []
    for m in EXPLICIT_EXP_HEADER_RE.finditer(text):
        num_str = (m.group(1) or m.group(2) or "").strip()
        if not num_str:
            continue
        if num_str.isdigit():
            num = int(num_str)
        else:
            num = ROMAN_NUMERALS.get(num_str.lower(), None)
            if num is None:
                continue
        trailing = m.group(3) or ""
        raw_matches.append({
            "start": m.start(),
            "number": num,
            "trailing": trailing,
            "header_end": m.end(),
        })

    # If no explicit experiment headers found, fallback to numbered dot/parenthesis headers
    if not raw_matches:
        for m in NUMBER_FALLBACK_RE.finditer(text):
            trailing = (m.group(2) or "").strip()
            # Ensure trailing is a concise heading, not a narrative sentence or page footer
            if (
                trailing
                and not re.search(r"\.\s+[A-Za-z]", trailing)
                and not re.search(r"(?i)\b(?:page\s*\d+|dr\.|prof\.|dept\.|q\d+)\b", trailing)
                and len(trailing.split()) <= 12
            ):
                raw_matches.append({
                    "start": m.start(),
                    "number": int(m.group(1)),
                    "trailing": trailing,
                    "header_end": m.end(),
                })

    if not raw_matches:
        content = clean_multiline(text)
        if not content:
            return []
        return [{"number": 1, "title": "Laboratory Manual", "content": content}]

    # Filter out Table of Contents / summary table duplicates (< 100 chars between successive headings)
    valid_matches = []
    for i, m in enumerate(raw_matches):
        next_pos = raw_matches[i + 1]["start"] if i + 1 < len(raw_matches) else len(text)
        chunk_len = next_pos - m["start"]
        if chunk_len < 100 and i + 1 < len(raw_matches):
            continue
        valid_matches.append(m)

    if not valid_matches:
        valid_matches = raw_matches

    experiments = []
    used_numbers: set[int] = set()

    for idx, m in enumerate(valid_matches):
        start_pos = m["start"]
        next_start = valid_matches[idx + 1]["start"] if idx + 1 < len(valid_matches) else len(text)
        section = text[start_pos:next_start].strip()

        number = m["number"]
        if number in used_numbers:
            number = max(used_numbers) + 1
        used_numbers.add(number)

        # Title extraction: check trailing part of heading line and following lines
        title_lines = []
        trailing = _clean_title_part(m["trailing"])
        if trailing and not STOP_WORDS_SECTION.match(trailing):
            title_lines.append(trailing)

        # Inspect lines immediately following header
        lines_after = text[m["header_end"]:next_start].splitlines()
        for line in lines_after:
            line_str = line.strip()
            if not line_str:
                continue
            cleaned_line = _clean_title_part(line_str)
            if not cleaned_line:
                continue
            if STOP_WORDS_SECTION.match(cleaned_line):
                break
            # Skip page headers / footers / date / marks lines / university mission
            if re.search(r"(?i)^(?:page\s*\d+|\d+|date|signature|marks|faculty|vision|mission)\b", cleaned_line):
                continue
            # If line is code, stop title collection
            if re.search(r"^(?:import |from |def |class |plt\.|sns\.|cv2\.|np\.|#|\$\w+)", cleaned_line):
                break
            # If line is a long narrative sentence (> 110 chars), likely body text
            if len(cleaned_line) > 110:
                break
            title_lines.append(cleaned_line)
            if len(title_lines) >= 2:  # Multiline title support
                break

        # Combine title lines
        if len(title_lines) == 1:
            raw_t = title_lines[0]
        elif len(title_lines) > 1:
            if title_lines[1].startswith("-") or title_lines[0].endswith("-"):
                raw_t = f"{title_lines[0].rstrip('- ')} - {title_lines[1].lstrip('- ')}"
            else:
                raw_t = f"{title_lines[0]} {title_lines[1]}"
        else:
            raw_t = ""

        # Fallback to Aim/Objective/Title in body if title is still empty
        if not raw_t or len(raw_t) < 3:
            aim_match = re.search(r"(?i)\b(?:aim|objective|title)\s*[:.\-]\s*([^\n\.]+)", section)
            if aim_match:
                raw_t = _clean_title_part(aim_match.group(1))
            else:
                raw_t = f"Experiment {number}"

        clean_t = format_experiment_title(raw_t, number)

        experiments.append(
            {
                "number": number,
                "title": clean_t,
                "content": clean_multiline(section),
            }
        )

    experiments.sort(key=lambda item: item["number"])
    return experiments

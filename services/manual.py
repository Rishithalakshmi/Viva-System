from __future__ import annotations

import re
from pathlib import Path

import pymupdf
from docx import Document

from utils.text import clean


EXPERIMENT_HEADING = re.compile(
    r"(?i)(?:ex(?:pt)?\.?\s*no\.?|experiment(?:\s*(?:no\.?|number|#))?|"
    r"lab(?:oratory)?\s*(?:exercise|experiment)|practical)\s*[:#.\-]?\s*(\d+)\b"
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
        return clean(text)
    if suffix in {".docx"}:
        document = Document(str(path))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        return clean(text)
    if suffix in {".txt", ".md"}:
        data = raw if raw is not None else path.read_bytes()
        return clean(data.decode("utf-8", errors="replace"))
    raise ValueError("Upload a PDF, DOCX or TXT laboratory manual.")


def _title_from_section(section: str, number: int) -> str:
    heading = section.split("\n", 1)[0]
    heading = EXPERIMENT_HEADING.sub("", heading, count=1)
    heading = re.sub(r"(?i)problem statement.*", "", heading).strip(" :-.\t")
    heading = clean(heading)
    if not heading or len(heading) < 4:
        body = clean(section[:400])
        body = EXPERIMENT_HEADING.sub("", body, count=1)
        for marker in ("Aim", "AIM", "Objective", "Title"):
            if marker.lower() in body.lower():
                match = re.search(rf"(?i){marker}\s*[:.\-]?\s*(.+?)(?:\.|$)", body)
                if match:
                    heading = clean(match.group(1))[:120]
                    break
        if not heading or len(heading) < 4:
            heading = f"Experiment {number}"
    return heading[:160]


def detect_experiments(text: str) -> list[dict]:
    text = text or ""
    matches = list(EXPERIMENT_HEADING.finditer(text))
    if not matches:
        content = clean(text)
        if not content:
            return []
        return [{"number": 1, "title": "Laboratory Manual", "content": content}]

    experiments = []
    used_numbers: set[int] = set()
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[match.start():end].strip()
        number = int(match.group(1))
        if number in used_numbers:
            number = max(used_numbers) + 1
        used_numbers.add(number)
        experiments.append(
            {
                "number": number,
                "title": _title_from_section(section, number),
                "content": clean(section),
            }
        )
    experiments.sort(key=lambda item: item["number"])
    return experiments

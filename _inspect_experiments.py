from services.manual import detect_experiments, extract_manual_text
from pathlib import Path
import re

for p in [Path("data/manuals/20260922_185142_ml_lab_manual.pdf"), Path("data/manuals/20261004_145758_Deep_Learning_-_Self_Curated_-_Lab_Manual.pdf")]:
    text = extract_manual_text(p)
    exps = detect_experiments(text)
    print(f"=== {p.name} ({len(exps)} experiments) ===")
    for e in exps:
        print(f"\n--- Exp {e['number']}: {e['title']} ---")
        lines = [line.strip() for line in e['content'].splitlines() if line.strip()]
        # Check if manual has embedded viva questions
        viva_section = False
        viva_lines = []
        for line in lines:
            if re.search(r'(?i)\bviva\s*questions?', line):
                viva_section = True
            if viva_section:
                viva_lines.append(line)
        if viva_lines:
            print("  Found Embedded Viva Section in manual:")
            for vl in viva_lines[:8]:
                print(f"    {vl}")
        else:
            print("  No embedded viva section. First 3 non-empty lines:")
            for l in lines[:3]:
                print(f"    {l}")

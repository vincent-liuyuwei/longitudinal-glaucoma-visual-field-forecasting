#!/usr/bin/env python3
"""Extract thesis structure and likely repeated prose from a PDF."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from pypdf import PdfReader


PDF = Path("./thesis.pdf")
OUT = Path("./thesis_audit")


def normalize(text: str) -> str:
    text = text.lower().replace("–", "-").replace("—", "-")
    text = re.sub(r"\b(table|figure)\s+\d+(?:\.\d+)*", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .\n\t")


def paragraphs(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", text)
    output = []
    for block in blocks:
        block = re.sub(r"\s*\n\s*", " ", block).strip()
        if len(normalize(block)) >= 140:
            output.append(block)
    return output


def heading_candidates(text: str) -> list[str]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    candidates = []
    patterns = (
        r"^(chapter\s+\d+)$",
        r"^\d+(?:\.\d+){0,3}\s+.{3,100}$",
        r"^(abstract|acknowledgements|contents|bibliography|references|conclusion)$",
    )
    for line in lines:
        if any(re.match(pattern, line, re.I) for pattern in patterns):
            candidates.append(line)
    return candidates[:15]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    page_dir = OUT / "pages"
    page_dir.mkdir(exist_ok=True)

    reader = PdfReader(str(PDF))
    pages = []
    all_paragraphs: list[tuple[int, str, str]] = []
    outline = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        (page_dir / f"page_{page_number:03d}.txt").write_text(text, encoding="utf-8")
        headings = heading_candidates(text)
        if headings:
            outline.append({"page": page_number, "headings": headings})
        page_paragraphs = paragraphs(text)
        for paragraph in page_paragraphs:
            all_paragraphs.append((page_number, paragraph, normalize(paragraph)))
        pages.append(
            {
                "page": page_number,
                "characters": len(text),
                "paragraphs": len(page_paragraphs),
                "headings": headings,
            }
        )

    exact = defaultdict(list)
    for page, paragraph, norm in all_paragraphs:
        exact[norm].append((page, paragraph))
    exact_repeats = [
        {"pages": sorted({p for p, _ in values}), "text": values[0][1]}
        for values in exact.values()
        if len({p for p, _ in values}) > 1
    ]

    near = []
    for i, (page_a, para_a, norm_a) in enumerate(all_paragraphs):
        for page_b, para_b, norm_b in all_paragraphs[i + 1 :]:
            if abs(page_a - page_b) < 2:
                continue
            length_ratio = min(len(norm_a), len(norm_b)) / max(len(norm_a), len(norm_b))
            if length_ratio < 0.7:
                continue
            ratio = SequenceMatcher(None, norm_a, norm_b, autojunk=True).ratio()
            if ratio >= 0.78:
                near.append(
                    {
                        "page_a": page_a,
                        "page_b": page_b,
                        "similarity": round(ratio, 3),
                        "text_a": para_a,
                        "text_b": para_b,
                    }
                )
    near.sort(key=lambda item: item["similarity"], reverse=True)

    payload = {
        "pdf": str(PDF),
        "page_count": len(reader.pages),
        "pages": pages,
        "outline": outline,
        "exact_repeats": exact_repeats,
        "near_repeats": near[:80],
    }
    (OUT / "audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = [f"Pages: {len(reader.pages)}", "", "OUTLINE"]
    for item in outline:
        report.append(f"PDF {item['page']}: " + " | ".join(item["headings"]))
    report.extend(["", "EXACT REPEATS"])
    for item in exact_repeats:
        report.append(f"Pages {item['pages']}: {item['text'][:500]}")
    report.extend(["", "NEAR REPEATS"])
    for item in near[:40]:
        report.append(
            f"Pages {item['page_a']}/{item['page_b']} ({item['similarity']}): "
            f"{item['text_a'][:260]} || {item['text_b'][:260]}"
        )
    (OUT / "report.txt").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()

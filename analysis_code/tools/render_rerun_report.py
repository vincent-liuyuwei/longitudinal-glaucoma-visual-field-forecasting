"""Render rerun status and result summaries for local visual inspection."""

from __future__ import annotations

import csv
import textwrap
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(".")
OUTPUT_DIR = Path("/private/tmp/glaucoma_rerun_report")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _read_text(path: Path, tail: int | None = None) -> str:
    if not path.exists():
        return f"[MISSING] {path}"
    text = path.read_text(encoding="utf-8", errors="replace")
    if tail is not None:
        lines = text.splitlines()
        text = "\n".join(lines[-tail:])
    return text


def _csv_preview(path: Path, max_rows: int = 25) -> str:
    if not path.exists():
        return f"[MISSING] {path}"
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.reader(handle))
    preview = rows[: max_rows + 1]
    return "\n".join(" | ".join(row) for row in preview)


def _file_info(path: Path) -> str:
    if not path.exists():
        return f"[MISSING] {path}"
    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
    return f"{path}\nsize={stat.st_size:,} bytes; modified={modified}"


def _font(size: int):
    candidates = (
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.dfont",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _wrap_lines(text: str, width: int = 130) -> list[str]:
    output: list[str] = []
    for line in text.splitlines():
        if not line:
            output.append("")
            continue
        output.extend(
            textwrap.wrap(
                line,
                width=width,
                replace_whitespace=False,
                drop_whitespace=False,
            )
            or [""]
        )
    return output


def _render_pages(text: str) -> None:
    font = _font(20)
    width, height = 1800, 2200
    margin = 45
    line_height = 29
    lines_per_page = (height - 2 * margin) // line_height
    lines = _wrap_lines(text)

    for page_index, start in enumerate(range(0, len(lines), lines_per_page), 1):
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        page_lines = lines[start : start + lines_per_page]
        for row, line in enumerate(page_lines):
            draw.text((margin, margin + row * line_height), line, fill="black", font=font)
        image.save(OUTPUT_DIR / f"report_{page_index}.png")

    (OUTPUT_DIR / "page_count.txt").write_text(
        str(max(1, (len(lines) + lines_per_page - 1) // lines_per_page)),
        encoding="utf-8",
    )


def main() -> None:
    rerun = ROOT / "results" / "rerun_20260730"
    results = ROOT / "results" / "confirmatory_subgroups"
    key_files = (
        results / "RESULTS_SUMMARY.md",
        results / "eye_balanced_metrics_bootstrap_ci.csv",
        results / "paired_deltas_bootstrap_ci.csv",
        results / "eye_strata.csv",
        results / "image_rotated" / "rotation_augmentation_audit.csv",
        ROOT / "paper" / "glaucoma_multimodal_forecasting_draft_v1.md",
    )

    sections = [
        "=== RERUN STATUS ===",
        _read_text(rerun / "status.txt"),
        f"exit_code={_read_text(rerun / 'exit_code.txt')}",
        "",
        "=== KEY FILE METADATA ===",
        "\n\n".join(_file_info(path) for path in key_files),
        "",
        "=== RESULTS SUMMARY ===",
        _read_text(results / "RESULTS_SUMMARY.md"),
        "",
        "=== EYE-BALANCED METRICS PREVIEW ===",
        _csv_preview(results / "eye_balanced_metrics_bootstrap_ci.csv", 40),
        "",
        "=== PAIRED DELTAS PREVIEW ===",
        _csv_preview(results / "paired_deltas_bootstrap_ci.csv", 50),
        "",
        "=== EYE STRATA PREVIEW ===",
        _csv_preview(results / "eye_strata.csv", 35),
        "",
        "=== ROTATION AUDIT PREVIEW ===",
        _csv_preview(
            results / "image_rotated" / "rotation_augmentation_audit.csv",
            35,
        ),
        "",
        "=== PIPELINE LOG TAIL ===",
        _read_text(rerun / "pipeline.log", tail=100),
    ]
    _render_pages("\n".join(sections))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Remove obvious padding sections from Course I (01-08) articles.

Heuristic rules:
- Delete from heading "## 追加: 自己診断と落とし穴メモ" up to (but not including) the next "## 参考文献".
- In lec08 part1, delete "### 10問チェック" and "### 追加ドリル（式の形だけ）" blocks (up to next heading at same or higher level).
- In lec03/07 part2, delete the large drill blocks starting at those headings:
  - lec03 part2: "## 自己診断（短問）" (we keep minimal summary sections above)
  - lec07 part2: "## 自己診断（短問）"

This intentionally does NOT try to "rewrite" content; only removes water-filling blocks.
"""

from __future__ import annotations

from pathlib import Path


ART = Path("~/Desktop/blog/Zenn/articles").expanduser()


def read_lines(p: Path) -> list[str]:
    return p.read_text(encoding="utf-8").splitlines()


def write_lines(p: Path, lines: list[str]) -> None:
    p.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def delete_range(lines: list[str], start: int, end: int) -> list[str]:
    return lines[:start] + lines[end:]


def find_heading(lines: list[str], heading: str) -> int | None:
    for i, ln in enumerate(lines):
        if ln.strip() == heading:
            return i
    return None


def remove_additional_section(lines: list[str]) -> list[str]:
    s = find_heading(lines, "## 追加: 自己診断と落とし穴メモ")
    if s is None:
        return lines
    # find next "## 参考文献" after s; if missing, delete until end.
    e = len(lines)
    for j in range(s + 1, len(lines)):
        if lines[j].strip() == "## 参考文献":
            e = j
            break
    # also remove surrounding blank lines (at most 2)
    while s > 0 and lines[s - 1].strip() == "":
        s -= 1
    while e < len(lines) and lines[e].strip() == "":
        e += 1
    return delete_range(lines, s, e)


def remove_block_from_heading_to_next(lines: list[str], heading: str, next_headings: list[str]) -> list[str]:
    s = find_heading(lines, heading)
    if s is None:
        return lines
    e = len(lines)
    for j in range(s + 1, len(lines)):
        if lines[j].startswith("## "):
            e = j
            break
        if lines[j].strip() in next_headings:
            e = j
            break
    while s > 0 and lines[s - 1].strip() == "":
        s -= 1
    while e < len(lines) and lines[e].strip() == "":
        e += 1
    return delete_range(lines, s, e)


def remove_selfcheck_section(lines: list[str]) -> list[str]:
    # Delete from "## 自己診断（短問）" up to "## 参考文献" (or end).
    s = find_heading(lines, "## 自己診断（短問）")
    if s is None:
        return lines
    e = len(lines)
    for j in range(s + 1, len(lines)):
        if lines[j].strip() == "## 参考文献":
            e = j
            break
    while s > 0 and lines[s - 1].strip() == "":
        s -= 1
    while e < len(lines) and lines[e].strip() == "":
        e += 1
    return delete_range(lines, s, e)


def main() -> None:
    files = []
    for n in range(1, 9):
        for part in (1, 2):
            p = ART / f"ml-lecture-{n:02d}-part{part}.md"
            if p.exists():
                files.append(p)

    for p in files:
        orig = read_lines(p)
        lines = orig

        # Global padding section
        lines = remove_additional_section(lines)

        # Targeted drill removals
        if p.name == "ml-lecture-08-part1.md":
            lines = remove_block_from_heading_to_next(lines, "### 10問チェック", [])
            lines = remove_block_from_heading_to_next(lines, "### 追加ドリル（式の形だけ）", [])

        if p.name in ("ml-lecture-03-part2.md", "ml-lecture-07-part2.md", "ml-lecture-08-part2.md"):
            lines = remove_selfcheck_section(lines)

        if lines != orig:
            write_lines(p, lines)


if __name__ == "__main__":
    main()


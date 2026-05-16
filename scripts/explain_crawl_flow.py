#!/usr/bin/env python3
"""Print a simple explanation of URL crawl flow in setup_and_crawl.py."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any, Dict, Optional


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _get_default_for_arg(func: ast.FunctionDef, arg_name: str) -> Optional[Any]:
    args = func.args.args
    defaults = func.args.defaults
    if not args:
        return None

    start = len(args) - len(defaults)
    for idx, arg in enumerate(args):
        if arg.arg != arg_name:
            continue
        if idx < start:
            return None
        default_node = defaults[idx - start]
        return _literal(default_node)
    return None


def _collect_defaults(py_file: Path) -> Dict[str, Dict[str, Any]]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    wanted = {
        "crawl_and_save": ["enable_video", "enable_asr", "keep_artifacts", "asr_max_seconds", "asr_language"],
        "run_crawl": ["enable_video", "enable_asr"],
        "crawl_urls": ["enable_video", "enable_asr"],
    }
    out: Dict[str, Dict[str, Any]] = {}

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in wanted:
            continue
        out[node.name] = {}
        for arg in wanted[node.name]:
            out[node.name][arg] = _get_default_for_arg(node, arg)
    return out


def _bool_text(value: Any) -> str:
    if value is True:
        return "TRUE"
    if value is False:
        return "FALSE"
    return f"{value!r}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain crawl/video flow simply")
    parser.add_argument("--file", default="setup_and_crawl.py", help="Path to setup_and_crawl.py")
    args = parser.parse_args()

    target = Path(args.file)
    if not target.exists():
        raise SystemExit(f"[ERROR] File not found: {target}")

    defaults = _collect_defaults(target)
    c = defaults.get("crawl_and_save", {})

    print("=== FLOW CRAWL TU URL (ban de hieu) ===")
    print("Input: website URL")
    print("\n1) Text pipeline (luon chay)")
    print("   - Trafilatura extract text")
    print("   - Neu text ngan -> fallback Selenium")
    print("   - Neu text dat nguong -> segment + luu file text")
    print("\n2) Video pipeline (chi chay khi enable_video = True)")
    print("   - Detect video tu HTML trang")
    print("   - Chia 2 nhom:")
    print("     A. Video YouTube (co youtube_id)")
    print("        + Uu tien transcript tu youtube_transcript_api")
    print("        + Neu thieu metadata/transcript -> yt-dlp fallback")
    print("        + transcript_source: youtube_transcript_api / yt_dlp_caption / none")
    print("     B. Video native (khong can YouTube, format nao cung co the thu)")
    print("        + yt-dlp dump thong tin tu page")
    print("        + Chon stream URL tot nhat (mp4/m3u8/...)" )
    print("        + Neu enable_asr=True -> chay ASR (faster-whisper)")
    print("        + transcript_source: asr_ephemeral neu ASR thanh cong")

    print("\n=== MAC DINH HIEN TAI (doc truc tiep tu code) ===")
    print(f"crawl_and_save.enable_video = {_bool_text(c.get('enable_video'))}")
    print(f"crawl_and_save.enable_asr   = {_bool_text(c.get('enable_asr'))}")
    print(f"crawl_and_save.keep_artifacts = {_bool_text(c.get('keep_artifacts'))}")
    print(f"crawl_and_save.asr_max_seconds = {c.get('asr_max_seconds')!r}")
    print(f"crawl_and_save.asr_language    = {c.get('asr_language')!r}")

    print("\nKet luan nhanh:")
    if c.get("enable_video") is True:
        print("- Dung, hien tai enable_video mac dinh la TRUE.")
    else:
        print("- Khong, enable_video khong mac dinh TRUE (kiem tra lai code).")


if __name__ == "__main__":
    main()

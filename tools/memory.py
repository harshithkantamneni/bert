"""6-op memory tool for bert-lab.

Anthropic memory_20250818 protocol with calibration tweaks for qwen3-class pilots:
- helpful errors on view/missing (lists nearby files)
- path sandboxing (no .. traversal)
- delete archives to history/ with breadcrumb (never destroys)
- atomic create via tmp + os.replace
"""
from __future__ import annotations

import datetime as dt
import os
import shutil
from pathlib import Path


class Memory:
    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # --- internal helpers --------------------------------------------------

    def _resolve(self, rel: str) -> Path:
        if ".." in Path(rel).parts:
            raise ValueError(f"path traversal not allowed: {rel}")
        target = (self.root / rel).resolve()
        try:
            target.relative_to(self.root)
        except ValueError:
            raise ValueError(f"path escapes sandbox root: {rel}") from None
        return target

    def _nearby_files(self, rel: str, limit: int = 8) -> list[str]:
        parent = self._resolve(rel).parent
        if not parent.exists():
            return []
        names = sorted(p.name for p in parent.iterdir() if p.is_file())
        return names[:limit]

    # --- public API --------------------------------------------------------

    def view(self, path: str, view_range: list[int] | None = None) -> str:
        target = self._resolve(path)
        if not target.exists():
            nearby = self._nearby_files(path)
            raise FileNotFoundError(
                f"file not found: {path}. Nearby files: {nearby}"
            )
        if target.is_dir():
            return "\n".join(sorted(p.name for p in target.iterdir()))
        text = target.read_text()
        if view_range is None:
            return text
        start, end = view_range
        lines = text.splitlines()
        return "\n".join(lines[start - 1:end])

    def create(self, path: str, file_text: str) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(file_text)
        os.replace(tmp, target)

    def str_replace(self, path: str, old_str: str, new_str: str) -> None:
        target = self._resolve(path)
        text = target.read_text()
        count = text.count(old_str)
        if count == 0:
            raise ValueError(f"old_str not found in {path}")
        if count > 1:
            raise ValueError(f"old_str not unique in {path} (found {count} times)")
        new_text = text.replace(old_str, new_str, 1)
        self.create(path, new_text)

    def insert(self, path: str, line_num: int, text: str) -> None:
        target = self._resolve(path)
        existing = target.read_text() if target.exists() else ""
        had_trailing_nl = existing.endswith("\n")
        lines = existing.splitlines()
        lines.insert(line_num, text)
        new_text = "\n".join(lines)
        if had_trailing_nl:
            new_text += "\n"
        self.create(path, new_text)

    def delete(self, path: str) -> None:
        target = self._resolve(path)
        if not target.exists():
            return
        date = dt.datetime.utcnow().strftime("%Y-%m-%d")
        archive_dir = self.root / "history" / date
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / target.name
        if archive_path.exists():
            ts = dt.datetime.utcnow().strftime("%H%M%S")
            archive_path = archive_dir / f"{target.stem}-{ts}{target.suffix}"
        shutil.move(str(target), str(archive_path))

    def rename(self, old_path: str, new_path: str) -> None:
        src = self._resolve(old_path)
        dst = self._resolve(new_path)
        if dst.exists():
            raise ValueError(f"destination exists: {new_path}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)

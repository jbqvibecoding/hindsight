"""MemOS adapter — MemCube portability (dump/load) for backup & transfer.

MemOS's standout is the ``GeneralMemCube``: a portable container that bundles a
bank's memory (textual + activation KV + parametric LoRA) and can ``dump()`` to
disk and ``load()`` back — ideal for backup/transfer. We use it to export a
bank as a self-describing envelope so Hermes ``backup_paths()`` can carry the
unified memory across a backup/import cycle.

When ``memos`` is importable we wrap ``GeneralMemCube``; otherwise we export a
lean JSON envelope of the verbatim substrate, which is enough to fully
reconstitute the bank (md is the source of truth, everything else derives).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

from .base import UnifiedAdapter

logger = logging.getLogger(__name__)


class MemosPortabilityAdapter(UnifiedAdapter):
    def __init__(self) -> None:
        self._available = False
        self._have_memos = False

    @property
    def name(self) -> str:
        return "memos"

    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        self._available = True
        try:
            from memos import GeneralMemCube  # type: ignore  # noqa: F401

            self._have_memos = True
            logger.info("memos present; MemCube export active (native)")
        except Exception:  # noqa: BLE001
            logger.info("memos absent; portable-envelope export active (lean fallback)")

    def export(self, bank: str, bank_dir: Path) -> Path | None:
        """Write a portable export envelope; return its path (or None).

        The envelope references the verbatim md substrate as truth and records
        the bank identity + a manifest. A native MemCube dump is added when the
        ``memos`` package is available.
        """
        try:
            export_dir = bank_dir / "export"
            export_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "schema": "unified-memory/portable-envelope/v1",
                "bank": bank,
                "exported_at": time.time(),
                "substrate": {
                    "log_dir": str((bank_dir / "log").resolve()),
                    "index": str((bank_dir / "index.jsonl").resolve()),
                },
                "memcube_native": self._have_memos,
            }
            if self._have_memos:
                manifest["memcube"] = self._dump_memcube(bank, export_dir)
            out = export_dir / "manifest.json"
            self._atomic_write_json(out, manifest)
            return out
        except Exception as e:  # noqa: BLE001
            logger.debug("memos export failed (non-fatal): %s", e)
            return None

    def _dump_memcube(self, bank: str, export_dir: Path) -> str | None:
        try:
            from memos import GeneralMemCube  # type: ignore

            cube = (
                GeneralMemCube(config=None) if _accepts_config(GeneralMemCube) else GeneralMemCube()
            )
            cube_dir = export_dir / "memcube"
            cube_dir.mkdir(parents=True, exist_ok=True)
            cube.dump(str(cube_dir))  # type: ignore[attr-defined]
            return str(cube_dir.resolve())
        except Exception as e:  # noqa: BLE001
            logger.debug("native MemCube dump skipped: %s", e)
            return None

    @staticmethod
    def _atomic_write_json(path: Path, data: dict) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".manifest-", suffix=".tmp")
        try:
            with open(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def export_paths(self, bank: str, bank_dir: Path) -> list[str]:
        export_dir = bank_dir / "export"
        return [str(export_dir)] if export_dir.exists() else []


def _accepts_config(cls: type) -> bool:
    try:
        import inspect

        return "config" in inspect.signature(cls.__init__).parameters
    except (ValueError, TypeError):
        return False

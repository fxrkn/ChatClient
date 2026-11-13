"""Filesystem locations for config, data and logs.

Set the ``CHATCLIENT_HOME`` environment variable to keep everything in a single
directory (useful for tests and portable installs).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import platformdirs

APP_NAME = "ChatClient"


@dataclass(frozen=True)
class AppPaths:
    config_dir: Path
    data_dir: Path
    log_dir: Path

    @property
    def config_file(self) -> Path:
        return self.config_dir / "config.toml"

    @property
    def key_file(self) -> Path:
        """API keys, used only when no OS keychain is available (owner-only permissions)."""
        return self.config_dir / "keys.json"

    @property
    def db_file(self) -> Path:
        return self.data_dir / "chatclient.db"

    @property
    def history_file(self) -> Path:
        return self.data_dir / "input_history.txt"

    @property
    def log_file(self) -> Path:
        return self.log_dir / "chatclient.log"

    def ensure(self) -> None:
        for d in (self.config_dir, self.data_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)


PORTABLE_MARKER = "portable"
PORTABLE_DIR = "ChatClientData"


def portable_home() -> Path | None:
    """Portable mode: a file named ``portable`` next to the install (the venv, or the
    folder holding it) keeps all data in ``ChatClientData`` beside it (e.g. on a USB
    stick). Otherwise data goes to the user's data folder, safe from reinstalls."""
    prefix = Path(sys.prefix)
    for folder in (prefix, prefix.parent):
        if (folder / PORTABLE_MARKER).is_file():
            return folder / PORTABLE_DIR
    return None


def resolve_paths(home: str | os.PathLike[str] | None = None) -> AppPaths:
    home = home or os.environ.get("CHATCLIENT_HOME") or portable_home()
    if home:
        root = Path(home).expanduser().resolve()
        return AppPaths(config_dir=root, data_dir=root, log_dir=root / "logs")
    return AppPaths(
        config_dir=Path(platformdirs.user_config_dir(APP_NAME, appauthor=False)),
        data_dir=Path(platformdirs.user_data_dir(APP_NAME, appauthor=False)),
        log_dir=Path(platformdirs.user_log_dir(APP_NAME, appauthor=False)),
    )

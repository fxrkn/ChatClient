"""Filesystem locations for config, data and logs.

Set the ``CHATCLIENT_HOME`` environment variable to keep everything in a single
directory (useful for tests and portable installs).
"""

from __future__ import annotations

import os
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


def resolve_paths(home: str | os.PathLike[str] | None = None) -> AppPaths:
    home = home or os.environ.get("CHATCLIENT_HOME")
    if home:
        root = Path(home).expanduser().resolve()
        return AppPaths(config_dir=root, data_dir=root, log_dir=root / "logs")
    return AppPaths(
        config_dir=Path(platformdirs.user_config_dir(APP_NAME, appauthor=False)),
        data_dir=Path(platformdirs.user_data_dir(APP_NAME, appauthor=False)),
        log_dir=Path(platformdirs.user_log_dir(APP_NAME, appauthor=False)),
    )

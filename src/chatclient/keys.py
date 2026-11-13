"""API key storage and lookup.

Lookup order for a provider's key:
  1. environment variable (``api_key_env`` in config): explicit, wins over everything
  2. OS keychain (Windows Credential Manager, macOS Keychain, GNOME Keyring/KWallet)
  3. ChatClient's own key file (only used when no keychain is available, e.g. Linux servers);
     readable by the current user only
  4. ``api_key`` written directly in config.toml (discouraged)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from chatclient.config import KEYRING_SERVICE, ProviderConfig

log = logging.getLogger(__name__)

SOURCE_ENV = "env"
SOURCE_KEYCHAIN = "keychain"
SOURCE_FILE = "file"
SOURCE_CONFIG = "config"


class KeyStore:
    def __init__(self, file: Path | None = None):
        self.file = file

    # ── lookup ──────────────────────────────────────────────────────────────

    def lookup(self, provider: str, pc: ProviderConfig) -> tuple[str | None, str | None]:
        """Return (key, source) where source is one of the SOURCE_* names."""
        if pc.api_key_env and (val := os.environ.get(pc.api_key_env, "").strip()):
            return val, SOURCE_ENV
        if val := self._keychain_get(provider):
            return val, SOURCE_KEYCHAIN
        if val := self._file_read().get(provider):
            return val, SOURCE_FILE
        if pc.api_key:
            return pc.api_key.strip(), SOURCE_CONFIG
        return None, None

    def get(self, provider: str, pc: ProviderConfig) -> str | None:
        return self.lookup(provider, pc)[0]

    # ── changes ─────────────────────────────────────────────────────────────

    def save(self, provider: str, key: str) -> str:
        """Store in the OS keychain, or in the key file when no keychain works."""
        key = key.strip()
        try:
            import keyring

            keyring.set_password(KEYRING_SERVICE, provider, key)
            if keyring.get_password(KEYRING_SERVICE, provider) == key:
                self._file_remove(provider)  # don't leave a stale copy behind
                return SOURCE_KEYCHAIN
        except Exception as e:
            log.info("keychain unavailable (%s); using the key file", e)
        if self.file is None:
            raise RuntimeError("no OS keychain available and no key file configured")
        data = self._file_read()
        data[provider] = key
        self._file_write(data)
        return SOURCE_FILE

    def remove(self, provider: str) -> bool:
        removed = False
        try:
            import keyring

            if keyring.get_password(KEYRING_SERVICE, provider):
                keyring.delete_password(KEYRING_SERVICE, provider)
                removed = True
        except Exception as e:
            log.debug("keychain delete failed for %s: %s", provider, e)
        return self._file_remove(provider) or removed

    # ── backends ────────────────────────────────────────────────────────────

    @staticmethod
    def _keychain_get(provider: str) -> str | None:
        try:
            import keyring

            val = keyring.get_password(KEYRING_SERVICE, provider)
            return val.strip() if val else None
        except Exception as e:  # no backend, locked keychain, D-Bus missing...
            log.debug("keychain lookup failed for %s: %s", provider, e)
            return None

    def _file_read(self) -> dict[str, str]:
        if self.file is None or not self.file.is_file():
            return {}
        try:
            data = json.loads(self.file.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
        except (OSError, ValueError) as e:
            log.warning("could not read key file %s: %s", self.file, e)
            return {}

    def _file_write(self, data: dict[str, str]) -> None:
        assert self.file is not None
        self.file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        # Create with owner-only permissions from the start (no window where it's readable).
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.file)
        if os.name != "nt":
            os.chmod(self.file, 0o600)

    def _file_remove(self, provider: str) -> bool:
        data = self._file_read()
        if provider not in data:
            return False
        del data[provider]
        if data:
            self._file_write(data)
        elif self.file is not None:
            self.file.unlink(missing_ok=True)
        return True


def mask(key: str) -> str:
    """'sk-or-v1-abcdef…wxyz' style preview that never reveals the key."""
    if len(key) <= 12:
        return "•" * len(key)
    return f"{key[:9]}…{key[-4:]}"

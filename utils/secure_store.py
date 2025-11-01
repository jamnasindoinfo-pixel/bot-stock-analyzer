"""Utility helpers for storing API credentials in encrypted form."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


DEFAULT_SALT_SIZE = 16
DEFAULT_ITERATIONS = 390_000


class CredentialDecryptionError(Exception):
    """Raised when stored credentials cannot be decrypted with the given passphrase."""


@dataclass
class SecureCredentialStore:
    """Simple encrypted storage backed by a single .enc JSON file."""

    location: Path

    def exists(self) -> bool:
        return self.location.exists()

    def _derive_key(self, passphrase: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=DEFAULT_ITERATIONS,
        )
        key = kdf.derive(passphrase.encode("utf-8"))
        return base64.urlsafe_b64encode(key)

    def _build_fernet(self, passphrase: str, salt: bytes) -> Fernet:
        return Fernet(self._derive_key(passphrase, salt))

    def save(self, data: Dict[str, str], passphrase: str) -> None:
        payload = json.dumps(data).encode("utf-8")
        salt = os.urandom(DEFAULT_SALT_SIZE)
        token = self._build_fernet(passphrase, salt).encrypt(payload)
        self.location.parent.mkdir(parents=True, exist_ok=True)
        with self.location.open("wb") as f:
            f.write(salt + token)

    def load(self, passphrase: str) -> Dict[str, str]:
        raw = self.location.read_bytes()
        if len(raw) <= DEFAULT_SALT_SIZE:
            raise CredentialDecryptionError("Credential file corrupt.")
        salt, token = raw[:DEFAULT_SALT_SIZE], raw[DEFAULT_SALT_SIZE:]
        try:
            decrypted = self._build_fernet(passphrase, salt).decrypt(token)
        except InvalidToken as exc:
            raise CredentialDecryptionError("Passphrase salah atau data rusak.") from exc
        return json.loads(decrypted.decode("utf-8"))


def sanitize_key_name(name: str) -> str:
    return name.strip().upper()


def filter_credentials(data: Dict[str, str]) -> Dict[str, str]:
    valid: Dict[str, str] = {}
    for key, value in data.items():
        cleaned = (value or "").strip()
        if cleaned:
            valid[sanitize_key_name(key)] = cleaned
    return valid

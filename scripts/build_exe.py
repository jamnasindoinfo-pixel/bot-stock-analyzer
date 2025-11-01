"""Helper script to bundle the CLI into an encrypted Windows executable."""

from __future__ import annotations

import secrets
import subprocess
import sys
from pathlib import Path

from packaging import version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRY_SCRIPT = PROJECT_ROOT / "scripts" / "stock_signal_cli.py"


def ensure_pyinstaller() -> "module":
    try:
        import PyInstaller  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import check only
        raise SystemExit(
            "PyInstaller belum terpasang. Jalankan 'pip install pyinstaller' terlebih dahulu."
        ) from exc
    return PyInstaller


def build() -> None:
    pyinstaller_module = ensure_pyinstaller()
    pyinstaller_version = version.parse(pyinstaller_module.__version__)

    dist_dir = PROJECT_ROOT / "dist"
    build_dir = PROJECT_ROOT / "build"
    spec_path = PROJECT_ROOT / "bot_forexmt5.spec"

    encryption_supported = pyinstaller_version < version.parse("6.0.0")
    encryption_key = secrets.token_hex(16) if encryption_supported else None

    add_data = [
        f"{PROJECT_ROOT / 'config.json'};.",
        f"{PROJECT_ROOT / 'requirements.txt'};.",
        f"{PROJECT_ROOT / 'strategy_profiles'};strategy_profiles"
        if (PROJECT_ROOT / 'strategy_profiles').exists()
        else None,
    ]

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--onefile",
        "--noconfirm",
        "--name",
        "BotForexMT5",
        "--paths",
        str(PROJECT_ROOT),
    ]

    if encryption_supported and encryption_key:
        cmd.extend(["--key", encryption_key])
    else:
        print(
            "[PERINGATAN] PyInstaller >= 6.0 tidak lagi mendukung opsi --key. "
            "Bytecode tidak dienkripsi. Jika tetap ingin fitur ini, instal PyInstaller<6.0"
        )

    cmd.extend(["--add-data", add_data[0], "--add-data", add_data[1]])

    if add_data[2]:
        cmd.extend(["--add-data", add_data[2]])

    hidden_imports = [
        "analyzers.market_analyzer",
        "bot.bot_manager",
        "trading.trade_manager",
        "scripts.stock_signal_cli",
    ]
    for mod in hidden_imports:
        cmd.extend(["--hidden-import", mod])

    cmd.append(str(ENTRY_SCRIPT))

    print("Menjalankan PyInstaller dengan perintah:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    print("\nSelesai! File exe dapat ditemukan di:", dist_dir / "BotForexMT5.exe")
    if encryption_key:
        print("Simpan juga kunci enkripsi PyInstaller berikut jika ingin rebuild:", encryption_key)
    if spec_path.exists():
        spec_path.unlink()
    if build_dir.exists():
        print("Folder build sementara berada di:", build_dir)


if __name__ == "__main__":
    build()

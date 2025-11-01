# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['D:\\Codding\\Project\\Bot-ForexMT5\\scripts\\stock_signal_cli.py'],
    pathex=['D:\\Codding\\Project\\Bot-ForexMT5'],
    binaries=[],
    datas=[('D:\\Codding\\Project\\Bot-ForexMT5\\config.json', '.'), ('D:\\Codding\\Project\\Bot-ForexMT5\\requirements.txt', '.')],
    hiddenimports=['analyzers.market_analyzer', 'bot.bot_manager', 'trading.trade_manager', 'scripts.stock_signal_cli'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='BotForexMT5',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

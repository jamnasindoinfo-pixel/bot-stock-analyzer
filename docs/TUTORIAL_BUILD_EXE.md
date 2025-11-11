# Panduan Membuat & Menjalankan BotForexMT5 versi `.exe`

## 1. Persiapan Lingkungan

1. Pastikan Python 3.10+ sudah terpasang di Windows.
2. Buka terminal PowerShell, lalu buat dan aktifkan virtual environment (opsional namun disarankan):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. Instal dependensi proyek (termasuk `cryptography`):
   ```powershell
   pip install -r requirements.txt
   ```
4. Instal PyInstaller. Jika Anda ingin enkripsi bytecode (`--key`), gunakan versi 5.x:
   ```powershell
   pip install "pyinstaller<6"
   ```
   Jika Anda tidak membutuhkan enkripsi, PyInstaller versi terbaru juga dapat digunakan.

## 2. Membangun `.exe`

Jalankan skrip pembungkus yang sudah disiapkan:

```powershell
python scripts\build_exe.py
```

Skrip ini akan:

- Menjalankan PyInstaller dengan mode `--onefile` dan enkripsi bytecode (`--key`).
- Menggabungkan `config.json` dan `requirements.txt` ke dalam paket.
- Menghasilkan `dist\BotForexMT5.exe` serta menampilkan kunci enkripsi PyInstaller yang dipakai (catat jika ingin rebuild).

## 3. Menjalankan Aplikasi di Mesin Lain

1. Salin file `dist\BotForexMT5.exe` ke laptop tujuan.
2. Pastikan folder `credentials\` (jika sudah dibuat) ikut disalin agar file `.enc` tetap tersedia. Jika belum ada, aplikasi akan membuatnya saat pertama kali jalan.
3. Klik dua kali `BotForexMT5.exe` atau jalankan via PowerShell:
   ```powershell
   .\BotForexMT5.exe
   ```

## 4. Tahap Setup Kredensial (Satu Kali Saja)

Pada eksekusi pertama (atau ketika file terenkripsi tidak ada), aplikasi akan menampilkan panel "Setup Kredensial". Ikuti langkah berikut:

1. Jawab `Y` ketika diminta melanjutkan setup.
2. Buat passphrase rahasia dua kali (akan dipakai untuk membuka kunci di masa depan).
3. Masukkan `GEMINI_API_KEY` dan `NEWS_API_KEY` Anda ketika diminta.
4. Aplikasi akan mengenkripsi kedua kunci tersebut dan menyimpannya pada `credentials\api_keys.enc`.
5. Setelah sukses, kredensial dimuat ke lingkungan runtime dan bot siap dipakai.

> **Catatan penting:** simpan passphrase Anda dengan aman. Tanpa passphrase, file terenkripsi tidak bisa dibuka.

## 5. Mengakses Kembali Aplikasi

Setiap kali aplikasi dijalankan:

- Jika file terenkripsi tersedia, Anda akan diminta memasukkan passphrase.
- Jika passphrase benar, bot otomatis menanam (`export`) API key ke environment dan melanjutkan ke antarmuka CLI.
- Apabila passphrase 3× salah, aplikasi akan berhenti untuk mencegah brute-force.

## 6. Tips Keamanan Tambahan

- Jangan membagikan passphrase maupun file `.enc` ke pihak yang tidak dipercaya.
- Untuk mereset kredensial, hapus `credentials\api_keys.enc` lalu jalankan ulang `.exe` agar proses setup diulang dari awal.
- Bila ingin merilis versi publik, jalankan kembali `scripts\build_exe.py` untuk memperoleh kunci enkripsi PyInstaller yang baru.

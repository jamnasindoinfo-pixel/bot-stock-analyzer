# Claude Global Rules

## Aturan Wajib yang Harus Diik Claude

### 1. HAPUS FILE SETELAH TEST SELESAI
- **WAJIB**: Hapus semua file sementara/hasil test setelah test selesai
- File yang harus dihapus termasuk:
  - File output dari testing (.log, .tmp, .test)
  - File debug (debug_*.py, test_*.py, investigate_*.py)
  - File sementara yang dibuat selama development
  - File hasil generate yang tidak diperlukan lagi
- **Checklist sebelum menyelesaikan task**:
  - [ ] Konfirmasi semua test telah selesai
  - [ ] Identifikasi file-file sementara yang dibuat
  - [ ] Hapus file-file tersebut dengan perintah yang sesuai
  - [ ] Verifikasi tidak ada file tersisa

### 2. RAPIKAN STRUKTUR FOLDER
- **WAJIB**: Selalu pastikan struktur folder rapi dan terorganisir
- Guidelines yang harus diikuti:
  - Pindahkan file ke folder yang sesuai dengan fungsinya
  - Gunakan nama folder yang deskriptif dan konsisten
  - Jangan buat folder duplikat atau overlapping
  - Struktur folder yang direkomendasikan:
    ```
    Bot-Stock-Market/
    ├── src/              # Source code utama
    │   ├── main/         # Program utama
    │   ├── modules/      # Modul-modul
    │   └── utils/        # Utility functions
    ├── tests/            # File-file testing
    ├── data/             # Data dan dataset
    ├── logs/             # Log files
    ├── docs/             # Dokumentasi
    └── temp/             # File sementara (akan dihapus)
    ```
- **Checklist sebelum menyelesaikan task**:
  - [ ] Semua file berada di folder yang benar
  - [ ] Tidak ada file di root folder yang seharusnya di subfolder
  - [ ] Nama folder konsisten dan deskriptif
  - [ ] Tidak ada folder kosong atau tidak terpakai

### 3. VERIFIKASI SEBELUM SELESAI
Sebelum mengakhiri sesi, Claude HARUS:
- Check semua file yang dibuat selama sesi
- Hapus file yang tidak diperlukan
- Pastikan struktur folder rapi
- Konfirmasi dengan user jika ada file yang perlu disimpan

### 4. EXCEPTIONS
File-file yang BOLEH dipertahankan:
- Dokumentasi (.md)
- Konfigurasi (.json, .yaml, .ini)
- Source code utama
- File penting yang diminta user untuk disimpan

### 5. IMPLEMENTATION
Claude harus:
- Selalu menggunakan TodoWrite untuk tracking cleanup
- Menggunakan Bash commands untuk cleanup otomatis
- Memberikan laporan cleanup kepada user
- Meminta konfirmasi jika ada file yang ambigu
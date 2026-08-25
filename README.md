# Sistem RAG - Manajemen Akreditasi Program Studi

## Langkah 1: Pindahkan File Tar

1. Download `open-webui-rag.tar` dari Google Drive
2. Buat folder `rag_app_offline` di komputer server:
   ```bash
   # Linux
   mkdir -p ~/rag_app_offline
   mv ~/Downloads/open-webui-rag.tar ~/rag_app_offline/
   
   # Windows (PowerShell)
   mkdir C:\rag_app_offline
   Move-Item ~\Downloads\open-webui-rag.tar C:\rag_app_offline\
   ```
3. Pindah ke folder tersebut:
   ```bash
   cd ~/rag_app_offline   # Linux
   cd C:\rag_app_offline  # Windows
   ```

---

## Langkah 2: Install Docker

### Linux
```bash
sudo apt update && sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
# Logout dan login ulang
```

### Windows
1. Download Docker Desktop dari https://docker.com/products/docker-desktop
2. Install → centang "Use WSL 2"
3. Restart komputer

---

## Langkah 3: Load Image Docker

```bash
# Load image dari file tar
docker load -i open-webui-rag.tar

# Verifikasi - harus muncul open-webui-rag:latest
docker images | findstr open-webui-rag
```

---

## Langkah 4: Install Ollama

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows
1. Download installer dari https://ollama.com/download
2. Jalankan installer (`OllamaSetup.exe`)
3. Ikuti petunjuk instalasi

### Verifikasi Installasi
```bash
# Cek versi Ollama - harus muncul versi
ollama --version
```

---

## Langkah 5: Download & Jalankan GLM-4.7-Flash

### 5a. Download Model (sekali saja, ~19GB)
```bash
# Jalankan perintah ini - akan otomatis download model
ollama run glm-4.7-flash
```

**TUNGGU proses download selesai** (10-30 menit tergantung internet). 
Akan muncul prompt `>>>` jika sudah selesai.

### 5b. Verifikasi Model Sudah Di-download
```bash
# Buka terminal BARU (jangan tutup terminal yang pertama)
# Jalankan perintah ini:
curl http://localhost:11434/api/tags
```

**Yang harus muncul:** nama model `glm-4.7-flash` di dalam response JSON.

### 5c. Pastikan Ollama Tetap Running
```bash
# Ollama harus tetap running di background
# Jika sudah tutup terminal, jalankan:
ollama serve
```

**PENTING:** 
- Jika pakai `ollama run`, Ollama akan tetap running di background
- Jika pakai `ollama serve`, terminal harus tetap dibuka
- **Jangan tutup terminal Ollama** sampai selesai setup

---

## Langkah 6: Jalankan Open WebUI

**SEBELUM menjalankan Langkah 6, pastikan:**
1. Ollama sudah running (`curl http://localhost:11434/api/tags` berhasil)
2. Model `glm-4.7-flash` sudah muncul di response

```bash
# Jalankan Open WebUI
docker run -d -p 8080:8080 ^
  --add-host=host.docker.internal:host-gateway ^
  -v open-webui:/app/backend/data ^
  --name open-webui ^
  --restart always ^
  open-webui-rag
```

---

## Langkah 7: Buka di Browser

1. Buka `http://localhost:8080`
2. Buat akun admin (email + password)
3. Berikan IP komputer server + port 8080 ke user lain

---

## Langkah 8: Konfigurasi Model

### 8a. Setting Koneksi Ollama
1. Login ke Open WebUI
2. Klik ikon **gear** (Settings) di pojok kiri atas
3. Klik **Connections** di sidebar kiri
4. Cari bagian **Ollama**:
   - **Base URL:** `http://host.docker.internal:11434`
   - **Model:** `glm-4.7-flash`
5. Klik **Save**

### 8b. Verifikasi Koneksi Berhasil
1. Klik **Chat** di sidebar kiri
2. Di bagian model (atas), harus muncul `glm-4.7-flash`
3. Jika tidak muncul, klik refresh (icon ↻) di sebelah nama model

### 8c. Test Chat
1. Pilih model `glm-4.7-flash`
2. Ketik pesan: "Halo, apa kabar?"
3. Jika berhasil, model akan membalas

**Jika model tidak muncul:** lihat section Troubleshooting di bawah.

---

## Langkah 9: Upload Dataset ke Knowledge Base

1. Buka **Workspace** → **Knowledge**
2. Klik **Create Knowledge**
3. Isi nama (mis. "Dataset Akreditasi")
4. Upload PDF satu per satu atau bulk upload
5. Tunggu proses indexing selesai

---

## Langkah 10: Buat Model RAG Default

1. Buka **Workspace** → **Models** → **Create Model**
2. Isi:
   - **Name:** `RAG Akreditasi`
   - **Base Model:** `glm-4.7-flash`
   - **System Prompt:**
     ```
     Anda adalah asisten akademik program studi Ilmu Informatika.
     Anda HANYA boleh menjawab berdasarkan KONTEKS di bawah ini.
     JANGAN gunakan pengetahuan dari luar konteks.
     Jika konteks tidak cukup atau tidak relevan, jawab: "Tidak ditemukan dalam dokumen."
     Jika konteks mengandung tabel, baca tabel tersebut untuk menjawab.
     Setiap bagian konteks diawali dengan [Halaman X].
     Sebutkan nomor halaman di akhir setiap poin jawaban.
     Jawab dalam bahasa Indonesia.
     ```
3. Pada bagian **Knowledge**, pilih knowledge base yang berisi semua dokumen
4. Klik **Save**

---

## Troubleshooting

### Model tidak muncul di Open WebUI?

**Cek 1: Ollama running?**
```bash
curl http://localhost:11434/api/tags
```
- ✅ Jika muncul JSON dengan nama model → Ollama running
- ❌ Jika error "Unable to connect" → Ollama tidak running

**Solusi:** Jalankan `ollama serve` di terminal baru

**Cek 2: Model sudah di-download?**
```bash
ollama list
```
- ✅ Jika muncul `glm-4.7-flash` → model sudah ada
- ❌ Jika kosong → model belum di-download

**Solusi:** Jalankan `ollama run glm-4.7-flash` dan tunggu download selesai

**Cek 3: Koneksi dari Docker?**
```bash
# Di luar Docker, test koneksi
curl http://localhost:11434/api/tags
```

Jika berhasil di luar Docker tapi tidak di Open WebUI:
- Pastikan Base URL di Settings = `http://host.docker.internal:11434`
- **JANGAN** pakai `localhost:11434` (tidak bisa diakses dari dalam Docker)

### Chat tidak jalan / timeout?

1. Pastikan model sudah di-load (cek `ollama list`)
2. Restart Ollama: `ollama serve`
3. Restart Open WebUI:
   ```bash
   docker restart open-webui
   ```

### Error "model not found"?

1. Pastikan nama model benar: `glm-4.7-flash` (huruf kecil, ada strip)
2. Download ulang: `ollama pull glm-4.7-flash`

---

## Ringkasan Command

```bash
# Install Ollama (Windows)
# Download dari https://ollama.com/download

# Download model (sekali saja)
ollama run glm-4.7-flash

# Verifikasi
ollama list
curl http://localhost:11434/api/tags

# Jalankan Ollama (jika perlu)
ollama serve

# Load Docker image
docker load -i open-webui-rag.tar

# Jalankan Open WebUI
docker run -d -p 8080:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always open-webui-rag

# Cek status
docker ps
docker logs open-webui
```

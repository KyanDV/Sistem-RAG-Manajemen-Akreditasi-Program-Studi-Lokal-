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

# Verifikasi
docker images | grep open-webui-rag
```

---

## Langkah 4: Install Ollama

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download dari https://ollama.com
```

---

## Langkah 5: Jalankan GLM-4.7-Flash

```bash
# Jalankan model (otomatis download ~19GB untuk Q4)
ollama run glm-4.7-flash

# Test koneksi
curl http://localhost:11434/api/chat -d '{
  "model": "glm-4.7-flash",
  "messages": [{"role": "user", "content": "Halo"}]
}'
```

**Tunggu model selesai di-load** (1-3 menit pertama kali).

---

## Langkah 6: Jalankan Open WebUI

```bash
docker run -d -p 8080:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  open-webui-rag
```

---

## Langkah 7: Buka di Browser

1. Buka `http://localhost:8080`
2. Buat akun admin (email + password)
3. Berikan IP komputer server + port 8080 ke user lain

---

## Langkah 8: Konfigurasi Model

1. Login ke Open WebUI
2. Buka **Settings** → **Connections**
3. Pada bagian **Ollama**:
   - Base URL: `http://host.docker.internal:11434`
   - Model: `glm-4.7-flash`
4. Klik **Save**

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

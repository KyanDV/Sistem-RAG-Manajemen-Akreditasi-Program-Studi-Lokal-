EMBEDDING_MODEL = "BAAI/bge-m3"
FINETUNED_EMBEDDING_PATH = "/home/kyan67verado/rag_app/data/bge-m3-akreditasi"
USE_FINETUNED = True
FINETUNED_THRESHOLD = 1.55
LLM_MODEL = "gpt-4o-mini"
CHROMA_DB_PATH = "/home/kyan67verado/rag_app/data/chroma_db"
CHUNK_TARGET_SIZE = 600
CHUNK_OVERLAP = 50
SIMILARITY_THRESHOLD = 1.2  # used for base model
ACADEMIC_TOP_K = 6
MAX_TYPO_DISTANCE = 2

DATA_DIRS = [
    "/home/kyan67verado/rag_app/data/Dataset",
]

ACADEMIC_KEYWORDS = [
    "cpl", "cpmk", "sub-cpmk", "minggu ke", "mg ke-",
    "materi", "pustaka", "bobot", "sks",
    "capaian pembelajaran", "indikator", "penilaian",
    "perangkat lunak", "perangkat keras",
    "dosen pengampu", "prasyarat", "matakuliah syarat",
    "rencana pembelajaran", "bahan kajian",
    "materi pembelajaran", "media pembelajaran",
    "sumber belajar", "estimasi waktu",
    "topik", "bahasan", "pokok bahasan",
    "deskripsi singkat", "profil lulusan",
    "struktur kurikulum", "komponen penilaian",
    "metode penilaian", "sebaran mata kuliah",
    "pengelompokan mata kuliah", "pilihan minat",
    "konversi mbkm", "ketua majelis",
    "instrumen", "akreditasi", "borang",
    "lkps", "led", "matriks penilaian",
    "prosedur", "pengendalian", "penjaminan mutu",
    "penetapan", "pelaksanaan", "evaluasi",
    "peningkatan mutu", "tata pamong",
    "visi", "misi", "tujuan", "strategi",
    "sdm", "dosen", "mahasiswa", "tendik",
    "penelitian", "pkm", "pengabdian",
    "publikasi", "kerja sama", "kerjasama",
    "dudika", "stakeholder", "pemangku kepentingan",
    "masa studi", "ipk", "lulusan",
    "kurikulum", "rps", "silabus", "sap",
    "ketua", "kaprodi", "program studi",
    "dekan", "fakultas", "informatika",
    "nama", "identitas", "penanggungjawab",
    "iaps", "ban-pt", "lam", "lam infokom",
    "akreditasi unggul", "baik sekali",
    "borang", "asesor", "akreditasi pertama",
    "reakreditasi", "ppep",
    "aps", "apt", "lam infokom",
    "tridarma", "luaran", "capaian tridarma",
]

COURSE_KEYWORDS = {
    "jaringan komputer": "IF24301",
    "pemrograman jaringan": "IF24701",
    "pengembangan game": "IF24508",
    "visi komputer": "IF24704",
    "pembelajaran mesin": "IF24503",
    "pengantar teknologi informasi cerdas": "IF24305",
    "pemrograman web": "IF24304",
    "algoritma pemrograman": "IF24102",
    "pengenalan pemrograman": "IF24101",
    "basis data": "IF24103",
    "kalkulus": "IF24104",
    "organisasi dan arsitektur komputer": "IF24105",
    "matematika diskrit": "IF24302",
    "teori graf": "IF24303",
    "rpl": "IF24306",
    "rekayasa perangkat lunak": "IF24306",
    "pemrograman mobile": "IF24501",
    "pemrograman berbasis platform": "IF24502",
    "hukum dan kebijakan teknologi informasi": "IF24505",
    "arsitektur software": "IF24509",
    "capstone project": "IF24702",
    "internet of things": "IF24703",
    "pengujian perangkat lunak": "IF24706",
}

COURSE_FILENAME_MAP = {
    "modul_praktikum_ml": "IF24503",
    "modul_praktikum_pm": "IF24501",
    "modul_praktikum_ppl": "IF24706",
    "modul_praktikum_rpl": "IF24306",
}

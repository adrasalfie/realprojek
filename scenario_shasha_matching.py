import pandas as pd
from datetime import datetime
from rapid import preclean_sijk, merge_sijk_to_sf, merge_lpse_to_sf


# =====================================================
# PROGRAM: GABUNG DATA LPSE & SIJK KE FILE UTAMA
# CATATAN:
# - SEMUA DATA FORMAT TEXT
# - FILE UTAMA TIDAK DIUBAH STRUKTUR
# - DATA LPSE & SIJK DI-APPEND
# - DUPLIKASI DIHAPUS
# =====================================================

# =====================================================
# 0. PARAMETER GLOBAL
# =====================================================
TAHUN_REVISI = datetime.now().year  # untuk mendapatkan tahun saat program dijalankan otomatis 2025, 2026, dst

# =====================================================
# 1. LOAD FILE (PAKSA STRING)
# =====================================================
df_utama = pd.read_excel("7500 sf.xlsx", dtype=str) # GANTI FILE SF (UTAMA) DISINI

df_lpse = pd.read_excel(
    "7500 lpse.xlsx", # GANTI FILE LPSE DISINI
    dtype=str #untuk memastikan semua kolom dibaca sebagai string
)

df_sijk = pd.read_excel(
    "7500 sijk.xlsx", # GANTI FILE SIJK DISINI
    dtype=str #untuk memastikan semua kolom dibaca sebagai string
)


# =====================================================
# 2. NORMALISASI NAMA KOLOM
# =====================================================
def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()
    return df


df_utama = normalize_columns(df_utama)
df_lpse = normalize_columns(df_lpse)
df_sijk = normalize_columns(df_sijk)

# ==============================
# SIMPAN JUMLAH & URUTAN ASLI SF
# ==============================
jumlah_sf_awal = len(df_utama)

# pastikan kolom no tidak berubah tipe
df_utama["no"] = df_utama["no"].astype(str)

# simpan index asli untuk menjaga urutan
df_utama["_sf_order"] = range(len(df_utama))

# =====================================================
# 3. STANDARISASI DATA LPSE
# =====================================================


def standardize_lpse(df):
    mapping = {
        # mapping lama
        "nama_penyedia": "nama_perusahaan",
        "npwp_penyedia": "npwp",
        "alamat": "alamat_perusahaan",
        "telepon": "no_telp",
        "fax": "fax",
        "website": "website",
        "nomor_izin_usaha": "nib",
        "bentuk_usaha": "badan_usaha",
        "kualifikasi_usaha": "kualifikasi",

        "kdprov": "prov",      # H -> E
        "kd_kab": "kab",       # K -> F

        # mapping tambahan
        "nmprov": "nm_prov",
        "nmkab": "nm_kab",
        "kd_klasifikasi": "kbli",
        "kd_penyedia": "kd_penyedia",
    }

    df = df.rename(columns={k: v for k, v in mapping.items()})
    df = df.loc[:, ~df.columns.duplicated()]

    # === KOLOM OTOMATIS FILE UTAMA ===
    df["tahun_revisi"] = str(TAHUN_REVISI)
    df["kategori"] = "F"
    df["sumber_data"] = "1"   # LPSE

    # Hapus duplikat kd_penyedia
    df = df.drop_duplicates(subset=["kd_penyedia"])

    return df


# =====================================================
# 4. STANDARISASI DATA SIJK
# =====================================================
def standardize_sijk(df):
    mapping = {
        # mapping lama
        "nama_bu": "nama_perusahaan",
        "npwp_bu": "npwp",
        "telepon_bu": "no_telp",  
        "email_bu": "email",   
        "alamat_bu": "alamat_perusahaan",
        "nib": "nib",

        "kdprov": "prov",                 
        "kd_kab": "kab",                  
        "sub_klasifikasi": "pekerjaan_utama",  

        # mapping tambahan
        "bentuk_usaha_bu": "badan_usaha",
        "nmprov": "nm_prov",
        "nmkab": "nm_kab",
        "kualifikasi_final": "kualifikasi",
        "skala_usaha": "skala_usaha",
    }

    df = df.rename(columns={k: v for k, v in mapping.items()})
    df = df.loc[:, ~df.columns.duplicated()]

    # === KOLOM OTOMATIS FILE UTAMA ===
    df["tahun_revisi"] = str(TAHUN_REVISI)
    df["kategori"] = "F"
    df["sumber_data"] = "2"   # SIJK

    return df


# =====================================================
# 5. PROSES STANDARISASI
# =====================================================
lpse_std = standardize_lpse(df_lpse)
sijk_std = standardize_sijk(df_sijk)


# =====================================================
# 6. SAMAKAN STRUKTUR KOLOM DENGAN FILE UTAMA
# =====================================================
target_columns = df_utama.columns.tolist()

lpse_std = lpse_std.reindex(columns=target_columns)
sijk_std = sijk_std.reindex(columns=target_columns)


# ==============================
# 7. PRE-CLEANING SIJK INTERNAL -> PROSES RAPIDFUZZ
# ==============================
sijk_std_clean, preview_sijk = preclean_sijk(
    sijk_std, threshold=91, preview=True)
preview_sijk.to_excel(
    "preview_rapidfuzz_internal_sijk.xlsx",
    index=False
)
print("=============== Preview Rapidfuzz berhasil dibuat ===============")

# ==============================
# 8. MERGE SIJK KE SF -> PROSES RAPIDFUZZ
# ==============================
df_utama_final, preview_merge = merge_sijk_to_sf(
    df_utama, sijk_std_clean, threshold=91, preview=True)
preview_merge.to_excel(
    "preview_similarity_merge_sijk_to_sf_.xlsx",
    index=False
)
print("=============== Preview Similarity Merge sijk to sf berhasil dibuat ===============")

# ==============================
# 9. MERGE LPSE KE SF -> RAPIDFUZZ
# ==============================
data_gabungan, preview_lpse_merge = merge_lpse_to_sf(
    df_utama_final,
    lpse_std,
    threshold=91,
    preview=True
)

preview_lpse_merge.to_excel(
    "preview_similarity_merge_lpse_to_sf.xlsx",
    index=False
)

print("=============== Preview Similarity Merge LPSE to SF berhasil dibuat ===============")


# =====================================================
# 10. PAKSA SEMUA KOLOM TEXT  (KECUALI KOLOM INTERNAL)
# =====================================================
cols_text = data_gabungan.columns.difference(["_sf_order"])
data_gabungan[cols_text] = (
    data_gabungan[cols_text]
    .fillna("")
    .astype(str)
)


# =====================================================
# 11. NORMALISASI UNTUK DEDUPLIKASI
# =====================================================
for col in ["nama_perusahaan", "nm_prov", "nm_kab"]:
    if col in data_gabungan.columns:
        data_gabungan[col] = (
            data_gabungan[col]
            .str.upper()
            .str.strip()
        )


# =====================================================
# 12. HAPUS DUPLIKAT TANPA MENYENTUH SF
# =====================================================

# Defragment
data_gabungan = data_gabungan.copy()

# Tandai SF asli
data_gabungan["_is_sf"] = data_gabungan["_sf_order"].notna()

# Pisahkan
data_sf = data_gabungan[data_gabungan["_is_sf"]].copy()
data_non_sf = data_gabungan[~data_gabungan["_is_sf"]].copy()

# PRIORITAS NON-SF
priority_map = {
    "2": 1,  # SIJK
    "1": 2   # LPSE
}

data_non_sf["_priority"] = (
    data_non_sf["sumber_data"]
    .map(priority_map)
    .fillna(99)
)

# Urutkan non-SF berdasarkan prioritas
data_non_sf = data_non_sf.sort_values("_priority")

# Deduplicate hanya non-SF
data_non_sf = data_non_sf.drop_duplicates(
    subset=["nama_perusahaan", "nm_prov", "nm_kab"],
    keep="first"
)

# Hapus helper
data_non_sf = data_non_sf.drop(columns=["_priority"])

# Gabungkan kembali
data_gabungan = pd.concat([data_sf, data_non_sf], ignore_index=True)


# =====================================================
# 13. Skala_usaha dikonversi ke angka
# =====================================================

# KUALIFIKASI
map_kualifikasi = {
    "spesialis": "1",
    "kecil": "2",
    "menengah": "3",
    "besar": "4",
    "tidak memiliki sbu aktif": "9",
}

data_gabungan["kualifikasi"] = (
    data_gabungan["kualifikasi"]
    .str.lower()
    .map(map_kualifikasi)
    .fillna(data_gabungan["kualifikasi"])
)


# SKALA USAHA (SIJK)
map_skala = {
    "kecil": "2",
    "menengah": "3",
    "besar": "4",
}

data_gabungan["skala_usaha"] = (
    data_gabungan["skala_usaha"]
    .str.lower()
    .map(map_skala)
    .fillna(data_gabungan["skala_usaha"])
)


# BADAN USAHA (LPSE + SIJK)
map_badan_usaha = {
    "pt. persero (bumn/bumd)": "1",
    "pt": "2",
    "cv": "3",
    "koperasi": "4",
    "kantor perwakilan bujka": "5",
}

data_gabungan["badan_usaha"] = (
    data_gabungan["badan_usaha"]
    .str.lower()
    .map(map_badan_usaha)
    .fillna("9")
)

# =====================================================
# 13. KEMBALIKAN URUTAN SESUAI FILE SF
# =====================================================

# Pisahkan SF dan non-SF
data_sf = data_gabungan[data_gabungan["_is_sf"]].copy()
data_non_sf = data_gabungan[~data_gabungan["_is_sf"]].copy()

# Urutkan SF berdasarkan urutan asli
if "_sf_order" in data_sf.columns:
    data_sf["_sf_order"] = pd.to_numeric(data_sf["_sf_order"], errors="coerce")
    data_sf = data_sf.sort_values("_sf_order")

# Gabungkan kembali: SF dulu, baru tambahan
data_gabungan = pd.concat([data_sf, data_non_sf], ignore_index=True)

# Hapus kolom bantu
data_gabungan = data_gabungan.drop(
    columns=["_is_sf", "_sf_order"], errors="ignore")

# =====================================================
# 14. SIMPAN HASIL
# =====================================================
data_gabungan.to_excel("File_Utama_Hasil.xlsx", index=False)

print("=============== PROSES APPEND DATA SELESAI ===============")

import pandas as pd
from datetime import datetime
from rapid import preclean_sijk, merge_sijk_to_sf

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
TAHUN_REVISI = datetime.now().year  # otomatis 2025, 2026, dst

# =====================================================
# 1. LOAD FILE (PAKSA STRING)
# =====================================================
df_utama = pd.read_excel("7500 sf.xlsx", dtype=str)

df_lpse = pd.read_excel(
    "7500 lpse.xlsx",
    dtype=str
)

df_sijk = pd.read_excel(
    "7500 sijk.xlsx",
    dtype=str
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

        # mapping tambahan
        "sub_klasifikasi": "pekerjaan_utama",
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
    sijk_std, threshold=75, preview=True)
preview_sijk.to_excel(
    "preview_rapidfuzz_internal_sijk.xlsx",
    index=False
)
print("=============== Preview Rapidfuzz berhasil dibuat ===============")

# ==============================
# 8. MERGE SIJK KE SF -> PROSES RAPIDFUZZ
# ==============================
df_utama_final, preview_merge = merge_sijk_to_sf(
    df_utama, sijk_std_clean, threshold=75, preview=True)
preview_merge.to_excel(
    "preview_similarity_merge_sijk_to_sf_.xlsx",
    index=False
)
print("=============== Preview Similarity Merge sijk to sf berhasil dibuat ===============")

# =====================================================
# 9. GABUNGKAN DATA (APPEND)
# =====================================================
data_gabungan = pd.concat(
    [df_utama_final, lpse_std],
    ignore_index=True
)


# =====================================================
# 10. PAKSA SEMUA KOLOM TEXT
# =====================================================
data_gabungan = data_gabungan.fillna("").astype(str)


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
# 12. HAPUS DUPLIKAT
# =====================================================
data_gabungan = data_gabungan.drop_duplicates(
    subset=["nama_perusahaan", "nm_prov", "nm_kab"]
)


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
# 14. SIMPAN HASIL
# =====================================================
data_gabungan.to_excel("File_Utama_Hasil.xlsx", index=False)

print("=============== PROSES APPEND DATA SELESAI ===============")

import pandas as pd
import re

# =====================================================
# PROGRAM: CEK & HAPUS DATA DUPLIKAT
# CATATAN:
# - SEMUA DATA DIPAKSA MENJADI TEXT (STRING)
# - File tidak diubah strukturnya
# =====================================================


# =====================================================
# 1. LOAD FILE (PAKSA SEMUA KOLOM STRING)
# =====================================================
df = pd.read_excel(
    "File_Utama_Hasil.xlsx",
    dtype=str
)


# =====================================================
# 2. PAKSA SEMUA KOLOM TEXT & AMANKAN NaN
# =====================================================
df = df.fillna("").astype(str)


# =====================================================
# 3. NORMALISASI TEKS (UNTUK PERBANDINGAN)
# =====================================================
def normalize_text(text):
    return str(text).lower().strip()


df["nama_perusahaan_norm"] = df["nama_perusahaan"].apply(normalize_text)
df["nm_kel_norm"] = df["nm_kel"].apply(normalize_text)
df["nm_kec_norm"] = df["nm_kec"].apply(normalize_text)
df["alamat_norm"] = df["alamat_perusahaan"].apply(normalize_text)


# =====================================================
# 4. FUNGSI CEK KATA DALAM ALAMAT
# =====================================================
def wilayah_in_address(wilayah, alamat):
    if wilayah == "" or alamat == "":
        return False

    # Hilangkan kata umum agar pencocokan fleksibel
    alamat = re.sub(
        r"\b(rt|rw|kel|kel\.|kecamatan|kec|kabupaten|kab|kota)\b",
        " ",
        alamat
    )

    return wilayah in alamat


# =====================================================
# 5. PROSES DETEKSI DATA DUPLIKAT
# =====================================================
rows_to_drop = set()

# Kelompokkan berdasarkan nama_perusahaan
for _, group in df.groupby("nama_perusahaan_norm"):

    # Data yang sudah punya kelurahan atau kecamatan
    data_lengkap = group[
        (group["nm_kel_norm"] != "") |
        (group["nm_kec_norm"] != "")
    ]

    # Data yang TIDAK punya kelurahan & kecamatan
    data_tidak_lengkap = group[
        (group["nm_kel_norm"] == "") &
        (group["nm_kec_norm"] == "")
    ]

    if data_lengkap.empty:
        continue

    # Bandingkan setiap data tidak lengkap
    for idx_no, row_no in data_tidak_lengkap.iterrows():
        alamat = row_no["alamat_norm"]

        for _, row_ok in data_lengkap.iterrows():
            nm_kel = row_ok["nm_kel_norm"]
            nm_kec = row_ok["nm_kec_norm"]

            # Jika kelurahan ATAU kecamatan ditemukan di alamat
            if (
                wilayah_in_address(nm_kel, alamat) or
                wilayah_in_address(nm_kec, alamat)
            ):
                rows_to_drop.add(idx_no)
                break


# =====================================================
# 6. HAPUS DATA DUPLIKAT
# =====================================================
df_clean = df.drop(index=list(rows_to_drop))


# =====================================================
# 7. HAPUS KOLOM BANTUAN
# =====================================================
df_clean = df_clean.drop(
    columns=[
        "nama_perusahaan_norm",
        "nm_kel_norm",
        "nm_kec_norm",
        "alamat_norm"
    ],
    errors="ignore"
)


# =====================================================
# 8. PAKSA ULANG SEMUA KOLOM MENJADI TEXT
# =====================================================
df_clean = df_clean.fillna("").astype(str)


# =====================================================
# 9. SIMPAN HASIL AKHIR (FORMAT TEXT)
# =====================================================
df_clean.to_excel(
    "File_Utama_Final.xlsx",
    index=False
)

print("=============== PROSES FILTER SELESAI ===============")

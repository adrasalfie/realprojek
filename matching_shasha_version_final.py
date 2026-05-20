import streamlit as st
import pandas as pd
from datetime import datetime
from rapid import preclean_sijk, merge_sijk_to_sf, merge_lpse_to_sf

# =====================================================
# CONFIG
# =====================================================
TAHUN_REVISI = str(datetime.now().year)

st.set_page_config(page_title="Pipeline LPSE-SIJK", layout="wide")
st.title("📊 Data Integration Pipeline (SF + LPSE + SIJK)")

# =====================================================
# FILE UPLOAD
# =====================================================
sf_file = st.file_uploader("Upload File Utama (SF)", type=["xlsx"])
lpse_file = st.file_uploader("Upload File LPSE", type=["xlsx"])
sijk_file = st.file_uploader("Upload File SIJK", type=["xlsx"])

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()
    return df


def standardize_lpse(df):
    mapping = {
        "nama_penyedia": "nama_perusahaan",
        "npwp_penyedia": "npwp",
        "alamat": "alamat_perusahaan",
        "telepon": "no_telp",
        "fax": "fax",
        "website": "website",
        "nomor_izin_usaha": "nib",
        "bentuk_usaha": "badan_usaha",
        "kualifikasi_usaha": "kualifikasi",
        "kdprov": "prov",
        "kd_kab": "kab",
        "nmprov": "nm_prov",
        "nmkab": "nm_kab",
        "kd_klasifikasi": "kbli",
        "kd_penyedia": "kd_penyedia",
    }

    df = df.rename(columns=mapping)
    df = df.loc[:, ~df.columns.duplicated()]

    df["tahun_revisi"] = TAHUN_REVISI
    df["kategori"] = "F"
    df["sumber_data"] = "1"

    return df.drop_duplicates(subset=["kd_penyedia"], errors="ignore")


def standardize_sijk(df):
    mapping = {
        "nama_bu": "nama_perusahaan",
        "npwp_bu": "npwp",
        "telepon_bu": "no_telp",
        "email_bu": "email",
        "alamat_bu": "alamat_perusahaan",
        "kdprov": "prov",
        "kd_kab": "kab",
        "sub_klasifikasi": "pekerjaan_utama",
        "bentuk_usaha_bu": "badan_usaha",
        "nmprov": "nm_prov",
        "nmkab": "nm_kab",
        "kualifikasi_final": "kualifikasi",
        "skala_usaha": "skala_usaha",
    }

    df = df.rename(columns=mapping)
    df = df.loc[:, ~df.columns.duplicated()]

    df["tahun_revisi"] = TAHUN_REVISI
    df["kategori"] = "F"
    df["sumber_data"] = "2"

    return df


def mapping_values(df):
    map_kualifikasi = {
        "spesialis": "1",
        "kecil": "2",
        "menengah": "3",
        "besar": "4",
        "tidak memiliki sbu aktif": "9",
    }

    map_skala = {
        "kecil": "2",
        "menengah": "3",
        "besar": "4",
    }

    map_badan = {
        "pt. persero (bumn/bumd)": "1",
        "pt": "2",
        "cv": "3",
        "koperasi": "4",
        "kantor perwakilan bujka": "5",
    }

    if "kualifikasi" in df:
        df["kualifikasi"] = df["kualifikasi"].str.lower().map(map_kualifikasi).fillna(df["kualifikasi"])

    if "skala_usaha" in df:
        df["skala_usaha"] = df["skala_usaha"].str.lower().map(map_skala).fillna(df["skala_usaha"])

    if "badan_usaha" in df:
        df["badan_usaha"] = df["badan_usaha"].str.lower().map(map_badan).fillna("9")

    return df


# =====================================================
# MAIN PROCESS
# =====================================================
if st.button("🚀 Run Pipeline"):

    if not all([sf_file, lpse_file, sijk_file]):
        st.error("⚠️ Please upload all required files")
        st.stop()

    with st.spinner("Processing..."):

        # LOAD
        df_utama = pd.read_excel(sf_file, dtype=str)
        df_lpse = pd.read_excel(lpse_file, dtype=str)
        df_sijk = pd.read_excel(sijk_file, dtype=str)

        # NORMALIZE
        df_utama = normalize_columns(df_utama)
        df_lpse = normalize_columns(df_lpse)
        df_sijk = normalize_columns(df_sijk)

        # SAVE SF ORDER
        df_utama["_sf_order"] = range(len(df_utama))

        # STANDARDIZE
        lpse_std = standardize_lpse(df_lpse)
        sijk_std = standardize_sijk(df_sijk)

        # ALIGN STRUCTURE
        target_cols = df_utama.columns.tolist()
        lpse_std = lpse_std.reindex(columns=target_cols)
        sijk_std = sijk_std.reindex(columns=target_cols)

        # RAPIDFUZZ PROCESS
        sijk_clean, _ = preclean_sijk(sijk_std, threshold=91, preview=False)
        df_merge, _ = merge_sijk_to_sf(df_utama, sijk_clean, threshold=91, preview=False)
        df_final, _ = merge_lpse_to_sf(df_merge, lpse_std, threshold=91, preview=False)

        # CLEAN TEXT
        df_final = df_final.fillna("").astype(str)

        for col in ["nama_perusahaan", "nm_prov", "nm_kab"]:
            if col in df_final.columns:
                df_final[col] = df_final[col].str.upper().str.strip()

        # SPLIT SF vs NON-SF
        df_final["_is_sf"] = df_final["_sf_order"].notna()

        df_sf = df_final[df_final["_is_sf"]].copy()
        df_non = df_final[~df_final["_is_sf"]].copy()

        # PRIORITY
        priority_map = {"2": 1, "1": 2}
        df_non["_priority"] = df_non["sumber_data"].map(priority_map).fillna(99)

        df_non = df_non.sort_values("_priority")
        df_non = df_non.drop_duplicates(
            subset=["nama_perusahaan", "nm_prov", "nm_kab"],
            keep="first"
        )

        # MERGE BACK
        df_final = pd.concat([df_sf, df_non], ignore_index=True)

        # MAPPING
        df_final = mapping_values(df_final)

        # RESTORE ORDER
        df_sf = df_final[df_final["_is_sf"]].sort_values("_sf_order")
        df_non = df_final[~df_final["_is_sf"]]

        df_final = pd.concat([df_sf, df_non], ignore_index=True)

        # DROP HELPER
        df_final = df_final.drop(columns=["_sf_order", "_is_sf"], errors="ignore")

        # SAVE
        output = "File_Utama_Hasil.xlsx"
        df_final.to_excel(output, index=False)

    st.success("✅ Pipeline Completed!")

    with open(output, "rb") as f:
        st.download_button("📥 Download Result", f, file_name=output)
import streamlit as st
import pandas as pd
from datetime import datetime
from rapid import (
    preclean_sijk,
    merge_sijk_to_sf,
    merge_lpse_to_sf
)

# =====================================================
# CONFIG
# =====================================================
TAHUN_REVISI = str(datetime.now().year)

st.set_page_config(
    page_title="Pipeline LPSE-SIJK",
    layout="wide"
)

st.title("📊 Data Integration Pipeline")
st.markdown("### SF + LPSE + SIJK Matching System")


# =====================================================
# FILE UPLOAD
# =====================================================
col1, col2, col3 = st.columns(3)

with col1:
    sf_file = st.file_uploader(
        "Upload File Utama (SF)",
        type=["xlsx"]
    )

with col2:
    lpse_file = st.file_uploader(
        "Upload File LPSE",
        type=["xlsx"]
    )

with col3:
    sijk_file = st.file_uploader(
        "Upload File SIJK",
        type=["xlsx"]
    )


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def normalize_columns(df):
    """
    Normalize column names:
    - lowercase
    - strip spaces
    """
    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


# =====================================================
# STANDARDIZE LPSE
# =====================================================
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

    # rename columns
    df = df.rename(columns=mapping)

    # remove duplicated columns
    df = df.loc[:, ~df.columns.duplicated()]

    # metadata
    df["tahun_revisi"] = TAHUN_REVISI
    df["kategori"] = "F"
    df["sumber_data"] = "1"

    # safe deduplicate
    if "kd_penyedia" in df.columns:

        df = df.drop_duplicates(
            subset=["kd_penyedia"]
        )

    return df


# =====================================================
# STANDARDIZE SIJK
# =====================================================
def standardize_sijk(df):

    mapping = {
        "nama_bu": "nama_perusahaan",
        "npwp_bu": "npwp",
        "telepon_bu": "no_telp",
        "email_bu": "email",
        "alamat_bu": "alamat_perusahaan",
        "nib": "nib",
        "kdprov": "prov",
        "kd_kab": "kab",
        "sub_klasifikasi": "pekerjaan_utama",
        "bentuk_usaha_bu": "badan_usaha",
        "nmprov": "nm_prov",
        "nmkab": "nm_kab",
        "kualifikasi_final": "kualifikasi",
        "skala_usaha": "skala_usaha",
    }

    # rename columns
    df = df.rename(columns=mapping)

    # remove duplicated columns
    df = df.loc[:, ~df.columns.duplicated()]

    # metadata
    df["tahun_revisi"] = TAHUN_REVISI
    df["kategori"] = "F"
    df["sumber_data"] = "2"

    return df


# =====================================================
# VALUE MAPPING
# =====================================================
def mapping_values(df):

    # =============================
    # KUALIFIKASI
    # =============================
    map_kualifikasi = {
        "spesialis": "1",
        "kecil": "2",
        "menengah": "3",
        "besar": "4",
        "tidak memiliki sbu aktif": "9",
    }

    if "kualifikasi" in df.columns:

        df["kualifikasi"] = (
            df["kualifikasi"]
            .astype(str)
            .str.lower()
            .map(map_kualifikasi)
            .fillna(df["kualifikasi"])
        )

    # =============================
    # SKALA USAHA
    # =============================
    map_skala = {
        "kecil": "2",
        "menengah": "3",
        "besar": "4",
    }

    if "skala_usaha" in df.columns:

        df["skala_usaha"] = (
            df["skala_usaha"]
            .astype(str)
            .str.lower()
            .map(map_skala)
            .fillna(df["skala_usaha"])
        )

    # =============================
    # BADAN USAHA
    # =============================
    map_badan = {
        "pt. persero (bumn/bumd)": "1",
        "pt": "2",
        "cv": "3",
        "koperasi": "4",
        "kantor perwakilan bujka": "5",
    }

    if "badan_usaha" in df.columns:

        df["badan_usaha"] = (
            df["badan_usaha"]
            .astype(str)
            .str.lower()
            .map(map_badan)
            .fillna("9")
        )

    return df


# =====================================================
# SAFE FUNCTION HANDLER
# =====================================================
def safe_function_result(result):
    """
    Handle:
    - dataframe only
    - tuple(dataframe, preview)
    """

    if isinstance(result, tuple):

        if len(result) >= 2:
            return result[0], result[1]

        return result[0], None

    return result, None


# =====================================================
# MAIN BUTTON
# =====================================================
if st.button("🚀 Run Pipeline"):

    # =================================================
    # VALIDATION
    # =================================================
    if not all([sf_file, lpse_file, sijk_file]):

        st.error(
            "⚠️ Please upload all required files"
        )

        st.stop()

    # =================================================
    # PROCESS
    # =================================================
    with st.spinner("Processing pipeline..."):

        # =============================================
        # LOAD FILES
        # =============================================
        df_utama = pd.read_excel(
            sf_file,
            dtype=str
        )

        df_lpse = pd.read_excel(
            lpse_file,
            dtype=str
        )

        df_sijk = pd.read_excel(
            sijk_file,
            dtype=str
        )

        # =============================================
        # NORMALIZE COLUMN NAMES
        # =============================================
        df_utama = normalize_columns(df_utama)
        df_lpse = normalize_columns(df_lpse)
        df_sijk = normalize_columns(df_sijk)

        # =============================================
        # SAVE ORIGINAL SF ORDER
        # =============================================
        df_utama["_sf_order"] = range(
            len(df_utama)
        )

        # =============================================
        # STANDARDIZATION
        # =============================================
        lpse_std = standardize_lpse(df_lpse)
        sijk_std = standardize_sijk(df_sijk)

        # =============================================
        # ALIGN STRUCTURE
        # =============================================
        target_columns = df_utama.columns.tolist()

        lpse_std = lpse_std.reindex(
            columns=target_columns
        )

        sijk_std = sijk_std.reindex(
            columns=target_columns
        )

        # =============================================
        # PRE-CLEAN SIJK
        # =============================================
        result_preclean = preclean_sijk(
            sijk_std,
            threshold=91,
            preview=True
        )

        sijk_clean, preview_sijk_internal = (
            safe_function_result(
                result_preclean
            )
        )

        # =============================================
        # MERGE SIJK -> SF
        # =============================================
        result_merge_sijk = merge_sijk_to_sf(
            df_utama,
            sijk_clean,
            threshold=91,
            preview=True
        )

        df_merge, preview_sijk_merge = (
            safe_function_result(
                result_merge_sijk
            )
        )

        # =============================================
        # MERGE LPSE -> SF
        # =============================================
        result_merge_lpse = merge_lpse_to_sf(
            df_merge,
            lpse_std,
            threshold=91,
            preview=True
        )

        df_final, preview_lpse_merge = (
            safe_function_result(
                result_merge_lpse
            )
        )

        # =============================================
        # FORCE ALL TEXT
        # =============================================
        cols_text = df_final.columns.difference(
            ["_sf_order"]
        )

        df_final[cols_text] = (
            df_final[cols_text]
            .fillna("")
            .astype(str)
        )

        # =============================================
        # NORMALIZE TEXT
        # =============================================
        for col in [
            "nama_perusahaan",
            "nm_prov",
            "nm_kab"
        ]:

            if col in df_final.columns:

                df_final[col] = (
                    df_final[col]
                    .astype(str)
                    .str.upper()
                    .str.strip()
                )

        # =============================================
        # SPLIT SF VS NON-SF
        # =============================================
        df_final["_is_sf"] = (
            df_final["_sf_order"].notna()
        )

        data_sf = df_final[
            df_final["_is_sf"]
        ].copy()

        data_non_sf = df_final[
            ~df_final["_is_sf"]
        ].copy()

        # =============================================
        # PRIORITY
        # =============================================
        priority_map = {
            "2": 1,  # SIJK
            "1": 2   # LPSE
        }

        data_non_sf["_priority"] = (
            data_non_sf["sumber_data"]
            .map(priority_map)
            .fillna(99)
        )

        # =============================================
        # SORT PRIORITY
        # =============================================
        data_non_sf = data_non_sf.sort_values(
            "_priority"
        )

        # =============================================
        # DEDUPLICATE NON-SF
        # =============================================
        data_non_sf = data_non_sf.drop_duplicates(
            subset=[
                "nama_perusahaan",
                "nm_prov",
                "nm_kab"
            ],
            keep="first"
        )

        # =============================================
        # REMOVE HELPER
        # =============================================
        data_non_sf = data_non_sf.drop(
            columns=["_priority"],
            errors="ignore"
        )

        # =============================================
        # RECOMBINE
        # =============================================
        df_final = pd.concat(
            [data_sf, data_non_sf],
            ignore_index=True
        )

        # =============================================
        # VALUE MAPPING
        # =============================================
        df_final = mapping_values(
            df_final
        )

        # =============================================
        # RESTORE ORIGINAL SF ORDER
        # =============================================
        data_sf = df_final[
            df_final["_is_sf"]
        ].copy()

        data_non_sf = df_final[
            ~df_final["_is_sf"]
        ].copy()

        if "_sf_order" in data_sf.columns:

            data_sf["_sf_order"] = pd.to_numeric(
                data_sf["_sf_order"],
                errors="coerce"
            )

            data_sf = data_sf.sort_values(
                "_sf_order"
            )

        # =============================================
        # FINAL COMBINE
        # =============================================
        df_final = pd.concat(
            [data_sf, data_non_sf],
            ignore_index=True
        )

        # =============================================
        # INSERTED DATA SUMMARY
        # =============================================
        inserted_data = df_final[
            ~df_final["_is_sf"]
        ].copy()

        total_inserted = len(
            inserted_data
        )

        # =============================================
        # SUMMARY BY KABUPATEN
        # =============================================
        if (
            len(inserted_data) > 0
            and "nm_kab" in inserted_data.columns
        ):

            summary_group_cols = []

            if "kab" in inserted_data.columns:
                summary_group_cols.append("kab")

            if "nm_kab" in inserted_data.columns:
                summary_group_cols.append("nm_kab")

            if len(summary_group_cols) > 0:

                summary_kab = (
                    inserted_data
                    .groupby(summary_group_cols)
                    .size()
                    .reset_index(name="jumlah_insert")
                    .sort_values(
                        "jumlah_insert",
                        ascending=False
                    )
                    .rename(
                        columns={
                            "kab": "kode_kabupaten",
                            "nm_kab": "nama_kabupaten"
                        }
                    )
                )

            else:

                summary_kab = pd.DataFrame()

        # =============================================
        # MATCHING AUDIT
        # =============================================
        audit_frames = []

        if preview_sijk_merge is not None:

            preview_sijk_merge["source"] = (
                "SIJK"
            )

            audit_frames.append(
                preview_sijk_merge
            )

        if preview_lpse_merge is not None:

            preview_lpse_merge["source"] = (
                "LPSE"
            )

            audit_frames.append(
                preview_lpse_merge
            )

        if len(audit_frames) > 0:

            matching_audit = pd.concat(
                audit_frames,
                ignore_index=True
            )

        else:

            matching_audit = pd.DataFrame()

        # =============================================
        # DROP HELPER COLUMNS
        # =============================================
        df_final = df_final.drop(
            columns=[
                "_sf_order",
                "_is_sf"
            ],
            errors="ignore"
        )

        # =============================================
        # SAVE FINAL OUTPUT
        # =============================================
        hasil_file = (
            "hasil_pipeline.xlsx"
        )

        df_final.to_excel(
            hasil_file,
            index=False
        )

        # =============================================
        # SAVE MATCHING AUDIT
        # =============================================
        audit_file = (
            "matching_audit.xlsx"
        )

        matching_audit.to_excel(
            audit_file,
            index=False
        )

    # =================================================
    # PROCESS COMPLETED
    # =================================================
    st.success(
        "✅ Pipeline completed successfully!"
    )

    # =================================================
    # SUMMARY METRICS
    # =================================================
    st.subheader(
        "📌 Summary Information"
    )

    st.metric(
        label="Total Perusahaan Inserted",
        value=f"{total_inserted:,}"
    )

    # =================================================
    # SUMMARY TABLE
    # =================================================
    st.subheader(
        "📍 Inserted Perusahaan by Kabupaten"
    )

    st.dataframe(
        summary_kab,
        use_container_width=True
    )

    # =================================================
    # MATCHING AUDIT PREVIEW
    # =================================================
    st.subheader(
        "🔍 Matching Audit Preview"
    )

    if len(matching_audit) > 0:

        st.dataframe(
            matching_audit.head(100),
            use_container_width=True
        )

    else:

        st.info(
            "No matching audit data available."
        )

    # =================================================
    # DOWNLOAD FINAL RESULT
    # =================================================
    with open(
        hasil_file,
        "rb"
    ) as f:

        st.download_button(
            label="📥 Download Final Result",
            data=f,
            file_name=hasil_file,
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            )
        )

    # =================================================
    # DOWNLOAD MATCHING AUDIT
    # =================================================
    with open(
        audit_file,
        "rb"
    ) as f:

        st.download_button(
            label="📥 Download Matching Audit",
            data=f,
            file_name=audit_file,
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            )
        )
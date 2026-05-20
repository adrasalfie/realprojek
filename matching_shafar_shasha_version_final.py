import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
from rapidfuzz import fuzz
from io import BytesIO

# =====================================================
# CONFIG
# =====================================================
TAHUN_REVISI = str(datetime.now().year)

st.set_page_config(
    page_title="Entity Matching Pipeline",
    layout="wide"
)

st.title("📊 SF - SIJK - LPSE Matching Pipeline")
st.markdown("### Interactive Fuzzy Matching & Entity Resolution System")

# =====================================================
# FILE UPLOAD
# =====================================================
col1, col2, col3 = st.columns(3)

with col1:
    sf_file = st.file_uploader(
        "Upload SF File",
        type=["xlsx"]
    )

with col2:
    sijk_file = st.file_uploader(
        "Upload SIJK File",
        type=["xlsx"]
    )

with col3:
    lpse_file = st.file_uploader(
        "Upload LPSE File",
        type=["xlsx"]
    )

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def load_excel(file):
    return pd.read_excel(file, dtype=str)


def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def normalize_columns(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


def normalize_address(addr):

    if pd.isna(addr):
        return ""

    addr = str(addr).upper().strip()

    addr = re.sub(r'\bRT\.?\s*\d+', '', addr)
    addr = re.sub(r'\bRW\.?\s*\d+', '', addr)

    prefix_words = [
        'KELURAHAN', 'KECAMATAN',
        'KEL.', 'KEC.',
        'KEL', 'KEC',
        'DESA', 'DS.',
        'DS', 'KOTA',
        'JALAN', 'JLN.',
        'JLN', 'JL.',
        'JL', 'NO.',
        'NOMOR', 'NO'
    ]

    for word in sorted(prefix_words, key=len, reverse=True):

        addr = re.sub(
            r'\b' + re.escape(word) + r'\b',
            ' ',
            addr
        )

    addr = re.sub(r'\d+', '', addr)
    addr = re.sub(r'[^\w\s]', ' ', addr)

    addr = ' '.join(addr.split())

    return addr.strip()


def make_kdkab(df, prov_col, kab_col, output_col):

    df = df.copy()

    df[output_col] = (
        df[prov_col]
        .astype(str)
        .str.strip()
        .str.zfill(2)
        +
        df[kab_col]
        .astype(str)
        .str.strip()
        .str.zfill(2)
    )

    return df


def strip_badan_usaha(name):

    tokens_remove = {
        "pt",
        "cv",
        "tbk",
        "ud",
        "fa",
        "firma",
        "koperasi",
        "persero"
    }

    name = re.sub(
        r'[^\w\s]',
        ' ',
        str(name).lower()
    )

    tokens = name.split()

    tokens = [
        t for t in tokens
        if t not in tokens_remove
    ]

    return " ".join(tokens)


def name_similarity(a, b):

    a = strip_badan_usaha(a)
    b = strip_badan_usaha(b)

    set_a = set(a.split())
    set_b = set(b.split())

    if not set_a and not set_b:
        return 100

    if not set_a or not set_b:
        return 0

    intersection = (
        set_a & set_b
    )

    return round(
        2 * len(intersection)
        /
        (len(set_a) + len(set_b))
        * 100,
        2
    )


def weighted_similarity(
    sim_name,
    sim_kab,
    sim_addr,
    len_name,
    len_addr
):

    weights = {
        "name": len_name,
        "kab": 4,
        "addr": len_addr
    }

    total = sum(weights.values())

    score = (
        sim_name * weights["name"]
        +
        sim_kab * weights["kab"]
        +
        sim_addr * weights["addr"]
    )

    return round(score / total, 2)


def decision_rule(
    sim_name,
    sim_kab,
    sim_addr,
    weighted
):

    if sim_kab < 100:
        return "INSERT"

    if sim_name >= 90:
        return "DROP"

    if weighted >= 90:
        return "DROP"

    if 80 <= weighted <= 89:

        if sim_addr >= 90:
            return "DROP"

        return "INSERT"

    return "INSERT"


# =====================================================
# PRE-CLEAN SIJK
# =====================================================
def preclean_sijk(
    df,
    threshold=75
):

    df = df.copy()

    df["_name"] = (
        df["nama_bu"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["_addr"] = (
        df["alamat_bu"]
        .astype(str)
        .apply(normalize_address)
    )

    rows_drop = set()
    preview_rows = []

    for (_, _), group in df.groupby(
        ["nmprov", "nmkab"]
    ):

        idxs = list(group.index)

        for i in range(len(idxs)):

            idx_i = idxs[i]

            if idx_i in rows_drop:
                continue

            txt_i = (
                df.loc[idx_i, "_name"]
                +
                " "
                +
                df.loc[idx_i, "_addr"]
            )

            for j in range(i + 1, len(idxs)):

                idx_j = idxs[j]

                if idx_j in rows_drop:
                    continue

                txt_j = (
                    df.loc[idx_j, "_name"]
                    +
                    " "
                    +
                    df.loc[idx_j, "_addr"]
                )

                sim = fuzz.token_set_ratio(
                    txt_i,
                    txt_j
                )

                if sim >= threshold:

                    rows_drop.add(idx_j)

                    preview_rows.append({
                        "nama_1": df.loc[idx_i, "nama_bu"],
                        "nama_2": df.loc[idx_j, "nama_bu"],
                        "similarity": sim,
                        "decision": "DROP_DUPLICATE"
                    })

    df = df.drop(index=list(rows_drop))

    preview_df = pd.DataFrame(preview_rows)

    return (
        df.reset_index(drop=True),
        preview_df
    )


# =====================================================
# MATCH FUNCTION
# =====================================================
def process_matching(
    df_source,
    df_sf,
    source_name_col,
    source_addr_col,
    source_kdkab_col,
    source_label
):

    inserted_rows = []
    audit_rows = []

    for idx, row in df_source.iterrows():

        source_name = safe_str(
            row[source_name_col]
        )

        source_addr = safe_str(
            row[source_addr_col]
        )

        source_kdkab = safe_str(
            row[source_kdkab_col]
        )

        same_kab = df_sf[
            df_sf["kdkab_sf"]
            ==
            source_kdkab
        ]

        best_score = -1
        best_sf = None
        best_sim_name = 0
        best_sim_addr = 0

        for sf_idx, sf_row in same_kab.iterrows():

            sf_name = safe_str(
                sf_row["nama_perusahaan"]
            )

            sf_addr = safe_str(
                sf_row["alamat_perusahaan"]
            )

            sim_name = name_similarity(
                source_name,
                sf_name
            )

            sim_addr = fuzz.token_set_ratio(
                normalize_address(source_addr),
                normalize_address(sf_addr)
            )

            sim_kab = 100

            weighted = weighted_similarity(
                sim_name,
                sim_kab,
                sim_addr,
                len(source_name),
                len(source_addr)
            )

            if weighted > best_score:

                best_score = weighted
                best_sf = sf_row
                best_sim_name = sim_name
                best_sim_addr = sim_addr

        # no candidate
        if best_sf is None:

            decision = "INSERT"

            audit_rows.append({
                "source": source_label,
                "source_name": row[source_name_col],
                "matched_sf": None,
                "similarity_name": 0,
                "similarity_address": 0,
                "weighted_similarity": 0,
                "decision": decision
            })

            inserted_rows.append(row)

            continue

        decision = decision_rule(
            best_sim_name,
            100,
            best_sim_addr,
            best_score
        )

        audit_rows.append({
            "source": source_label,
            "source_name": row[source_name_col],
            "matched_sf": best_sf["nama_perusahaan"],
            "similarity_name": best_sim_name,
            "similarity_address": best_sim_addr,
            "weighted_similarity": best_score,
            "decision": decision
        })

        if decision == "INSERT":

            inserted_rows.append(row)

    return (
        pd.DataFrame(inserted_rows),
        pd.DataFrame(audit_rows)
    )


# =====================================================
# RUN BUTTON
# =====================================================
if st.button("🚀 Run Pipeline"):

    if not all([
        sf_file,
        sijk_file,
        lpse_file
    ]):

        st.error(
            "Please upload all files."
        )

        st.stop()

    with st.spinner("Processing..."):

        # =============================================
        # LOAD
        # =============================================
        df_sf = load_excel(sf_file)
        df_sijk = load_excel(sijk_file)
        df_lpse = load_excel(lpse_file)

        # =============================================
        # NORMALIZE
        # =============================================
        df_sf = normalize_columns(df_sf)
        df_sijk = normalize_columns(df_sijk)
        df_lpse = normalize_columns(df_lpse)

        # =============================================
        # CREATE KDKAB
        # =============================================
        df_sf = make_kdkab(
            df_sf,
            "prov",
            "kab",
            "kdkab_sf"
        )

        df_sijk = make_kdkab(
            df_sijk,
            "kdprov",
            "kd_kab",
            "kdkab"
        )

        df_lpse = make_kdkab(
            df_lpse,
            "kdprov",
            "kd_kab",
            "kdkab"
        )

        # =============================================
        # PRE-CLEAN SIJK
        # =============================================
        df_sijk_clean, preview_dup = (
            preclean_sijk(df_sijk)
        )

        # =============================================
        # MATCH SIJK
        # =============================================
        insert_sijk, audit_sijk = (
            process_matching(
                df_sijk_clean,
                df_sf,
                "nama_bu",
                "alamat_bu",
                "kdkab",
                "SIJK"
            )
        )

        # =============================================
        # MATCH LPSE
        # =============================================
        insert_lpse, audit_lpse = (
            process_matching(
                df_lpse,
                df_sf,
                "nama_penyedia",
                "alamat",
                "kdkab",
                "LPSE"
            )
        )

        # =============================================
        # COMBINE INSERT
        # =============================================
        insert_sijk["sumber_data"] = "2"
        insert_lpse["sumber_data"] = "1"

        df_insert = pd.concat([
            insert_sijk,
            insert_lpse
        ])

        # =============================================
        # PRIORITY DEDUP
        # =============================================
        priority_map = {
            "2": 1,
            "1": 2
        }

        df_insert["_priority"] = (
            df_insert["sumber_data"]
            .map(priority_map)
        )

        df_insert = df_insert.sort_values(
            "_priority"
        )

        # dynamic name column
        if "nama_bu" in df_insert.columns:
            df_insert["nama_final"] = (
                df_insert["nama_bu"]
                .fillna(
                    df_insert.get(
                        "nama_penyedia",
                        ""
                    )
                )
            )
        else:
            df_insert["nama_final"] = (
                df_insert["nama_penyedia"]
            )

        df_insert = df_insert.drop_duplicates(
            subset=[
                "nama_final",
                "kdkab"
            ],
            keep="first"
        )

        # =============================================
        # AUDIT
        # =============================================
        audit_df = pd.concat([
            audit_sijk,
            audit_lpse
        ])

        # =============================================
        # CONVERT INSERT DATA TO SF TEMPLATE
        # =============================================

        df_insert_sf = pd.DataFrame(
            index=df_insert.index,
            columns=df_sf.columns
        )

        # =============================================
        # MAP SIJK
        # =============================================
        if "nama_bu" in df_insert.columns:

            mask_sijk = (
                df_insert["sumber_data"] == "2"
            )

            df_insert_sf.loc[mask_sijk, "nama_perusahaan"] = (
                df_insert.loc[mask_sijk, "nama_bu"]
            )

            df_insert_sf.loc[mask_sijk, "alamat_perusahaan"] = (
                df_insert.loc[mask_sijk, "alamat_bu"]
            )

            df_insert_sf.loc[mask_sijk, "npwp"] = (
                df_insert.loc[mask_sijk, "npwp_bu"]
            )

            df_insert_sf.loc[mask_sijk, "prov"] = (
                df_insert.loc[mask_sijk, "kdprov"]
            )

            df_insert_sf.loc[mask_sijk, "kab"] = (
                df_insert.loc[mask_sijk, "kd_kab"]
            )

            df_insert_sf.loc[mask_sijk, "nm_prov"] = (
                df_insert.loc[mask_sijk, "nmprov"]
            )

            df_insert_sf.loc[mask_sijk, "nm_kab"] = (
                df_insert.loc[mask_sijk, "nmkab"]
            )

            df_insert_sf.loc[mask_sijk, "badan_usaha"] = (
                df_insert.loc[mask_sijk, "bentuk_usaha_bu"]
            )

            df_insert_sf.loc[mask_sijk, "kualifikasi"] = (
                df_insert.loc[mask_sijk, "kualifikasi_final"]
            )

            df_insert_sf.loc[mask_sijk, "skala_usaha"] = (
                df_insert.loc[mask_sijk, "skala_usaha"]
            )

            df_insert_sf.loc[mask_sijk, "no_telp"] = (
                df_insert.loc[mask_sijk, "telepon_bu"]
            )

            df_insert_sf.loc[mask_sijk, "email"] = (
                df_insert.loc[mask_sijk, "email_bu"]
            )

            df_insert_sf.loc[mask_sijk, "nib"] = (
                df_insert.loc[mask_sijk, "nib"]
            )

            df_insert_sf.loc[mask_sijk, "pekerjaan_utama"] = (
                df_insert.loc[mask_sijk, "sub_klasifikasi"]
            )


        # =============================================
        # MAP LPSE
        # =============================================
        if "nama_penyedia" in df_insert.columns:

            mask_lpse = (
                df_insert["sumber_data"] == "1"
            )

            df_insert_sf.loc[mask_lpse, "nama_perusahaan"] = (
                df_insert.loc[mask_lpse, "nama_penyedia"]
            )

            df_insert_sf.loc[mask_lpse, "alamat_perusahaan"] = (
                df_insert.loc[mask_lpse, "alamat"]
            )

            df_insert_sf.loc[mask_lpse, "npwp"] = (
                df_insert.loc[mask_lpse, "npwp_penyedia"]
            )

            df_insert_sf.loc[mask_lpse, "prov"] = (
                df_insert.loc[mask_lpse, "kdprov"]
            )

            df_insert_sf.loc[mask_lpse, "kab"] = (
                df_insert.loc[mask_lpse, "kd_kab"]
            )

            df_insert_sf.loc[mask_lpse, "nm_prov"] = (
                df_insert.loc[mask_lpse, "nmprov"]
            )

            df_insert_sf.loc[mask_lpse, "nm_kab"] = (
                df_insert.loc[mask_lpse, "nmkab"]
            )

            df_insert_sf.loc[mask_lpse, "badan_usaha"] = (
                df_insert.loc[mask_lpse, "bentuk_usaha"]
            )

            df_insert_sf.loc[mask_lpse, "kualifikasi"] = (
                df_insert.loc[mask_lpse, "kualifikasi_usaha"]
            )

            df_insert_sf.loc[mask_lpse, "no_telp"] = (
                df_insert.loc[mask_lpse, "telepon"]
            )

            df_insert_sf.loc[mask_lpse, "fax"] = (
                df_insert.loc[mask_lpse, "fax"]
            )

            df_insert_sf.loc[mask_lpse, "website"] = (
                df_insert.loc[mask_lpse, "website"]
            )

            df_insert_sf.loc[mask_lpse, "nib"] = (
                df_insert.loc[mask_lpse, "nomor_izin_usaha"]
            )

            df_insert_sf.loc[mask_lpse, "kbli"] = (
                df_insert.loc[mask_lpse, "kd_klasifikasi"]
            )


        # =============================================
        # METADATA
        # =============================================
        df_insert_sf["tahun_revisi"] = (
            TAHUN_REVISI
        )

        df_insert_sf["kategori"] = "F"

        df_insert_sf["sumber_data"] = (
            df_insert["sumber_data"]
            .values
        )

        df_insert_sf = (
            df_insert_sf
            .fillna("")
        )
        
        # =============================================
        # FINAL SF
        # =============================================
        df_final = pd.concat([
            df_sf,
            df_insert_sf
        ])

        # =============================================
        # MAP KUALIFIKASI & SKALA_USAHA TO ANGKA
        # =============================================
        map_kualifikasi = {
            "spesialis":               "1",
            "kecil":                   "2",
            "menengah":                "3",
            "besar":                   "4",
            "tidak memiliki sbu aktif": "9"
        }

        map_skala = {
            "kecil":   "2",
            "menengah": "3",
            "besar":   "4"
        }

        map_badan_usaha_exact = {
            "persero":                    "1",
            "pt":                         "2",
            "cv":                         "3",
            "commanditer":                "3",
            "persekutuan komanditer":     "3",
            "koperasi":                   "4",
            "kantor perwakilan bujka":    "5"
        }

        # regex patterns: urutan dari paling spesifik ke umum
        badan_usaha_patterns = [
            (r"persero",                                   "1"),
            (r"\bpt\b|perseroan terbatas",                 "2"),
            (r"\bcv\b|commanditer|persekutuan komanditer", "3"),
            (r"koperasi",                                  "4"),
            (r"kantor perwakilan bujka",                   "5"),
        ]

        def map_badan_usaha(val):
            v = str(val).strip().lower()
            if v in map_badan_usaha_exact:
                return map_badan_usaha_exact[v]
            for pattern, kode in badan_usaha_patterns:
                if re.search(pattern, v):
                    return kode
            return val

        if "kualifikasi" in df_final.columns:
            df_final["kualifikasi"] = (
                df_final["kualifikasi"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(map_kualifikasi)
                .fillna(df_final["kualifikasi"])
            )

        if "skala_usaha" in df_final.columns:
            df_final["skala_usaha"] = (
                df_final["skala_usaha"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(map_skala)
                .fillna(df_final["skala_usaha"])
            )

        if "badan_usaha" in df_final.columns:
            df_final["badan_usaha"] = (
                df_final["badan_usaha"]
                .apply(map_badan_usaha)
            )

        # =============================================
        # SUMMARY
        # =============================================
        total_insert = len(df_insert)

        st.success(
            "Pipeline completed successfully!"
        )

        st.subheader(
            "📌 Summary Information"
        )

        st.metric(
            "Total Perusahaan Inserted",
            f"{total_insert:,}"
        )

        # =============================================
        # SUMMARY BY KAB
        # =============================================
        summary_group_cols = []

        if "kd_kab" in df_insert.columns:
            summary_group_cols.append(
                "kd_kab"
            )

        if "nmkab" in df_insert.columns:
            summary_group_cols.append(
                "nmkab"
            )

        if len(summary_group_cols) > 0:

            summary_kab = (
                df_insert
                .groupby(summary_group_cols)
                .size()
                .reset_index(
                    name="jumlah_insert"
                )
                .sort_values(
                    "jumlah_insert",
                    ascending=False
                )
            )

            st.subheader(
                "📍 Insert by Kabupaten"
            )

            st.dataframe(
                summary_kab,
                use_container_width=True
            )

        # =============================================
        # AUDIT PREVIEW
        # =============================================
        st.subheader(
            "🔍 Matching Audit Preview"
        )

        st.dataframe(
            audit_df.head(100),
            use_container_width=True
        )

        # =============================================
        # CREATE DOWNLOAD FILES
        # =============================================
        output_final = BytesIO()
        output_audit = BytesIO()

        with pd.ExcelWriter(
            output_final,
            engine="openpyxl"
        ) as writer:

            df_final.to_excel(
                writer,
                index=False,
                sheet_name="Final_SF"
            )

            preview_dup.to_excel(
                writer,
                index=False,
                sheet_name="SIJK_Duplicate_Preview"
            )

        with pd.ExcelWriter(
            output_audit,
            engine="openpyxl"
        ) as writer:

            audit_df.to_excel(
                writer,
                index=False,
                sheet_name="Audit"
            )

        output_final.seek(0)
        output_audit.seek(0)

        # =============================================
        # DOWNLOAD BUTTONS
        # =============================================
        col1, col2 = st.columns(2)

        # =============================================
        # GENERATE DYNAMIC FILE NAME
        # =============================================

        prov_code = "unknown"

        # priority from SF
        if "prov" in df_sf.columns:

            prov_unique = (
                df_sf["prov"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
            )

            if len(prov_unique) > 0:
                prov_code = prov_unique[0]

        # clean filename
        prov_code = (
            str(prov_code)
            .replace("/", "_")
            .replace("\\", "_")
            .strip()
        )

        hasil_filename = (
            f"{prov_code}_hasil_pipeline.xlsx"
        )

        audit_filename = (
            f"{prov_code}_matching_audit.xlsx"
        )

        with col1:

            st.download_button(
                label="📥 Download Final Result",
                data=output_final,
                file_name=hasil_filename,
                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                )
            )

        with col2:

            st.download_button(
                label="📥 Download Matching Audit",
                data=output_audit,
                file_name=audit_filename,
                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                )
            )
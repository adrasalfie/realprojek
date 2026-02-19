import pandas as pd
from rapidfuzz import fuzz


# ==============================
# 1. PRE-CLEANING SIJK INTERNAL
# ==============================
def preclean_sijk(sijk: pd.DataFrame, threshold: int = 75, preview: bool = False):
    sijk_clean = sijk.copy()

    preview_rows = []

    for col in ["nama_perusahaan", "alamat_perusahaan"]:
        if col in sijk_clean.columns:
            sijk_clean[col + "_norm"] = sijk_clean[col].str.upper().str.strip()

    rows_to_drop = set()

    for (kdprov, kd_kab), group in sijk_clean.groupby(["nm_prov", "nm_kab"]):
        for i, row_i in group.iterrows():
            if i in rows_to_drop:
                continue

            for j, row_j in group.iterrows():
                if i >= j or j in rows_to_drop:
                    continue

                similarity = fuzz.token_set_ratio(
                    f"{row_i['nama_perusahaan_norm']} {row_i['alamat_perusahaan_norm']}",
                    f"{row_j['nama_perusahaan_norm']} {row_j['alamat_perusahaan_norm']}"
                )

                if similarity >= threshold:
                    rows_to_drop.add(j)

                    preview_rows.append({
                        "kd_prov_1": row_i.get("prov", ""),
                        "kd_prov_2": row_j.get("prov", ""),
                        "kd_kab_1": row_i.get("kab", ""),
                        "kd_kab_2": row_j.get("kab", ""),

                        "nama_1": row_i["nama_perusahaan"],
                        "alamat_1": row_i["alamat_perusahaan"],
                        "nama_2": row_j["nama_perusahaan"],
                        "alamat_2": row_j["alamat_perusahaan"],

                        "similarity": similarity,
                        "aksi": "DROP_DUPLICATE_SIJK"
                    })

    sijk_clean = sijk_clean.drop(index=list(rows_to_drop))

    sijk_clean = sijk_clean.drop(
        columns=[c for c in sijk_clean.columns if c.endswith("_norm")],
        errors="ignore"
    )

    preview_df = pd.DataFrame(preview_rows)

    return (sijk_clean, preview_df) if preview else sijk_clean


# ==============================
# 2. MERGE SIJK KE SF
# ==============================
def merge_sijk_to_sf(sf, sijk, threshold=75, preview=False):
    sf_copy = sf.copy()
    sijk_copy = sijk.copy()

    preview_rows = []

    # NORMALISASI UNTUK FUZZY
    for col in ["nama_perusahaan", "alamat_perusahaan"]:
        if col in sf_copy.columns:
            sf_copy[col + "_norm"] = sf_copy[col].str.upper().str.strip()
        if col in sijk_copy.columns:
            sijk_copy[col + "_norm"] = sijk_copy[col].str.upper().str.strip()

    # ITERASI DATA SIJK
    for _, sijk_row in sijk_copy.iterrows():
        kdprov = sijk_row.get("prov", "")
        kdkab = sijk_row.get("kab", "")

        # Filter kandidat SF berdasarkan KODE wilayah
        candidates = sf_copy[
            (sf_copy["prov"] == kdprov) &
            (sf_copy["kab"] == kdkab)
        ]

        matched = False

        for sf_idx, sf_row in candidates.iterrows():
            similarity = fuzz.token_set_ratio(
                f"{sijk_row['nama_perusahaan_norm']} {sijk_row['alamat_perusahaan_norm']}",
                f"{sf_row['nama_perusahaan_norm']} {sf_row['alamat_perusahaan_norm']}"
            )

            if similarity >= threshold:
                # ENRICH
                for col in sf_copy.columns:
                    if col in sijk_copy.columns and sf_copy.at[sf_idx, col] == "" and sijk_row[col] != "":
                        sf_copy.at[sf_idx, col] = sijk_row[col]

                matched = True

                preview_rows.append({
                    "kd_prov": kdprov,
                    "kd_kab": kdkab,
                    "nama_sijk": sijk_row["nama_perusahaan"],
                    "nama_sf": sf_row["nama_perusahaan"],
                    "alamat_sijk": sijk_row["alamat_perusahaan"],
                    "alamat_sf": sf_row["alamat_perusahaan"],
                    "similarity": similarity,
                    "aksi": "ENRICH_SF"
                })
                break

        # JIKA TIDAK MATCH → APPEND KE SF
        if not matched:
            sf_copy = pd.concat(
                [sf_copy, sijk_row.to_frame().T], ignore_index=True)

            preview_rows.append({
                "kd_prov": kdprov,
                "kd_kab": kdkab,
                "nama_sijk": sijk_row["nama_perusahaan"],
                "nama_sf": "",
                "alamat_sijk": sijk_row["alamat_perusahaan"],
                "alamat_sf": "",
                "similarity": 0,
                "aksi": "APPEND_NEW"
            })

    sf_copy = sf_copy.drop(
        columns=[c for c in sf_copy.columns if c.endswith("_norm")],
        errors="ignore"
    )

    preview_df = pd.DataFrame(preview_rows)

    return (sf_copy, preview_df) if preview else sf_copy


# ==============================
# 3. MERGE LPSE KE SF
# ==============================
def merge_lpse_to_sf(sf, lpse, threshold=75, preview=False):
    sf_copy = sf.copy()
    lpse_copy = lpse.copy()

    preview_rows = []

    # NORMALISASI UNTUK FUZZY
    for col in ["nama_perusahaan", "alamat_perusahaan"]:
        if col in sf_copy.columns:
            sf_copy[col + "_norm"] = sf_copy[col].str.upper().str.strip()
        if col in lpse_copy.columns:
            lpse_copy[col + "_norm"] = lpse_copy[col].str.upper().str.strip()

    # ITERASI DATA LPSE
    for _, lpse_row in lpse_copy.iterrows():
        kdprov = lpse_row.get("prov", "")
        kdkab = lpse_row.get("kab", "")

        # Filter kandidat SF berdasarkan wilayah
        candidates = sf_copy[
            (sf_copy["prov"] == kdprov) &
            (sf_copy["kab"] == kdkab)
        ]

        matched = False

        for sf_idx, sf_row in candidates.iterrows():
            similarity = fuzz.token_set_ratio(
                f"{lpse_row['nama_perusahaan_norm']} {lpse_row['alamat_perusahaan_norm']}",
                f"{sf_row['nama_perusahaan_norm']} {sf_row['alamat_perusahaan_norm']}"
            )

            if similarity >= threshold:
                # ENRICH
                for col in sf_copy.columns:
                    if col in lpse_copy.columns and sf_copy.at[sf_idx, col] == "" and lpse_row[col] != "":
                        sf_copy.at[sf_idx, col] = lpse_row[col]

                matched = True

                preview_rows.append({
                    "kd_prov": kdprov,
                    "kd_kab": kdkab,
                    "nama_lpse": lpse_row["nama_perusahaan"],
                    "nama_sf": sf_row["nama_perusahaan"],
                    "alamat_lpse": lpse_row["alamat_perusahaan"],
                    "alamat_sf": sf_row["alamat_perusahaan"],
                    "similarity": similarity,
                    "aksi": "ENRICH_SF_FROM_LPSE"
                })
                break

        # JIKA TIDAK MATCH → APPEND
        if not matched:
            sf_copy = pd.concat(
                [sf_copy, lpse_row.to_frame().T],
                ignore_index=True
            )

            preview_rows.append({
                "kd_prov": kdprov,
                "kd_kab": kdkab,
                "nama_lpse": lpse_row["nama_perusahaan"],
                "nama_sf": "",
                "alamat_lpse": lpse_row["alamat_perusahaan"],
                "alamat_sf": "",
                "similarity": 0,
                "aksi": "APPEND_NEW_FROM_LPSE"
            })

    sf_copy = sf_copy.drop(
        columns=[c for c in sf_copy.columns if c.endswith("_norm")],
        errors="ignore"
    )

    preview_df = pd.DataFrame(preview_rows)

    return (sf_copy, preview_df) if preview else sf_copy

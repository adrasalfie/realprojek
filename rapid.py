import pandas as pd
from rapidfuzz import fuzz


# ======================================================
# LOGIKA MATCHING:
#
# Kunci match = nama_perusahaan (fuzzy) + prov + kab
#
# PERUBAHAN PENTING vs versi lama:
#   1. Pakai fuzz.RATIO (bukan token_set_ratio) → urutan kata dipertahankan
#      sehingga "MANDIRI DUA" ≠ "DUA MANDIRI" → tidak false match
#   2. Strip badan usaha (CV, PT, dll) sebelum compare nama
#   3. Threshold dinaikkan ke 90 → lebih ketat, aman untuk nama pendek
#   4. Kolom PROTECTED tidak boleh di-overwrite saat ENRICH
#
# | Kondisi                                                       | Aksi   |
# |---------------------------------------------------------------|--------|
# | nama >=90 + prov+kab sama + alamat partial_ratio >=70         | ENRICH |
# | nama >=90 + prov+kab sama + alamat partial_ratio < 70         | INSERT |
# | nama < 90 atau prov/kab beda                                  | INSERT |
#
# Contoh:
#   EKA PUTRA  : "JL GAGAK" vs "Jalan Gagak, Desa..." → partial=93 → ENRICH
#   AGUNG RAYA : "JL TRANS SULAWESI" vs "Jalan Cendrawasih" → partial=47 → INSERT
# ======================================================

# Kolom yang TIDAK BOLEH di-overwrite saat ENRICH
PROTECTED_COLS = {
    "no", "tahun", "id", "kip",
    "prov", "kab", "kec", "kel",
    "nm_prov", "nm_kab", "nm_kec", "nm_kel",
    "nbs", "nsbs", "sls",
    "nama_perusahaan",
    "sumber_data",       # KRITIS: jangan overwrite sumber asli SF
    "tahun_revisi",
    "kategori",
    "_sf_order",
}

# Suffix badan usaha yang distrip sebelum compare nama
_BADAN_USAHA_SUFFIXES = [
    ", CV", ", PT", ", UD", ", FA", ", KOPERASI",
    ",CV", ",PT", ",UD", ",FA",
]


def _strip_badan_usaha(nama: str) -> str:
    """Hapus suffix badan usaha agar compare nama lebih akurat."""
    nama = nama.upper().strip()
    for s in _BADAN_USAHA_SUFFIXES:
        if nama.endswith(s):
            nama = nama[: -len(s)].strip()
            break
    return nama


ALAMAT_THRESHOLD = 70  # partial_ratio alamat: >= 70 ENRICH, < 70 INSERT


def _alamat_score(alamat_a: str, alamat_b: str) -> float:
    """
    Hitung similarity dua alamat pakai partial_ratio.
    Toleran terhadap alamat yang lebih panjang (ada tambahan kelurahan/kecamatan).
    """
    a = str(alamat_a).upper().strip()
    b = str(alamat_b).upper().strip()
    if not a or not b or a == "NAN" or b == "NAN":
        return 0.0
    return fuzz.partial_ratio(a, b)


def _nama_score(nama_a: str, nama_b: str) -> float:
    """
    Hitung similarity dua nama perusahaan.
    - Strip badan usaha dulu (CV, PT, dll)
    - Exact match setelah strip → 100
    - Selain itu pakai fuzz.ratio (urutan kata dipertahankan)
    """
    a = _strip_badan_usaha(nama_a)
    b = _strip_badan_usaha(nama_b)
    if a == b:
        return 100.0
    return fuzz.ratio(a, b)


# ==============================
# 1. PRE-CLEANING SIJK INTERNAL
# ==============================
def preclean_sijk(sijk: pd.DataFrame, threshold: int = 90, preview: bool = False):
    """
    Hapus duplikat INTERNAL di dalam data SIJK sebelum di-merge ke SF.
    Kunci duplikat: nama mirip (ratio >= threshold) + nm_prov + nm_kab.
    """
    sijk_clean = sijk.copy()
    preview_rows = []

    sijk_clean["_nama_norm"] = sijk_clean["nama_perusahaan"].fillna("").str.upper().str.strip()
    sijk_clean["_alamat_norm"] = sijk_clean["alamat_perusahaan"].fillna("").str.upper().str.strip()

    rows_to_drop = set()

    for (nm_prov, nm_kab), group in sijk_clean.groupby(["nm_prov", "nm_kab"]):
        indices = list(group.index)
        for ii, i in enumerate(indices):
            if i in rows_to_drop:
                continue
            row_i = sijk_clean.loc[i]

            for j in indices[ii + 1:]:
                if j in rows_to_drop:
                    continue
                row_j = sijk_clean.loc[j]

                score = _nama_score(row_i["_nama_norm"], row_j["_nama_norm"])
                if score < threshold:
                    continue

                rows_to_drop.add(j)
                preview_rows.append({
                    "prov_1": row_i.get("prov", ""),
                    "prov_2": row_j.get("prov", ""),
                    "kab_1": row_i.get("kab", ""),
                    "kab_2": row_j.get("kab", ""),
                    "nama_1": row_i["nama_perusahaan"],
                    "alamat_1": row_i["alamat_perusahaan"],
                    "nama_2": row_j["nama_perusahaan"],
                    "alamat_2": row_j["alamat_perusahaan"],
                    "nama_similarity": round(score, 1),
                    "aksi": "DROP_DUPLICATE_SIJK"
                })

    sijk_clean = sijk_clean.drop(index=list(rows_to_drop))
    sijk_clean = sijk_clean.drop(
        columns=["_nama_norm", "_alamat_norm"],
        errors="ignore"
    )

    preview_df = pd.DataFrame(preview_rows)
    return (sijk_clean, preview_df) if preview else sijk_clean


# ==============================
# 2. MERGE SIJK KE SF
# ==============================
def merge_sijk_to_sf(sf, sijk, threshold=90, preview=False):
    """
    Merge SIJK ke SF.

    Kunci match = nama_perusahaan (fuzz.ratio >= threshold) + prov + kab sama persis.
    - Match → ENRICH kolom SF yang kosong (kolom protected tidak disentuh).
    - Tidak match → APPEND sebagai baris baru.
    """
    sf_copy   = sf.copy()
    sijk_copy = sijk.copy()
    preview_rows = []

    sf_copy["_nama_norm"]   = sf_copy["nama_perusahaan"].fillna("").str.upper().str.strip()
    sijk_copy["_nama_norm"] = sijk_copy["nama_perusahaan"].fillna("").str.upper().str.strip()

    for _, sijk_row in sijk_copy.iterrows():
        kdprov = sijk_row.get("prov", "")
        kdkab  = sijk_row.get("kab", "")

        # Filter kandidat: prov + kab HARUS sama persis
        candidates = sf_copy[
            (sf_copy["prov"] == kdprov) &
            (sf_copy["kab"] == kdkab)
        ]

        best_nama_score  = 0
        best_alamat_score = 0
        best_idx          = None

        for sf_idx, sf_row in candidates.iterrows():
            nama_s = _nama_score(sijk_row["_nama_norm"], sf_row["_nama_norm"])
            if nama_s < threshold:
                continue

            # Nama match → cek alamat
            alamat_s = _alamat_score(
                sijk_row.get("alamat_perusahaan", ""),
                sf_row.get("alamat_perusahaan", "")
            )

            # Pilih kandidat terbaik berdasarkan nama dulu, lalu alamat
            if nama_s > best_nama_score or (nama_s == best_nama_score and alamat_s > best_alamat_score):
                best_nama_score   = nama_s
                best_alamat_score = alamat_s
                best_idx          = sf_idx

        if best_idx is not None and best_alamat_score >= ALAMAT_THRESHOLD:
            # ENRICH: nama mirip + alamat mirip → sama perusahaan
            for col in sf_copy.columns:
                if col.startswith("_"):
                    continue
                if col in PROTECTED_COLS:
                    continue
                if col in sijk_copy.columns:
                    val_sf   = str(sf_copy.at[best_idx, col]).strip()
                    val_sijk = str(sijk_row.get(col, "")).strip()
                    if val_sf in ("", "nan") and val_sijk not in ("", "nan"):
                        sf_copy.at[best_idx, col] = val_sijk

            preview_rows.append({
                "kd_prov": kdprov,
                "kd_kab": kdkab,
                "nama_sijk": sijk_row["nama_perusahaan"],
                "nama_sf": sf_copy.at[best_idx, "nama_perusahaan"],
                "alamat_sijk": sijk_row["alamat_perusahaan"],
                "alamat_sf": sf_copy.at[best_idx, "alamat_perusahaan"],
                "nama_similarity": round(best_nama_score, 1),
                "alamat_similarity": round(best_alamat_score, 1),
                "aksi": "ENRICH_SF"
            })

        else:
            # APPEND: tidak ada match ATAU alamat terlalu berbeda → baris baru
            aksi_detail = "APPEND_NEW" if best_idx is None else "APPEND_BEDA_ALAMAT"
            sf_copy = pd.concat(
                [sf_copy, sijk_row.to_frame().T], ignore_index=True
            )
            preview_rows.append({
                "kd_prov": kdprov,
                "kd_kab": kdkab,
                "nama_sijk": sijk_row["nama_perusahaan"],
                "nama_sf": sf_copy.at[best_idx, "nama_perusahaan"] if best_idx is not None else "",
                "alamat_sijk": sijk_row["alamat_perusahaan"],
                "alamat_sf": sf_copy.at[best_idx, "alamat_perusahaan"] if best_idx is not None else "",
                "nama_similarity": round(best_nama_score, 1),
                "alamat_similarity": round(best_alamat_score, 1),
                "aksi": aksi_detail
            })

    sf_copy = sf_copy.drop(
        columns=["_nama_norm", "_alamat_norm"],
        errors="ignore"
    )

    preview_df = pd.DataFrame(preview_rows)
    return (sf_copy, preview_df) if preview else sf_copy


# ==============================
# 3. MERGE LPSE KE SF
# ==============================
def merge_lpse_to_sf(sf, lpse, threshold=90, preview=False):
    """
    Merge LPSE ke SF.

    Kunci match = nama_perusahaan (fuzz.ratio >= threshold) + prov + kab sama persis.
    - Match → ENRICH kolom SF yang kosong (kolom protected tidak disentuh).
    - Tidak match → APPEND sebagai baris baru.
    """
    sf_copy   = sf.copy()
    lpse_copy = lpse.copy()
    preview_rows = []

    sf_copy["_nama_norm"]   = sf_copy["nama_perusahaan"].fillna("").str.upper().str.strip()
    lpse_copy["_nama_norm"] = lpse_copy["nama_perusahaan"].fillna("").str.upper().str.strip()

    for _, lpse_row in lpse_copy.iterrows():
        kdprov = lpse_row.get("prov", "")
        kdkab  = lpse_row.get("kab", "")

        # Filter kandidat: prov + kab HARUS sama persis
        candidates = sf_copy[
            (sf_copy["prov"] == kdprov) &
            (sf_copy["kab"] == kdkab)
        ]

        best_nama_score   = 0
        best_alamat_score = 0
        best_idx          = None

        for sf_idx, sf_row in candidates.iterrows():
            nama_s = _nama_score(lpse_row["_nama_norm"], sf_row["_nama_norm"])
            if nama_s < threshold:
                continue

            # Nama match → cek alamat
            alamat_s = _alamat_score(
                lpse_row.get("alamat_perusahaan", ""),
                sf_row.get("alamat_perusahaan", "")
            )

            if nama_s > best_nama_score or (nama_s == best_nama_score and alamat_s > best_alamat_score):
                best_nama_score   = nama_s
                best_alamat_score = alamat_s
                best_idx          = sf_idx

        if best_idx is not None and best_alamat_score >= ALAMAT_THRESHOLD:
            # ENRICH: nama mirip + alamat mirip → sama perusahaan
            for col in sf_copy.columns:
                if col.startswith("_"):
                    continue
                if col in PROTECTED_COLS:
                    continue
                if col in lpse_copy.columns:
                    val_sf   = str(sf_copy.at[best_idx, col]).strip()
                    val_lpse = str(lpse_row.get(col, "")).strip()
                    if val_sf in ("", "nan") and val_lpse not in ("", "nan"):
                        sf_copy.at[best_idx, col] = val_lpse

            preview_rows.append({
                "kd_prov": kdprov,
                "kd_kab": kdkab,
                "nama_lpse": lpse_row["nama_perusahaan"],
                "nama_sf": sf_copy.at[best_idx, "nama_perusahaan"],
                "alamat_lpse": lpse_row["alamat_perusahaan"],
                "alamat_sf": sf_copy.at[best_idx, "alamat_perusahaan"],
                "nama_similarity": round(best_nama_score, 1),
                "alamat_similarity": round(best_alamat_score, 1),
                "aksi": "ENRICH_SF_FROM_LPSE"
            })

        else:
            # APPEND: tidak ada match ATAU alamat terlalu berbeda → baris baru
            aksi_detail = "APPEND_NEW_FROM_LPSE" if best_idx is None else "APPEND_BEDA_ALAMAT_LPSE"
            sf_copy = pd.concat(
                [sf_copy, lpse_row.to_frame().T],
                ignore_index=True
            )
            preview_rows.append({
                "kd_prov": kdprov,
                "kd_kab": kdkab,
                "nama_lpse": lpse_row["nama_perusahaan"],
                "nama_sf": sf_copy.at[best_idx, "nama_perusahaan"] if best_idx is not None else "",
                "alamat_lpse": lpse_row["alamat_perusahaan"],
                "alamat_sf": sf_copy.at[best_idx, "alamat_perusahaan"] if best_idx is not None else "",
                "nama_similarity": round(best_nama_score, 1),
                "alamat_similarity": round(best_alamat_score, 1),
                "aksi": aksi_detail
            })

    sf_copy = sf_copy.drop(
        columns=["_nama_norm", "_alamat_norm"],
        errors="ignore"
    )

    preview_df = pd.DataFrame(preview_rows)
    return (sf_copy, preview_df) if preview else sf_copy
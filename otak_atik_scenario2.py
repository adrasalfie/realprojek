import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz

########################
# PART 0 PREPROCESSING
########################

def deduplicate_by_key(df, key_col, label=None):
    df = df.copy()

    if key_col not in df.columns:
        raise ValueError(f"Column '{key_col}' not found in dataframe")

    before = len(df)

    # keep rows with real key only
    mask = df[key_col].notna()
    df_valid = df[mask].copy()
    df_invalid = df[~mask].copy()

    df_valid[key_col] = df_valid[key_col].astype(str).str.strip()

    df_valid = (
        df_valid
        .sort_values(by=[key_col])
        .drop_duplicates(subset=[key_col], keep="first")
    )

    df = pd.concat([df_valid, df_invalid], ignore_index=True)

    after = len(df)

    name = label if label is not None else key_col
    print(f"[DEDUP] {name} by '{key_col}': {before} -> {after}")

    return df.reset_index(drop=True)


# =====================================================
# PRE-CLEANING SIJK (Fuzzy Duplicate Removal)
# =====================================================
def preclean_sijk(sijk: pd.DataFrame, threshold: int = 75, preview: bool = False):
    sijk_clean = sijk.copy()
    preview_rows = []

    # FIX #3: cek kolom lebih robust — cari nama kolom yang tersedia
    # kolom nama bisa "nama_bu", kolom alamat bisa "alamat_bu"
    col_nama   = next((c for c in ["nama_bu", "nama_perusahaan"] if c in sijk_clean.columns), None)
    col_alamat = next((c for c in ["alamat_bu", "alamat_perusahaan"] if c in sijk_clean.columns), None)
    col_prov   = next((c for c in ["nmprov", "nm_prov"] if c in sijk_clean.columns), None)
    col_kab    = next((c for c in ["nmkab", "nm_kab"] if c in sijk_clean.columns), None)

    missing = [name for name, val in [
        ("nama (nama_bu/nama_perusahaan)", col_nama),
        ("alamat (alamat_bu/alamat_perusahaan)", col_alamat),
        ("provinsi (nmprov/nm_prov)", col_prov),
        ("kabupaten (nmkab/nm_kab)", col_kab),
    ] if val is None]

    if missing:
        for m in missing:
            print(f"[WARNING] Kolom '{m}' tidak ditemukan di SIJK — preclean dilewati")
        return (sijk_clean, pd.DataFrame()) if preview else sijk_clean

    sijk_clean["_nama_norm"]   = sijk_clean[col_nama].astype(str).str.upper().str.strip()
    sijk_clean["_alamat_norm"] = sijk_clean[col_alamat].astype(str).str.upper().str.strip()

    rows_to_drop = set()

    print(f"[PRE-CLEAN] SIJK: Memulai fuzzy duplicate removal (threshold={threshold})...")

    for (nmprov, nmkab), group in sijk_clean.groupby([col_prov, col_kab]):
        group_indices = list(group.index)

        for i_idx in range(len(group_indices)):
            i = group_indices[i_idx]
            if i in rows_to_drop:
                continue

            row_i  = sijk_clean.loc[i]
            text_i = f"{row_i['_nama_norm']} {row_i['_alamat_norm']}"

            for j_idx in range(i_idx + 1, len(group_indices)):
                j = group_indices[j_idx]
                if j in rows_to_drop:
                    continue

                row_j  = sijk_clean.loc[j]
                text_j = f"{row_j['_nama_norm']} {row_j['_alamat_norm']}"

                similarity = fuzz.token_set_ratio(text_i, text_j)

                if similarity >= threshold:
                    rows_to_drop.add(j)

                    preview_rows.append({
                        "nama_1":     row_i[col_nama],
                        "alamat_1":   row_i[col_alamat],
                        "nama_2":     row_j[col_nama],
                        "alamat_2":   row_j[col_alamat],
                        "similarity": similarity,
                        "aksi":       "DROP"
                    })

    if rows_to_drop:
        sijk_clean = sijk_clean.drop(index=list(rows_to_drop))

    sijk_clean = sijk_clean.drop(
        columns=["_nama_norm", "_alamat_norm"],
        errors="ignore"
    )

    print(f"[PRE-CLEAN] SIJK: {len(rows_to_drop)} duplikat fuzzy dihapus")

    preview_df = pd.DataFrame(preview_rows)

    if preview:
        return sijk_clean.reset_index(drop=True), preview_df
    else:
        return sijk_clean.reset_index(drop=True)


########################
# PART 1 HELPER FUNCTION
########################

def load_excel(path_excel):
    df = pd.read_excel(path_excel, dtype=str)
    return df

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def normalize_address(addr):
    """
    Normalisasi alamat: hapus prefix (JL, KEL, KEC, DESA, dll) dan angka,
    tapi TETAP SIMPAN nama jalan/kelurahan/kecamatan setelah prefix tersebut.
    """
    if pd.isna(addr) or str(addr).strip() == "":
        return ""

    addr = str(addr).upper().strip()

    # Hapus RT/RW beserta angkanya
    addr = re.sub(r'\bRT\.?\s*\d+', '', addr)
    addr = re.sub(r'\bRW\.?\s*\d+', '', addr)

    # Hapus HANYA kata prefix sebagai kata berdiri sendiri,
    # BUKAN beserta nama setelahnya — agar nama kel/kec tetap bisa dibandingkan
    prefix_words = [
        'KELURAHAN', 'KECAMATAN', 'KEL.', 'KEC.',
        'KEL', 'KEC', 'DESA', 'DS.', 'DS', 'KOTA',
        'JALAN', 'JLN.', 'JLN', 'JL.', 'JL',
        'NO.', 'NOMOR', 'NO',
    ]
    # Urutkan dari terpanjang agar tidak partial replace (misal KEL sebelum KELURAHAN)
    for word in sorted(prefix_words, key=len, reverse=True):
        addr = re.sub(r'\b' + re.escape(word) + r'\b', ' ', addr)

    # Hapus angka (nomor rumah, kode pos, dll)
    addr = re.sub(r'\d+', '', addr)

    # Hapus karakter khusus
    addr = re.sub(r'[^\w\s]', ' ', addr)

    # Hapus spasi berlebih
    addr = ' '.join(addr.split())

    return addr.strip()

def split_kdkab(kdkab):
    s = safe_str(kdkab)
    if len(s) < 4:
        return "", ""
    return s[:2], s[2:4]

def make_sf_kdkab(df_sf):
    df_sf = df_sf.copy()
    df_sf["kdkab_sf"] = (
        df_sf["prov"].astype(str).str.strip().str.zfill(2) +
        df_sf["kab"].astype(str).str.strip().str.zfill(2)
    )
    return df_sf

# FIX #2: fungsi untuk membuat kolom kdkab di SIJK dan LPSE
def make_source_kdkab(df, col_prov, col_kab, result_col="kdkab"):
    """
    Gabungkan kolom kode provinsi dan kode kabupaten menjadi satu kolom kdkab (4 digit).
    """
    df = df.copy()
    if col_prov not in df.columns or col_kab not in df.columns:
        raise ValueError(
            f"[ERROR] Kolom '{col_prov}' atau '{col_kab}' tidak ditemukan. "
            f"Kolom tersedia: {list(df.columns)}"
        )
    df[result_col] = (
        df[col_prov].astype(str).str.strip().str.zfill(2) +
        df[col_kab].astype(str).str.strip().str.zfill(2)
    )
    return df

def calc_similarity(a, b):
    return fuzz.token_sort_ratio(safe_str(a), safe_str(b))

def calc_char_length(x):
    return len(safe_str(x))

def weighted_average_similarity(sim, length):
    total = sum(length.values())
    if total == 0:
        return 0.0

    score = 0
    for k in sim:
        score += sim[k] * length[k]

    return score / total

# Kata badan usaha yang dihapus sebelum membandingkan nama perusahaan.
_BADAN_USAHA_TOKENS = {
    'cv', 'pt', 'ud', 'tb', 'fa', 'firma',
    'koperasi', 'persero', 'tbk', 'perseroda',
}

def strip_badan_usaha(name: str) -> str:
    """Hapus token badan usaha dari nama sebelum dibandingkan."""
    import re as _re
    tokens = _re.sub(r'[^\w\s]', ' ', name.lower()).split()
    tokens = [t for t in tokens if t not in _BADAN_USAHA_TOKENS]
    return " ".join(tokens)

def name_similarity(a: str, b: str) -> float:
    """
    Similarity nama berbasis TOKEN EXACT (Jaccard).
    Contoh: "MANDIRI DUA" vs "DUTA MANDIRI"
      → set_a={mandiri,dua} set_b={duta,mandiri}
      → intersection={mandiri} → score = 2*1/(2+2)*100 = 50 ✅
    """
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a and not set_b:
        return 100.0
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    return round(2 * len(intersection) / (len(set_a) + len(set_b)) * 100, 4)



########################
# PART 2 COMPARISON
########################

def compare_one_to_one(
    source_row,
    sf_row,
    source_name_col,
    source_kdkab_col,
    source_addr_col
):
    src_name = safe_str(source_row[source_name_col])
    src_kab  = safe_str(source_row[source_kdkab_col])
    src_addr = safe_str(source_row[source_addr_col])

    sf_name  = safe_str(sf_row["nama_perusahaan"])
    sf_kab   = safe_str(sf_row["kdkab_sf"])
    # Gabungkan alamat_perusahaan + nm_kec + nm_kel dari SF menjadi satu string.
    # Ini menangani kasus alamat SF hanya berisi nama jalan, sedangkan info
    # kecamatan/kelurahan tersimpan di kolom terpisah — sementara sumber data
    # (LPSE/SIJK) menggabungkan semua info itu dalam satu kolom alamat.
    sf_addr_parts = [
        safe_str(sf_row.get("alamat_perusahaan", "")),
        safe_str(sf_row.get("nm_kec", "")),
        safe_str(sf_row.get("nm_kel", "")),
    ]
    sf_addr = " ".join(p for p in sf_addr_parts if p).strip()

    sim    = {}
    length = {}

    # name — strip badan usaha lalu bandingkan dengan token exact (Jaccard)
    # token_sort/set_ratio berbasis Levenshtein karakter tidak bisa membedakan
    # "MANDIRI DUA" vs "DUTA MANDIRI" karena beda hanya 2 karakter dari 23
    src_name_stripped = strip_badan_usaha(src_name)
    sf_name_stripped  = strip_badan_usaha(sf_name)
    sim["name"]    = name_similarity(src_name_stripped, sf_name_stripped)
    length["name"] = max(len(src_name_stripped), 1)

    # kab (STRICT)
    sim["kab"]    = 100 if src_kab == sf_kab and src_kab != "" else 0
    length["kab"] = len(src_kab)

    # address — hanya dihitung kalau KEDUA sisi ada isinya
    # sf_addr sudah menggabungkan alamat + kecamatan + kelurahan dari SF
    # sf_addr_empty dihitung dari hasil GABUNGAN (bukan hanya alamat_perusahaan)
    sf_addr_empty  = (sf_addr == "")
    src_addr_empty = (src_addr == "")

    if not src_addr_empty and not sf_addr_empty:
        src_addr_norm  = normalize_address(src_addr)
        sf_addr_norm   = normalize_address(sf_addr)
        sim["addr"]    = fuzz.token_set_ratio(src_addr_norm, sf_addr_norm)
        length["addr"] = len(src_addr_norm)

    wavg = weighted_average_similarity(sim, length)

    # Return sf_addr (string gabungan) agar bisa ditampilkan di monitoring
    return sim, length, wavg, sf_addr_empty, src_addr_empty, sf_addr



########################
# PART 3 DECISION RULE
########################

def decision_rule(sim, wavg, sf_addr_empty=False, src_addr_empty=False):
    sim_name = sim.get("name", 0)
    sim_kab  = sim.get("kab", 0)
    sim_addr = sim.get("addr", None)

    if sim_kab == 100 and sim_name >= 90:
        if sf_addr_empty and not src_addr_empty and wavg < 90:
            return "INSERT"
        return "DROP"

    # ── strong weighted match ──────────────────────────────────────────────────
    if wavg >= 90:
        return "DROP"

    # ── borderline weighted match ──────────────────────────────────────────────
    if 80 <= wavg <= 89:

        # kab beda → beda wilayah
        if sim_kab < 100:
            return "INSERT"

        # kab sama, SF tidak punya alamat tapi source punya → tidak bisa dikonfirmasi
        if sf_addr_empty and not src_addr_empty:
            return "INSERT"

        # kab sama, nama cukup mirip (80–89) → konfirmasi lewat alamat
        if 80 <= sim_name <= 89:
            if sim_addr is not None and sim_addr >= 90:
                return "DROP"
            return "INSERT"

        return "INSERT"

    # ── weak weighted match ────────────────────────────────────────────────────
    return "INSERT"


########################
# PART 4 MATCH TO SF
########################

def find_best_sf_match(
    source_row,
    df_sf,
    source_name_col,
    source_kdkab_col,
    source_addr_col
):
    best_score  = -1
    best_sim    = None
    best_sf_idx = None

    src_name = safe_str(source_row[source_name_col])
    src_kab  = safe_str(source_row[source_kdkab_col])

    # PRIORITAS 1: EXACT MATCH nama (case-insensitive, strip badan usaha) + kab sama
    src_name_stripped = strip_badan_usaha(src_name)

    exact_match_mask = (
        df_sf["nama_perusahaan"].apply(
            lambda n: strip_badan_usaha(safe_str(n)) == src_name_stripped
        ) &
        (df_sf["kdkab_sf"] == src_kab)
    )
    exact_matches = df_sf[exact_match_mask]

    if not exact_matches.empty:
        exact_idx = exact_matches.index[0]
        exact_row = df_sf.loc[exact_idx]
        sim, _, wavg, sf_addr_empty, src_addr_empty, sf_addr_str = compare_one_to_one(
            source_row, exact_row,
            source_name_col, source_kdkab_col, source_addr_col
        )
        return exact_idx, sim, wavg, sf_addr_empty, src_addr_empty, sf_addr_str

    # PRIORITAS 2: FUZZY MATCH — hanya dalam kab yang sama
    same_kab = df_sf[df_sf["kdkab_sf"] == src_kab]

    if same_kab.empty:
        # Tidak ada perusahaan di kab ini di SF → langsung INSERT
        return None, {}, 0.0, False, False, ""

    best_sf_addr_empty  = False
    best_src_addr_empty = False
    best_sf_addr_str    = ""

    for idx, sf_row in same_kab.iterrows():
        sim, _, wavg, sf_addr_empty, src_addr_empty, sf_addr_str = compare_one_to_one(
            source_row, sf_row,
            source_name_col, source_kdkab_col, source_addr_col
        )
        if wavg > best_score:
            best_score          = wavg
            best_sim            = sim
            best_sf_idx         = idx
            best_sf_addr_empty  = sf_addr_empty
            best_src_addr_empty = src_addr_empty
            best_sf_addr_str    = sf_addr_str

    return best_sf_idx, best_sim, best_score, best_sf_addr_empty, best_src_addr_empty, best_sf_addr_str


########################
# PART 5 PROCESS MATCH
########################

def process_source_table_with_monitoring(
    df_source,
    df_sf,
    source_name_col,
    source_kdkab_col,
    source_addr_col,
    source_label,
    additional_columns_mapping=None,
    auto_fill_columns=None
):

    inserted_rows  = []
    monitoring_rows = []

    for src_idx, row in df_source.iterrows():

        sf_idx, sim, best_wavg, sf_addr_empty, src_addr_empty, sf_addr_str = find_best_sf_match(
            row,
            df_sf,
            source_name_col,
            source_kdkab_col,
            source_addr_col
        )

        # Tidak ada kandidat di kab yang sama → langsung INSERT
        if sf_idx is None:
            sim            = {}
            best_wavg      = 0.0
            sf_addr_empty  = False
            src_addr_empty = False
            sf_addr_str    = ""
            decision       = "INSERT"
        else:
            decision = decision_rule(
                sim, best_wavg,
                sf_addr_empty=sf_addr_empty,
                src_addr_empty=src_addr_empty
            )

        # ----------------------------
        # monitoring record
        # ----------------------------
        mon = {
            "source":              source_label,
            "source_row_index":    src_idx,
            "source_name":         row[source_name_col],
            "source_kdkab":        row[source_kdkab_col],
            "source_address":      row[source_addr_col],
            "matched_sf_index":    sf_idx,
            "sf_name":             df_sf.loc[sf_idx, "nama_perusahaan"] if sf_idx is not None else None,
            "sf_kdkab":            df_sf.loc[sf_idx, "kdkab_sf"]        if sf_idx is not None else None,
            "sf_address":          sf_addr_str,   # isi alamat SF gabungan (alamat+kec+kel)
            "sim_name":            sim.get("name", None),
            "sim_kab":             sim.get("kab", None),
            "sim_address":         sim.get("addr", None),
            "weighted_similarity": best_wavg,
            "decision":            decision
        }

        monitoring_rows.append(mon)

        if decision == "INSERT":

            prov, kab = split_kdkab(row[source_kdkab_col])

            new_row = {
                "nama_perusahaan":   row[source_name_col],
                "alamat_perusahaan": row[source_addr_col],
                "prov":              prov,
                "kab":               kab,
                "source":            source_label
            }

            if additional_columns_mapping:
                for source_col, target_col in additional_columns_mapping.items():
                    new_row[target_col] = row.get(source_col, "")

            if auto_fill_columns:
                for col_name, col_value in auto_fill_columns.items():
                    new_row[col_name] = col_value

            inserted_rows.append(new_row)

    return pd.DataFrame(inserted_rows), pd.DataFrame(monitoring_rows)


########################
# PART 5b KONVERSI MAPPING
# FIX #1: pindahkan fungsi konversi ke luar run_second_scenario_pipeline_and_save
#         agar tidak nested dan bisa dipakai ulang
########################

def apply_mapping_konversi(df, col_name, map_dict):
    if col_name in df.columns:
        df = df.copy()
        df[col_name] = (
            df[col_name]
            .astype(str)
            .str.lower()
            .map(map_dict)
            .fillna(df[col_name])
        )
    return df


# FIX #1: apply_contains_mapping_badan_usaha dipindah keluar (tidak lagi nested)
def apply_contains_mapping_badan_usaha(df, col_name):
    if col_name not in df.columns:
        return df

    map_badan_usaha_rules = [
        {"keywords": ["persero"],                               "value": "1"},
        {"keywords": ["pt"],                                    "value": "2"},
        {"keywords": ["cv", "commanditer", "persekutuan komanditer"], "value": "3"},
        {"keywords": ["koperasi"],                              "value": "4"},
        {"keywords": ["kantor perwakilan bujka"],               "value": "5"},
    ]

    def find_match(value):
        if pd.isna(value) or str(value).strip() == "" or str(value) == "nan":
            return "9"

        value_clean = str(value).lower().strip()

        for rule in map_badan_usaha_rules:
            for kw in rule["keywords"]:
                if kw in value_clean:
                    return rule["value"]

        return "9"

    df = df.copy()
    df[col_name] = df[col_name].apply(find_match)
    return df


########################
# PART 6 MAIN RUNNER
########################

def run_second_scenario_pipeline_and_save(
    path_sf,
    path_sijk,
    path_lpse,
    output_sf_path,
    output_monitoring_path,
):

    from datetime import datetime

    # ---- load
    df_sf   = load_excel(path_sf)
    df_sijk = load_excel(path_sijk)
    df_lpse = load_excel(path_lpse)

    # ---- prepare SF: buat kolom kdkab_sf
    df_sf = make_sf_kdkab(df_sf)

    # ---- PRE-CLEANING SIJK (Fuzzy Duplicate Removal)
    df_sijk, preview_sijk_df = preclean_sijk(df_sijk, threshold=75, preview=True)

    # ---- dedup LPSE by kd_penyedia
    df_lpse = deduplicate_by_key(df_lpse, "kd_penyedia", "LPSE")

    # FIX #2: buat kolom kdkab untuk SIJK dan LPSE
    # Sesuaikan nama kolom prov/kab jika berbeda di file Anda
    df_sijk = make_source_kdkab(df_sijk, col_prov="kdprov", col_kab="kd_kab")
    df_lpse = make_source_kdkab(df_lpse, col_prov="kdprov", col_kab="kd_kab")

    TAHUN_REVISI = str(datetime.now().year)

    # ---- additional mapping for SIJK
    mapping_sijk = {
        "npwp_bu":         "npwp",
        "bentuk_usaha_bu": "badan_usaha",
        "nmprov":          "nm_prov",
        "nmkab":           "nm_kab",
        "kualifikasi_final": "kualifikasi",
        "skala_usaha":     "skala_usaha",
        "telepon_bu":      "no_telp",
        "email_bu":        "email",
        "sub_klasifikasi": "pekerjaan_utama",
        "nib":             "nib"
    }

    # ---- additional mapping for LPSE
    mapping_lpse = {
        "npwp_penyedia":   "npwp",
        "telepon":         "no_telp",
        "fax":             "fax",
        "website":         "website",
        "bentuk_usaha":    "badan_usaha",
        "kualifikasi_usaha": "kualifikasi",
        "nmprov":          "nm_prov",
        "nmkab":           "nm_kab",
        "kd_klasifikasi":  "kbli",
        "kd_penyedia":     "kd_penyedia",
        "nomor_izin_usaha": "nib"
    }

    # ---- auto fill columns
    auto_fill_sijk = {
        "tahun_revisi": TAHUN_REVISI,
        "kategori":     "F",
        "sumber_data":  "2"
    }

    auto_fill_lpse = {
        "tahun_revisi": TAHUN_REVISI,
        "kategori":     "F",
        "sumber_data":  "1"
    }

    insert_sijk, monitor_sijk = process_source_table_with_monitoring(
        df_source                 = df_sijk,
        df_sf                     = df_sf,
        source_name_col           = "nama_bu",
        source_kdkab_col          = "kdkab",
        source_addr_col           = "alamat_bu",
        source_label              = "sijk",
        additional_columns_mapping = mapping_sijk,
        auto_fill_columns         = auto_fill_sijk
    )

    insert_lpse, monitor_lpse = process_source_table_with_monitoring(
        df_source                 = df_lpse,
        df_sf                     = df_sf,
        source_name_col           = "nama_penyedia",
        source_kdkab_col          = "kdkab",
        source_addr_col           = "alamat",
        source_label              = "lpse",
        additional_columns_mapping = mapping_lpse,
        auto_fill_columns         = auto_fill_lpse
    )

    # ---- combine insert
    df_insert_all = pd.concat([insert_sijk, insert_lpse], ignore_index=True)
    df_monitoring = pd.concat([monitor_sijk, monitor_lpse], ignore_index=True)

    # FIX #6: dedup antar hasil INSERT SIJK vs LPSE
    # Prioritas: SIJK (sumber_data=2) lebih dipercaya dari LPSE (sumber_data=1)
    if not df_insert_all.empty and "nama_perusahaan" in df_insert_all.columns:
        priority_map = {"2": 1, "1": 2}
        df_insert_all["_priority"] = (
            df_insert_all["sumber_data"]
            .map(priority_map)
            .fillna(99)
        )
        df_insert_all = df_insert_all.sort_values("_priority")
        df_insert_all = df_insert_all.drop_duplicates(
            subset=["nama_perusahaan", "prov", "kab"],
            keep="first"
        )
        df_insert_all = df_insert_all.drop(columns=["_priority"])
        print(f"[DEDUP] insert_all setelah dedup SIJK vs LPSE: {len(df_insert_all)} baris")

    # ---- konversi mapping
    map_kualifikasi = {
        "spesialis":              "1",
        "kecil":                  "2",
        "menengah":               "3",
        "besar":                  "4",
        "tidak memiliki sbu aktif": "9",
    }
    df_insert_all = apply_mapping_konversi(df_insert_all, "kualifikasi", map_kualifikasi)

    map_skala = {
        "kecil":   "2",
        "menengah": "3",
        "besar":   "4",
    }
    df_insert_all = apply_mapping_konversi(df_insert_all, "skala_usaha", map_skala)
    df_insert_all = apply_contains_mapping_badan_usaha(df_insert_all, "badan_usaha")

    # ---- Final SF
    # FIX #7: hapus kolom bantu kdkab_sf sebelum disimpan
    df_sf_for_output = df_sf.drop(columns=["kdkab_sf"], errors="ignore")
    df_sf_final = pd.concat([df_sf_for_output, df_insert_all], ignore_index=True)

    # ---- save
    with pd.ExcelWriter(output_sf_path, engine='openpyxl') as writer:
        df_sf_final.to_excel(writer, sheet_name='SF_Final', index=False)

        if not preview_sijk_df.empty:
            preview_sijk_df.to_excel(writer, sheet_name='Preview_SIJK_Duplikat', index=False)
            print(f"[PREVIEW] {len(preview_sijk_df)} data duplikat SIJK → worksheet 'Preview_SIJK_Duplikat'")
        else:
            print("[PREVIEW] Tidak ada duplikat fuzzy yang ditemukan di SIJK")

    df_monitoring.to_excel(output_monitoring_path, index=False)

    print(f"[DONE] SF Final: {len(df_sf_final)} baris | Monitoring: {len(df_monitoring)} baris")

    return df_sf_final, df_monitoring


########################
# PART 7 TRIAL
########################
if __name__ == "__main__":

    PATH_SF   = "7500 sf.xlsx"
    PATH_SIJK = "7500 sijk.xlsx"
    PATH_LPSE = "7500 lpse.xlsx"

    OUT_SF      = "sf_after_scenario2.xlsx"
    OUT_MONITOR = "monitoring_weighted_similarity.xlsx"

    run_second_scenario_pipeline_and_save(
        PATH_SF,
        PATH_SIJK,
        PATH_LPSE,
        OUT_SF,
        OUT_MONITOR
    )

    print("Scenario-2 matching finished.")
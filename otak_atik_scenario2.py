import pandas as pd
import numpy as np
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
def preclean_sijk(sijk: pd.DataFrame, threshold: int = 75):
    """
    Hapus duplikat SIJK berdasarkan fuzzy matching Nama + Alamat.
    Tidak menggunakan NIB, tapi membandingkan kemiripan data.
    """
    sijk_clean = sijk.copy()
    
    # Cek apakah kolom yang dibutuhkan ada
    required_cols = ["nama_bu", "alamat_bu", "nmprov", "nmkab"]
    for col in required_cols:
        if col not in sijk_clean.columns:
            print(f"[WARNING] Kolom '{col}' tidak ditemukan di SIJK")
            return sijk_clean
    
    # Normalisasi kolom untuk perbandingan
    sijk_clean["_nama_norm"] = sijk_clean["nama_bu"].str.upper().str.strip()
    sijk_clean["_alamat_norm"] = sijk_clean["alamat_bu"].str.upper().str.strip()
    
    rows_to_drop = set()
    total_dropped = 0
    
    print(f"[PRE-CLEAN] SIJK: Memulai fuzzy duplicate removal...")
    
    # Group berdasarkan nmprov dan nmkab
    for (nmprov, nmkab), group in sijk_clean.groupby(["nmprov", "nmkab"]):
        group_indices = list(group.index)
        
        for i_idx in range(len(group_indices)):
            i = group_indices[i_idx]
            if i in rows_to_drop:
                continue
            
            row_i = sijk_clean.loc[i]
            text_i = f"{row_i['_nama_norm']} {row_i['_alamat_norm']}"
            
            for j_idx in range(i_idx + 1, len(group_indices)):
                j = group_indices[j_idx]
                if j in rows_to_drop:
                    continue
                
                row_j = sijk_clean.loc[j]
                text_j = f"{row_j['_nama_norm']} {row_j['_alamat_norm']}"
                
                # Hitung similarity menggunakan token_set_ratio
                similarity = fuzz.token_set_ratio(text_i, text_j)
                
                if similarity >= threshold:
                    rows_to_drop.add(j)
    
    # Hapus baris yang teridentifikasi sebagai duplikat
    if rows_to_drop:
        sijk_clean = sijk_clean.drop(index=list(rows_to_drop))
        total_dropped = len(rows_to_drop)
    
    # Hapus kolom temporary
    sijk_clean = sijk_clean.drop(
        columns=["_nama_norm", "_alamat_norm"],
        errors="ignore"
    )
    
    print(f"[PRE-CLEAN] SIJK: {total_dropped} duplikat fuzzy dihapus")
    
    return sijk_clean.reset_index(drop=True)

########################
# PART 1 HELPER FUNCTION
########################

def load_excel(path_excel):
    df = pd.read_excel(path_excel, dtype=str)
    return df

def safe_str(x):
    """
    To ensure safe string so empty text is ""
    """
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def split_kdkab(kdkab):
    """
    To split kdkab 4 digits as 2 digits of prov and 2 digits of kab
    """
    s = safe_str(kdkab)
    if len(s) < 4:
        return "", ""
    return s[:2], s[2:4]

def make_sf_kdkab(df_sf):
    df_sf = df_sf.copy()
    df_sf["kdkab_sf"] = (df_sf["prov"].astype(str).str.zfill(2) + df_sf["kab"].astype(str).str.zfill(2))
    return df_sf

def calc_similarity(a, b):
    return fuzz.token_sort_ratio(safe_str(a), safe_str(b))

def calc_char_length(x):
    return(len(safe_str(x)))

def weighted_average_similarity(sim, length):
    total = sum(length.values())
    if total == 0:
        return 0.0
    
    score = 0
    for k in sim:
        score += sim[k] * length[k]

    return score / total

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
    sf_addr  = safe_str(sf_row["alamat_perusahaan"])

    sim = {}
    length = {}

    # name
    sim["name"] = fuzz.token_sort_ratio(src_name, sf_name)
    length["name"] = len(src_name)

    # kab (STRICT)
    sim["kab"] = 100 if src_kab == sf_kab and src_kab != "" else 0
    length["kab"] = len(src_kab)

    # address only if both exist
    if src_addr != "" and sf_addr != "":
        sim["addr"] = fuzz.token_sort_ratio(src_addr, sf_addr)
        length["addr"] = len(src_addr)

    wavg = weighted_average_similarity(sim, length)

    return sim, length, wavg


########################
# PART 3 DECISION RULE
########################

def decision_rule(sim, wavg):

    sim_name = sim.get("name", 0)
    sim_kab  = sim.get("kab", 0)
    sim_addr = sim.get("addr", None)

    # ------------------------------------------------
    # strong weighted match
    # ------------------------------------------------
    if wavg >= 90:
        return "DROP"

    # ------------------------------------------------
    # borderline weighted match
    # ------------------------------------------------
    if 80 <= wavg <= 89:

        # a) kdkab = 100 and name 90–100
        if sim_kab == 100 and 90 <= sim_name <= 100:
            return "DROP"

        # b) kdkab = 100 and name 80–89 → check address
        if sim_kab == 100 and 80 <= sim_name <= 89:

            # address exists and supports
            if sim_addr is not None and 90 <= sim_addr <= 100:
                return "DROP"

            # otherwise
            return "INSERT"

        # c) name very high but different kab
        if sim_kab < 100 and 90 <= sim_name <= 100:
            return "INSERT"

        return "INSERT"

    # ------------------------------------------------
    # weak weighted match
    # ------------------------------------------------
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

    best_score = -1
    best_sim = None
    best_sf_idx = None

    src_kab = safe_str(source_row[source_kdkab_col])

    same_kab = df_sf[df_sf["kdkab_sf"] == src_kab]

    candidate_sf = same_kab if len(same_kab) > 0 else df_sf

    for idx, sf_row in candidate_sf.iterrows():

        sim, length, wavg = compare_one_to_one(
            source_row,
            sf_row,
            source_name_col,
            source_kdkab_col,
            source_addr_col
        )

        if wavg > best_score:
            best_score = wavg
            best_sim = sim
            best_sf_idx = idx

    return best_sf_idx, best_sim, best_score

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
    additional_columns_mapping= None,
    auto_fill_columns=None
):

    inserted_rows = []
    monitoring_rows = []

    for src_idx, row in df_source.iterrows():

        sf_idx, sim, best_wavg = find_best_sf_match(
            row,
            df_sf,
            source_name_col,
            source_kdkab_col,
            source_addr_col
        )

        decision = decision_rule(sim, best_wavg)

        # ----------------------------
        # monitoring record
        # ----------------------------
        mon = {
            "source": source_label,
            "source_row_index": src_idx,

            "source_name": row[source_name_col],
            "source_kdkab": row[source_kdkab_col],
            "source_address": row[source_addr_col],

            "matched_sf_index": sf_idx,
            "sf_name": df_sf.loc[sf_idx, "nama_perusahaan"] if sf_idx is not None else None,
            "sf_kdkab": df_sf.loc[sf_idx, "kdkab_sf"] if sf_idx is not None else None,

            "sim_name": sim.get("name", None),
            "sim_kab": sim.get("kab", None),
            "sim_address": sim.get("addr", None),

            "weighted_similarity": best_wavg,
            "decision": decision
        }

        monitoring_rows.append(mon)

        # ----------------------------
        # insertion
        # ----------------------------
        if decision == "INSERT":

            prov, kab = split_kdkab(row[source_kdkab_col])

            new_row = {
                "nama_perusahaan": row[source_name_col],
                "alamat_perusahaan": row[source_addr_col],
                "prov": prov,
                "kab": kab,
                "source": source_label
            }

            # Menyalin kolom-kolom tambahan berdasarkan mapping
            if additional_columns_mapping:
                for source_col, target_col in additional_columns_mapping.items():
                    new_row[target_col] = row.get(source_col, "")

            # Mengisi kolom otomatis
            if auto_fill_columns:
                for col_name, col_value in auto_fill_columns.items():
                    new_row[col_name] = col_value

            inserted_rows.append(new_row)

    return pd.DataFrame(inserted_rows), pd.DataFrame(monitoring_rows)


########################
# PART 6 MAIN RUNNER
########################

def run_second_scenario_pipeline_and_save(
    path_sf,
    path_sijk,
    path_lpse,
    output_sf_path,
    output_monitoring_path
):

    # ---- load again from excel
    from datetime import datetime
    df_sf   = load_excel(path_sf)
    df_sijk = load_excel(path_sijk)
    df_lpse = load_excel(path_lpse)

    # ---- prepare sf
    df_sf = make_sf_kdkab(df_sf)

    # =====================================================
    # PRE-CLEANING SIJK (Fuzzy Duplicate Removal)
    # Hapus duplikat berdasarkan Nama + Alamat (BUKAN NIB)
    # =====================================================
    df_sijk = preclean_sijk(df_sijk, threshold=75)

    # ---- deduplicate sijk and lpse
    # df_sijk = deduplicate_by_key(df_sijk, "nib", "SIJK")
    df_lpse = deduplicate_by_key(df_lpse, "kd_penyedia", "LPSE")

    #---- Year of revision for monitoring
    TAHUN_REVISI = str(datetime.now().year)

     # ---- additional mapping for SIJK
    mapping_sijk = {
        "npwp_bu": "npwp",
        "bentuk_usaha_bu": "badan_usaha",
        "nmprov": "nm_prov",
        "nmkab": "nm_kab",
        "kualifikasi_final": "kualifikasi",
        "skala_usaha": "skala_usaha",
        "telepon_bu": "no_telp",      
        "email_bu": "email",          
        "sub_klasifikasi": "pekerjaan_utama"
    }

    # ---- additional mapping for LPSE
    mapping_lpse = {
        "npwp_penyedia": "npwp",
        "telepon": "no_telp",
        "fax": "fax",
        "website": "website",
        "bentuk_usaha": "badan_usaha",
        "kualifikasi_usaha": "kualifikasi",
        "nmprov": "nm_prov",
        "nmkab": "nm_kab",
        "kd_klasifikasi": "kbli",
        "kd_penyedia": "kd_penyedia",
    }


    #--- auto fill columns for sijk
    auto_fill_sijk = {
        "tahun_revisi": TAHUN_REVISI,
        "kategori": "F",
        "sumber_data": "2"
    }

    #--- auto fill columns for lpse
    auto_fill_lpse = {
        "tahun_revisi": TAHUN_REVISI,
        "kategori": "F",
        "sumber_data": "1"
    }

    # ---- SIJK
    insert_sijk, monitor_sijk = process_source_table_with_monitoring(
        df_source = df_sijk,
        df_sf = df_sf,
        source_name_col = "nama_bu",
        source_kdkab_col = "kdkab",
        source_addr_col = "alamat_bu",
        source_label = "sijk",
        additional_columns_mapping = mapping_sijk,
        auto_fill_columns = auto_fill_sijk
    )

    # ---- LPSE
    insert_lpse, monitor_lpse = process_source_table_with_monitoring(
        df_source = df_lpse,
        df_sf = df_sf,
        source_name_col = "nama_penyedia",
        source_kdkab_col = "kdkab",
        source_addr_col = "alamat",
        source_label = "lpse",
        additional_columns_mapping = mapping_lpse,
        auto_fill_columns = auto_fill_lpse
    )

    # ---- combine
    df_insert_all = pd.concat([insert_sijk, insert_lpse], ignore_index=True)
    df_monitoring = pd.concat([monitor_sijk, monitor_lpse], ignore_index=True)
    
    # Fungsi helper untuk melakukan mapping konversi
    def apply_mapping_konversi(df, col_name, map_dict):
        if col_name in df.columns:
            df[col_name] = (
                df[col_name]
                .astype(str)
                .str.lower()
                .map(map_dict)
                .fillna(df[col_name])
            )
        return df

#Mapping BADAN USAHA
    def apply_contains_mapping_badan_usaha(df, col_name):
        if col_name not in df.columns:
            return df

        map_badan_usaha_rules = [
            {"keywords": ["persero"], "value": "1"},
            {"keywords": ["pt"], "value": "2"},
            {"keywords": ["cv", "commanditer", "persekutuan komanditer"], "value": "3"},
            {"keywords": ["koperasi"], "value": "4"},
            {"keywords": ["kantor perwakilan bujka"], "value": "5"},
        ]
        
        def find_match(value):
            if pd.isna(value) or str(value).strip() == "" or str(value) == "nan":
                return "9" # Default jika kosong
            
            value_clean = str(value).lower().strip()
            
            for rule in map_badan_usaha_rules:
                keywords = rule["keywords"]
                target_value = rule["value"]
                
                for kw in keywords:
                    if kw in value_clean: 
                        return target_value
            
            return "9" # Default kalau tidak match
        
        df[col_name] = df[col_name].apply(find_match)
        return df

    # Mapping KUALIFIKASI
    map_kualifikasi = {
        "spesialis": "1",
        "kecil": "2",
        "menengah": "3",
        "besar": "4",
        "tidak memiliki sbu aktif": "9",
    }
    df_insert_all = apply_mapping_konversi(df_insert_all, "kualifikasi", map_kualifikasi)

    # Mapping SKALA USAHA (SIJK)
    map_skala = {
        "kecil": "2",
        "menengah": "3",
        "besar": "4",
    }
    df_insert_all = apply_mapping_konversi(df_insert_all, "skala_usaha", map_skala)

    # Mapping BADAN USAHA (LPSE + SIJK)
    df_insert_all = apply_contains_mapping_badan_usaha(df_insert_all, "badan_usaha")


    # ---- final sf
    df_sf_final = pd.concat([df_sf, df_insert_all], ignore_index=True)

    # ---- save
    df_sf_final.to_excel(output_sf_path, index=False)
    df_monitoring.to_excel(output_monitoring_path, index=False)

    return df_sf_final, df_monitoring

########################
# PART 7 TRIAL
########################
if __name__ == "__main__":

    PATH_SF   = "7500 sf.xlsx"
    PATH_SIJK = "7500 sijk.xlsx"
    PATH_LPSE = "7500 lpse.xlsx"

    OUT_SF       = "sf_after_scenario2.xlsx"
    OUT_MONITOR  = "monitoring_weighted_similarity.xlsx"

    run_second_scenario_pipeline_and_save(
        PATH_SF,
        PATH_SIJK,
        PATH_LPSE,
        OUT_SF,
        OUT_MONITOR
    )

    print("Scenario-2 matching finished.")
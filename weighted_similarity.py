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



########################
# PART 1 HELPER FUNCTION
########################

def load_excel(path_excel):
    df = pd.read_excel(path_excel)
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
    source_label
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
    df_sf   = load_excel(path_sf)
    df_sijk = load_excel(path_sijk)
    df_lpse = load_excel(path_lpse)

    # ---- prepare sf
    df_sf = make_sf_kdkab(df_sf)

    # ---- deduplicate sijk and lpse
    df_sijk = deduplicate_by_key(df_sijk, "nib", "SIJK")
    df_lpse = deduplicate_by_key(df_lpse, "kd_penyedia", "LPSE")

    # ---- SIJK
    insert_sijk, monitor_sijk = process_source_table_with_monitoring(
        df_source = df_sijk,
        df_sf = df_sf,
        source_name_col = "nama_bu",
        source_kdkab_col = "kdkab",
        source_addr_col = "alamat_bu",
        source_label = "sijk"
    )

    # ---- LPSE
    insert_lpse, monitor_lpse = process_source_table_with_monitoring(
        df_source = df_lpse,
        df_sf = df_sf,
        source_name_col = "nama_penyedia",
        source_kdkab_col = "kdkab",
        source_addr_col = "alamat",
        source_label = "lpse"
    )

    # ---- combine
    df_insert_all = pd.concat([insert_sijk, insert_lpse], ignore_index=True)
    df_monitoring = pd.concat([monitor_sijk, monitor_lpse], ignore_index=True)

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

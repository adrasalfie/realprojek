# ====================================================
# A program to match the SF file against LPSE and SIJK
# Proposed by: Shafar, Shasha, Inggar
# ====================================================



import pandas as pd
from rapidfuzz import fuzz
import time
TIME_START = time.time()

# ===================================================
# PART 1: CONFIGURATION GLOBAL
# ===================================================

DEFAULT_MATCH_CONFIG = {
    "threshold" : {
        "name_strong" : 90,
        "name_border" : 88,
        "kab_support" : 90,
        "address_support" : 85,
    }
}

SF_MATCH_CONFIG = {
    "fields" : {
        "name" : "nama_perusahaan",
        "kab" : ["prov", "kab"],
        "address" : "alamat_perusahaan",
    },
    "threshold" : DEFAULT_MATCH_CONFIG["threshold"]
}

SIJK_MATCH_CONFIG = {
    "fields" : {
        "name" : "nama_bu",
        "kab" : "kdkab",
        "address" : "alamat_bu",
    },
    "threshold" : DEFAULT_MATCH_CONFIG["threshold"]
}

LPSE_MATCH_CONFIG = {
    "fields" : {
        "name" : "nama_penyedia",
        "kab" : "kdkab",
        "address" : "alamat",
    },
    "threshold" : DEFAULT_MATCH_CONFIG["threshold"]
}

def get_kab_code(row, kab_field):
    if kab_field is None:
        return None

    # single column
    if isinstance(kab_field, str):
        val = row.get(kab_field, "")
        return "" if pd.isna(val) else str(val)

    # concatenation
    if isinstance(kab_field, (list, tuple)):
        parts = []
        for c in kab_field:
            v = row.get(c, "")
            if pd.isna(v):
                v = ""
            parts.append(str(v))
        return "".join(parts)

    return None

# ===================================================
# PART 2: NORMALISATION
# ===================================================

def normalise_text(x):
    """
    to normalise text data such as missing value as "" and normalise column name

    Parameter:
    x (str): the text to be normalised

    Return:
    str(x).upper().strip() (str)
    """
    if pd.isna(x):
        return ""
    return str(x).upper().strip()

def add_normalised_columns(df, fields):
    df = df.copy()

    for k, col in fields.items():

        # we only normalise real text columns
        if k not in ("name", "address"):
            continue

        # column must be a single column (not list)
        if not isinstance(col, str):
            continue

        if col not in df.columns:
            continue

        df[f"__norm_{k}"] = df[col].apply(normalise_text)

    return df




# ===================================================
# PART 3: SIMILARITY
# to calculate the similarity formula
# ===================================================

def name_similarity(a, b):
    """
    to formulate the similarity score of name

    Parameters:
    a (str): the text as the compared text
    b (str): the base comparison

    Return:
    fuzz.token_set_ratio(a, b) (float)
    """
    if not a or not b:
        return 0
    return fuzz.token_set_ratio(a, b)

def aux_similarity(a, b):
    """
    to formulate the similarity score of auxiliary variables

    Parameters:
    a (str): the text as the compared text
    b (str): the base comparison

    Return:
    fuzz.token_set_ratio(a, b) (float)
    """
    if not a or not b:
        return None
    return fuzz.token_set_ratio(a, b)

def kab_code_match(a, b):
    if not a or not b:
        return None
    return 100 if str(a) == str(b) else 0

# ===================================================
# PART 4: MATCH BY NAMA PERUSAHAAN
# ===================================================

def find_best_name_match(source_name, sf_name_series):
    """
    to compare the similarity by nama perusahaan

    Parameters:
    source_name (str): the source data to be compared
    sf_name_series (series): the series of sf nama perusahaan

    Return:
    best_idx (idx), best_score (float)
    """
    best_idx = None
    best_score = -1

    for idx, sf_name in sf_name_series.items():
        score = name_similarity(source_name, sf_name)
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx, best_score

# ===================================================
# PART 5: CREATE DECISION
# ===================================================

def decide_match(name_score, kab_score, addr_score, thresholds):
    """
    To decide whether the record should be inserted, recompare with kabupaten/kota and address, or not be inserted.

    Parameters:
    name_score (float): the fuzzy similarity score of nama perusahaan
    kab_score (float) : the fuzzy similarity score of kabupaten/kota
    addr_score (float): the fuzzy similarity score of alamat
    threshold (int)   : the threshold set for decision

    Return:
    True (bool), False (bool)
    """

    # Check the strong match name
    if name_score >= thresholds["name_strong"]:
        return True, "MATCH_NAME_STRONG"
    
    # If in a grey area we need supporting comparison
    if name_score >= thresholds["name_border"]:
        support = False # set default as false so that it can be interchangeable during process inside this function

        # Check the kabupaten score if passing the threshold
        if kab_score is not None and kab_score >= thresholds["kab_support"]:
            support = True

        # Check the address score if passing the threshold
        if addr_score is not None and addr_score >= thresholds["address_support"]:
            support = True

        # If support shows the true, then it matches sf
        if support:
            return True, "MATCH_BORDER_WITH_SUPPORT"
        
        # Otherwise, insert because no supporting border
        return False, "BORDER_NO_SUPPORT"
    
    # If nothing matches, then insert to SF
    return False, "WEAK_NAME_APPEND"

# ===================================================
# PART 6: INSERT DATA
# ===================================================

def enrich_sf_row(sf_row, src_row, columns):
    """
    to insert row from unmatched data to sf

    Parameters:
    sf_row (int)  : number of sf row
    src_row (int) : number of source row
    columns (list): list of columns
    
    Return:
    sf_row (int)
    """

    sf_row = sf_row.copy()

    for col in columns:
        if col not in sf_row.index:
            continue

        sf_val = sf_row[col]
        src_val = src_row[col]

        if (pd.isna(sf_val) or str(sf_val).strip() == "") and (not pd.isna(src_val) and str(src_val).strip() != ""):
            sf_row[col] = src_val
    
    return sf_row

def build_new_sf_row_from_source(sf_columns, src_row, sf_fields, src_fields):
    """
    Build a new SF row using logical-field mapping
    + special rule: split kdkab -> prov, kab
    """

    new_row = pd.Series(index=sf_columns, dtype=object)

    # -----------------------------
    # normal logical mapping
    # -----------------------------
    for logical_key, sf_col in sf_fields.items():

        if logical_key not in src_fields:
            continue

        src_col = src_fields[logical_key]

        # normal 1-to-1 column mapping
        if isinstance(sf_col, str) and isinstance(src_col, str):
            if sf_col in sf_columns and src_col in src_row.index:
                new_row[sf_col] = src_row[src_col]

    # -----------------------------
    # special rule for kab code
    # SF : ["prov","kab"]
    # SRC: "kdkab"
    # -----------------------------
    sf_kab_field  = sf_fields.get("kab")
    src_kab_field = src_fields.get("kab")

    if isinstance(sf_kab_field, (list, tuple)) and isinstance(src_kab_field, str):

        if src_kab_field in src_row.index:
            raw = src_row[src_kab_field]

            if pd.notna(raw):
                raw = str(raw).strip()

                if len(raw) >= 4:
                    prov = raw[:2]
                    kab  = raw[-2:]

                    sf_prov_col = sf_kab_field[0]
                    sf_kab_col  = sf_kab_field[1]

                    if sf_prov_col in sf_columns:
                        new_row[sf_prov_col] = prov

                    if sf_kab_col in sf_columns:
                        new_row[sf_kab_col] = kab

    # -----------------------------
    # copy physically common columns
    # (id, nib, kualifikasi, etc)
    # -----------------------------
    for c in sf_columns:
        if c in src_row.index and pd.isna(new_row.get(c)):
            new_row[c] = src_row[c]

    return new_row

# ===================================================
# PART 7: MAIN MERGE ENGINE
# ===================================================

def merge_source_into_sf(sf_df, source_df, sf_config, source_config, preview = False):
    sf_fields = sf_config["fields"]
    src_fields = source_config["fields"]

    sf_name_col = sf_fields["name"]
    src_name_col = src_fields["name"]

    if sf_name_col not in sf_df.columns:
        raise ValueError(f"SF missing required column: {sf_name_col}")

    if src_name_col not in source_df.columns:
        raise ValueError(f"Source missing required column: {src_name_col}")

    thresholds = source_config["threshold"]

    sf = add_normalised_columns(sf_df, sf_fields)
    src = add_normalised_columns(source_df, src_fields)

    
    sf_name_norm = sf["__norm_name"]
    common_columns = [c for c in sf_df.columns if c in source_df.columns]
    preview_rows = []

    for _, src_row in src.iterrows():
        src_name_norm = src_row["__norm_name"]
        best_idx, name_score = find_best_name_match(src_name_norm, sf_name_norm)

        kab_score = None
        addr_score = None

        if best_idx is not None:
            sf_row = sf.loc[best_idx]
            
            # if "kab" in fields and fields["kab"] in sf.columns and fields["kab"] in src.columns:
            #     kab_score = kab_code_match(src_row[fields["kab"]], sf_row[fields["kab"]])

            
            src_kab_field = src_fields.get("kab")
            sf_kab_field  = sf_fields.get("kab")
                
            if src_kab_field is not None and sf_kab_field is not None:
                src_kab = get_kab_code(src_row, src_kab_field)
                sf_kab  = get_kab_code(sf_row, sf_kab_field)
                kab_score = kab_code_match(src_kab, sf_kab)


            if "__norm_address" in sf.columns and "__norm_address" in src.columns:
                addr_score = aux_similarity(src_row["__norm_address"], sf_row["__norm_address"])
            
            matched, decision = decide_match(name_score, kab_score, addr_score, thresholds)

        else:
            matched = False
            decision = "NO_CANDIDATE"

        if matched:
            enriched = enrich_sf_row(sf.loc[best_idx], src_row[common_columns], common_columns)

            sf.loc[best_idx, common_columns] = enriched

            preview_rows.append({
                "source_name" : src_row[src_name_col],
                "sf_name" : sf.loc[best_idx, sf_name_col],
                "name_score" : name_score,
                "kab_score" : kab_score,
                "address_score" : addr_score,
                "decision" : decision
            })

        else:
            
            # Append new row
            # Append new row (mapped into SF schema)
            new_row = build_new_sf_row_from_source(sf.columns, src_row, sf_fields, src_fields)

            sf = pd.concat([sf, new_row.to_frame().T], ignore_index=True)


            # Refresh search index
            sf_name_norm = sf["__norm_name"]
            
            preview_rows.append({
                "source_name" : src_row[src_name_col],
                "sf_name" : "",
                "name_score" : name_score,
                "kab_score" : kab_score,
                "address_score" : addr_score,
                "decision" : decision
            })

    sf = sf.drop(columns = [c for c in sf.columns if c.startswith("__norm_")], errors = "ignore")

    if preview:
        return sf, pd.DataFrame(preview_rows)
    
    return sf

# ===================================================
# PART 8: MAIN PROGRAM
# ===================================================

if __name__ == "__main__":
    sf = pd.read_excel("7500 sf.xlsx", dtype = str)
    sijk = pd.read_excel("7500 sijk.xlsx", dtype = str)
    lpse = pd.read_excel("7500 lpse.xlsx", dtype = str)

    sf2, preview_sijk = merge_source_into_sf(sf, sijk, sf_config = SF_MATCH_CONFIG, source_config = SIJK_MATCH_CONFIG, preview = True)


    preview_sijk.to_excel("preview_merge_sijk.xlsx", index = False)

    sf3, preview_lpse = merge_source_into_sf(sf2, lpse, sf_config = SF_MATCH_CONFIG, source_config = LPSE_MATCH_CONFIG, preview = True)

    preview_lpse.to_excel("preview_merge_lpse.xlsx", index = False)

    sf3.to_excel("sf_final.xlsx", index = False)

    print("Done.")
    END_TIME = time.time()
    PROCESS_TIME = END_TIME - TIME_START
    print(f"processing time: {PROCESS_TIME:.2f} seconds")
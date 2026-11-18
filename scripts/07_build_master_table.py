from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DESC_FILE = (
    ROOT
    / "04_descriptors"
    / "ILs"
    / "IL_descriptors_master.csv"
)

H2O_FILE = (
    ROOT
    / "05_H2O"
    / "production"
    / "H2O_infinite_dilution_298K.csv"
)

H2_FILE = (
    ROOT
    / "06_H2"
    / "production"
    / "H2_infinite_dilution_Henry_298K.csv"
)

OUT = ROOT / "07_master_table"
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

MASTER_FILE = (
    OUT
    / "IL_master_table_4617.csv"
)

REPORT_FILE = (
    OUT
    / "stage7_validation.txt"
)


# ============================================================
# LOAD
# ============================================================

desc = pd.read_csv(DESC_FILE)
h2o = pd.read_csv(H2O_FILE)
h2 = pd.read_csv(H2_FILE)

EXPECTED = 4617


# ============================================================
# PRE-MERGE QC
# ============================================================

for name, df in [
    ("descriptors", desc),
    ("H2O", h2o),
    ("H2", h2),
]:

    if len(df) != EXPECTED:
        raise RuntimeError(
            f"{name}: expected {EXPECTED} rows, "
            f"found {len(df)}"
        )

    if df["IL_ID"].duplicated().any():
        raise RuntimeError(
            f"{name}: duplicate IL_ID detected"
        )


if set(desc["IL_ID"]) != set(h2o["IL_ID"]):
    raise RuntimeError(
        "Descriptor and H2O IL_ID sets differ"
    )

if set(desc["IL_ID"]) != set(h2["IL_ID"]):
    raise RuntimeError(
        "Descriptor and H2 IL_ID sets differ"
    )


# ============================================================
# KEEP ONLY THERMODYNAMIC COLUMNS NEEDED FROM H2O / H2
# ============================================================

h2o_keep = h2o[
    [
        "IL_ID",
        "gamma_inf_H2O",
        "ln_gamma_inf_H2O",
        "status",
    ]
].rename(
    columns={
        "status": "stage5_H2O_status"
    }
)

h2_keep = h2[
    [
        "IL_ID",
        "gamma_inf_H2",
        "ln_gamma_inf_H2",
        "kHcp_H2_mol_L_atm",
        "kHcc_H2_dimensionless",
        "status",
    ]
].rename(
    columns={
        "status": "stage6_H2_status"
    }
)


# ============================================================
# MERGE
# ============================================================

master = (
    desc
    .merge(
        h2o_keep,
        on="IL_ID",
        how="left",
        validate="one_to_one",
    )
    .merge(
        h2_keep,
        on="IL_ID",
        how="left",
        validate="one_to_one",
    )
    .sort_values("IL_ID")
    .reset_index(drop=True)
)


# ============================================================
# ADD FINAL COMPUTATIONAL STATUS
# ============================================================

master["all_thermodynamics_pass"] = (
    (master["stage5_H2O_status"] == "PASS")
    &
    (master["stage6_H2_status"] == "PASS")
)

master["master_status"] = np.where(
    master["stage4c_status"] == "FAIL",
    "FAIL",
    np.where(
        ~master["all_thermodynamics_pass"],
        "FAIL",
        np.where(
            master["contains_warning_ion"],
            "PASS_WITH_SOURCE_WARNING",
            "PASS",
        )
    )
)


# ============================================================
# ORDER IMPORTANT COLUMNS FIRST
# ============================================================

first_cols = [
    "IL_ID",

    "cation_id",
    "cation_name",
    "cation_abbreviation",

    "anion_id",
    "anion_name",
    "anion_abbreviation",

    "cation_count",
    "anion_count",
    "stoichiometry",

    "A_IL_A2",
    "V_IL_A3",

    "MOM2",
    "MOM2_A",

    "MOM3",
    "MOM3_A",

    "HBACC2",
    "HBACC2_A",

    "HBDON2",
    "HBDON2_A",

    "gamma_inf_H2O",
    "ln_gamma_inf_H2O",

    "gamma_inf_H2",
    "ln_gamma_inf_H2",

    "kHcp_H2_mol_L_atm",
    "kHcc_H2_dimensionless",

    "contains_warning_ion",
    "stage4c_status",
    "stage5_H2O_status",
    "stage6_H2_status",
    "master_status",
]

remaining = [
    c
    for c in master.columns
    if c not in first_cols
]

master = master[
    first_cols
    + remaining
]


# ============================================================
# SAVE
# ============================================================

master.to_csv(
    MASTER_FILE,
    index=False,
)


# ============================================================
# HASH
# ============================================================

sha256 = hashlib.sha256(
    MASTER_FILE.read_bytes()
).hexdigest()


# ============================================================
# VALIDATION
# ============================================================

n_pass = int(
    (
        master["master_status"]
        == "PASS"
    ).sum()
)

n_warn = int(
    (
        master["master_status"]
        == "PASS_WITH_SOURCE_WARNING"
    ).sum()
)

n_fail = int(
    (
        master["master_status"]
        == "FAIL"
    ).sum()
)


missing_thermo = int(
    master[
        [
            "gamma_inf_H2O",
            "ln_gamma_inf_H2O",
            "gamma_inf_H2",
            "ln_gamma_inf_H2",
            "kHcp_H2_mol_L_atm",
        ]
    ]
    .isna()
    .any(axis=1)
    .sum()
)


duplicate_ids = int(
    master["IL_ID"]
    .duplicated()
    .sum()
)


nonneutral = int(
    (
        master[
            "formula_unit_net_charge"
        ]
        != 0
    ).sum()
)


if (
    len(master) == EXPECTED
    and
    duplicate_ids == 0
    and
    missing_thermo == 0
    and
    nonneutral == 0
    and
    n_fail == 0
):

    overall = (
        "PASS_WITH_SOURCE_WARNING"
        if n_warn
        else "PASS"
    )

else:

    overall = "FAIL"


# ============================================================
# REPORT
# ============================================================

report = f"""
============================================================
STAGE 7 - MASTER DATASET ASSEMBLY
============================================================

Overall status                  : {overall}

------------------------------------------------------------
ROWS / IDENTIFIERS
------------------------------------------------------------

Expected ILs                    : {EXPECTED}
Master rows                     : {len(master)}

Unique IL IDs                   : {master["IL_ID"].nunique()}
Duplicate IL IDs                : {duplicate_ids}

------------------------------------------------------------
STOICHIOMETRY
------------------------------------------------------------

1:1 ILs                         : {(master["stoichiometry"] == "1:1").sum()}
2:1 ILs                         : {(master["stoichiometry"] == "2:1").sum()}

Non-neutral formula units       : {nonneutral}

------------------------------------------------------------
DATA COMPLETENESS
------------------------------------------------------------

Rows missing thermodynamic data : {missing_thermo}

H2O PASS                        : {(master["stage5_H2O_status"] == "PASS").sum()}
H2 PASS                         : {(master["stage6_H2_status"] == "PASS").sum()}

------------------------------------------------------------
MASTER STATUS
------------------------------------------------------------

PASS                            : {n_pass}
PASS_WITH_SOURCE_WARNING        : {n_warn}
FAIL                            : {n_fail}

------------------------------------------------------------
THERMODYNAMIC RANGES
------------------------------------------------------------

ln_gamma_inf_H2O:
    min = {master["ln_gamma_inf_H2O"].min()}
    max = {master["ln_gamma_inf_H2O"].max()}

ln_gamma_inf_H2:
    min = {master["ln_gamma_inf_H2"].min()}
    max = {master["ln_gamma_inf_H2"].max()}

kHcp_H2 [mol/L/atm]:
    min = {master["kHcp_H2_mol_L_atm"].min()}
    max = {master["kHcp_H2_mol_L_atm"].max()}

------------------------------------------------------------
OUTPUT
------------------------------------------------------------

{MASTER_FILE}

SHA256:
{sha256}

============================================================
"""

REPORT_FILE.write_text(
    report
)

print(report)

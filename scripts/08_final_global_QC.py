from pathlib import Path
import hashlib
import math

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MASTER_FILE = (
    ROOT
    / "07_master_table"
    / "IL_master_table_4617.csv"
)

OUT = ROOT / "08_QC"
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

ROW_QC_FILE = (
    OUT
    / "stage8_row_QC.csv"
)

REPORT_FILE = (
    OUT
    / "stage8_final_QC.txt"
)


# ============================================================
# CONSTANTS / TOLERANCES
# ============================================================

EXPECTED_ILS = 4617
EXPECTED_CATIONS = 81
EXPECTED_ANIONS = 57

EXPECTED_11 = 4536
EXPECTED_21 = 81

EXPECTED_WARNINGS = 57

LOG_TOL = 1.0e-10
NORMALIZED_TOL = 1.0e-10
AREA_TOL_PCT = 0.05


# ============================================================
# LOAD MASTER
# ============================================================

df = pd.read_csv(
    MASTER_FILE
)

if len(df) != EXPECTED_ILS:
    raise RuntimeError(
        f"Expected {EXPECTED_ILS} rows, "
        f"found {len(df)}"
    )


# ============================================================
# MASTER HASH
# ============================================================

master_sha256 = hashlib.sha256(
    MASTER_FILE.read_bytes()
).hexdigest()


# ============================================================
# ROW-LEVEL QC FLAGS
# ============================================================

qc = pd.DataFrame()

qc["IL_ID"] = df["IL_ID"]


# ------------------------------------------------------------
# IDENTIFIERS
# ------------------------------------------------------------

qc["duplicate_IL_ID"] = (
    df["IL_ID"].duplicated(
        keep=False
    )
)

qc["missing_cation_id"] = (
    df["cation_id"].isna()
)

qc["missing_anion_id"] = (
    df["anion_id"].isna()
)


# ------------------------------------------------------------
# STOICHIOMETRY
# ------------------------------------------------------------

qc["nonneutral_formula_unit"] = (
    df["formula_unit_net_charge"]
    != 0
)

qc["bad_stoichiometry"] = ~df[
    "stoichiometry"
].isin(
    [
        "1:1",
        "2:1",
    ]
)


# IDA dianion must be 2:1
qc["A057_not_2to1"] = (
    (df["anion_id"] == "A057")
    &
    (df["stoichiometry"] != "2:1")
)

# All non-IDA anions should be 1:1
qc["non_A057_not_1to1"] = (
    (df["anion_id"] != "A057")
    &
    (df["stoichiometry"] != "1:1")
)


# ------------------------------------------------------------
# REQUIRED NUMERIC FIELDS
# ------------------------------------------------------------

required_numeric = [
    "A_IL_A2",
    "V_IL_A3",
    "MOM0_A2",
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
]

numeric_values = df[
    required_numeric
].apply(
    pd.to_numeric,
    errors="coerce",
)

qc["missing_or_nonfinite_numeric"] = (
    ~np.isfinite(
        numeric_values
    )
).any(axis=1)


# ------------------------------------------------------------
# POSITIVITY / PHYSICAL NUMERICAL SANITY
# ------------------------------------------------------------

qc["bad_area"] = (
    ~np.isfinite(df["A_IL_A2"])
    |
    (df["A_IL_A2"] <= 0)
)

qc["bad_volume"] = (
    ~np.isfinite(df["V_IL_A3"])
    |
    (df["V_IL_A3"] <= 0)
)

qc["bad_MOM0"] = (
    ~np.isfinite(df["MOM0_A2"])
    |
    (df["MOM0_A2"] <= 0)
)

qc["bad_gamma_H2O"] = (
    ~np.isfinite(
        df["gamma_inf_H2O"]
    )
    |
    (
        df["gamma_inf_H2O"]
        <= 0
    )
)

qc["bad_gamma_H2"] = (
    ~np.isfinite(
        df["gamma_inf_H2"]
    )
    |
    (
        df["gamma_inf_H2"]
        <= 0
    )
)

qc["bad_kHcp_H2"] = (
    ~np.isfinite(
        df["kHcp_H2_mol_L_atm"]
    )
    |
    (
        df["kHcp_H2_mol_L_atm"]
        <= 0
    )
)

qc["bad_kHcc_H2"] = (
    ~np.isfinite(
        df["kHcc_H2_dimensionless"]
    )
    |
    (
        df["kHcc_H2_dimensionless"]
        <= 0
    )
)

qc["negative_HB_descriptor"] = (
    (df["HBACC2"] < -1e-12)
    |
    (df["HBDON2"] < -1e-12)
)


# ============================================================
# LOG CONSISTENCY
# ============================================================

calc_ln_H2O = np.log(
    df["gamma_inf_H2O"]
)

calc_ln_H2 = np.log(
    df["gamma_inf_H2"]
)

df_H2O_log_error = (
    df["ln_gamma_inf_H2O"]
    - calc_ln_H2O
)

df_H2_log_error = (
    df["ln_gamma_inf_H2"]
    - calc_ln_H2
)

qc["ln_gamma_H2O_inconsistent"] = (
    np.abs(
        df_H2O_log_error
    )
    > LOG_TOL
)

qc["ln_gamma_H2_inconsistent"] = (
    np.abs(
        df_H2_log_error
    )
    > LOG_TOL
)


# ============================================================
# NORMALIZED DESCRIPTOR CONSISTENCY
# ============================================================

mom2_A_calc = (
    df["MOM2"]
    / df["MOM0_A2"]
)

mom3_A_calc = (
    df["MOM3"]
    / df["MOM0_A2"]
)

hbacc_A_calc = (
    df["HBACC2"]
    / df["MOM0_A2"]
)

hbdon_A_calc = (
    df["HBDON2"]
    / df["MOM0_A2"]
)


mom2_A_err = (
    df["MOM2_A"]
    - mom2_A_calc
)

mom3_A_err = (
    df["MOM3_A"]
    - mom3_A_calc
)

hbacc_A_err = (
    df["HBACC2_A"]
    - hbacc_A_calc
)

hbdon_A_err = (
    df["HBDON2_A"]
    - hbdon_A_calc
)


qc["MOM2_A_inconsistent"] = (
    np.abs(
        mom2_A_err
    )
    > NORMALIZED_TOL
)

qc["MOM3_A_inconsistent"] = (
    np.abs(
        mom3_A_err
    )
    > NORMALIZED_TOL
)

qc["HBACC2_A_inconsistent"] = (
    np.abs(
        hbacc_A_err
    )
    > NORMALIZED_TOL
)

qc["HBDON2_A_inconsistent"] = (
    np.abs(
        hbdon_A_err
    )
    > NORMALIZED_TOL
)


# ============================================================
# AREA CONSISTENCY
# ============================================================

area_error_pct_recalc = (
    100.0
    * (
        df["MOM0_A2"]
        - df["A_IL_A2"]
    )
    / df["A_IL_A2"]
)

qc["area_consistency_fail"] = (
    np.abs(
        area_error_pct_recalc
    )
    > AREA_TOL_PCT
)


# ============================================================
# STAGE STATUS CONSISTENCY
# ============================================================

qc["stage5_not_PASS"] = (
    df["stage5_H2O_status"]
    != "PASS"
)

qc["stage6_not_PASS"] = (
    df["stage6_H2_status"]
    != "PASS"
)

qc["stage4_FAIL"] = (
    df["stage4c_status"]
    == "FAIL"
)


# ============================================================
# C071 SOURCE-WARNING AUDIT
# ============================================================

expected_warning = (
    df["cation_id"]
    == "C071"
)

actual_warning = (
    df["contains_warning_ion"]
    == True
)

qc["warning_flag_mismatch"] = (
    expected_warning
    != actual_warning
)

expected_master_status = np.where(
    expected_warning,
    "PASS_WITH_SOURCE_WARNING",
    "PASS",
)

qc["master_status_mismatch"] = (
    df["master_status"]
    != expected_master_status
)


# ============================================================
# HARD FAILURE SUMMARY PER ROW
# ============================================================

hard_failure_columns = [
    "duplicate_IL_ID",
    "missing_cation_id",
    "missing_anion_id",

    "nonneutral_formula_unit",
    "bad_stoichiometry",
    "A057_not_2to1",
    "non_A057_not_1to1",

    "missing_or_nonfinite_numeric",

    "bad_area",
    "bad_volume",
    "bad_MOM0",

    "bad_gamma_H2O",
    "bad_gamma_H2",

    "bad_kHcp_H2",
    "bad_kHcc_H2",

    "negative_HB_descriptor",

    "ln_gamma_H2O_inconsistent",
    "ln_gamma_H2_inconsistent",

    "MOM2_A_inconsistent",
    "MOM3_A_inconsistent",
    "HBACC2_A_inconsistent",
    "HBDON2_A_inconsistent",

    "area_consistency_fail",

    "stage5_not_PASS",
    "stage6_not_PASS",
    "stage4_FAIL",

    "warning_flag_mismatch",
    "master_status_mismatch",
]

qc["hard_failure"] = (
    qc[
        hard_failure_columns
    ]
    .any(axis=1)
)


qc["source_warning"] = (
    expected_warning
)


qc["final_QC_status"] = np.where(
    qc["hard_failure"],
    "FAIL",
    np.where(
        qc["source_warning"],
        "PASS_WITH_SOURCE_WARNING",
        "PASS",
    )
)


# ============================================================
# SAVE ROW QC
# ============================================================

qc.to_csv(
    ROW_QC_FILE,
    index=False,
)


# ============================================================
# GLOBAL COUNTS
# ============================================================

unique_IL = (
    df["IL_ID"].nunique()
)

unique_cations = (
    df["cation_id"].nunique()
)

unique_anions = (
    df["anion_id"].nunique()
)

unique_pairs = (
    df[
        [
            "cation_id",
            "anion_id",
        ]
    ]
    .drop_duplicates()
    .shape[0]
)

count_11 = int(
    (
        df["stoichiometry"]
        == "1:1"
    ).sum()
)

count_21 = int(
    (
        df["stoichiometry"]
        == "2:1"
    ).sum()
)

count_A057 = int(
    (
        df["anion_id"]
        == "A057"
    ).sum()
)

warning_count = int(
    expected_warning.sum()
)

n_pass = int(
    (
        qc["final_QC_status"]
        == "PASS"
    ).sum()
)

n_warn = int(
    (
        qc["final_QC_status"]
        == "PASS_WITH_SOURCE_WARNING"
    ).sum()
)

n_fail = int(
    (
        qc["final_QC_status"]
        == "FAIL"
    ).sum()
)


# ============================================================
# IL ID SEQUENCE CHECK
# ============================================================

expected_ids = [
    f"IL{i:04d}"
    for i in range(
        1,
        EXPECTED_ILS + 1
    )
]

actual_ids = (
    df["IL_ID"]
    .sort_values()
    .tolist()
)

IL_sequence_ok = (
    actual_ids
    == expected_ids
)


# ============================================================
# OVERALL RESULT
# ============================================================

global_checks = {

    "rows":
        len(df) == EXPECTED_ILS,

    "unique_IL":
        unique_IL == EXPECTED_ILS,

    "IL_sequence":
        IL_sequence_ok,

    "cations":
        unique_cations
        == EXPECTED_CATIONS,

    "anions":
        unique_anions
        == EXPECTED_ANIONS,

    "pairs":
        unique_pairs
        == EXPECTED_ILS,

    "1to1":
        count_11
        == EXPECTED_11,

    "2to1":
        count_21
        == EXPECTED_21,

    "A057_count":
        count_A057
        == EXPECTED_21,

    "warnings":
        warning_count
        == EXPECTED_WARNINGS,

    "no_hard_failures":
        n_fail == 0,
}


if all(
    global_checks.values()
):

    if n_warn > 0:
        overall = (
            "PASS_WITH_SOURCE_WARNING"
        )
    else:
        overall = "PASS"

else:

    overall = "FAIL"


# ============================================================
# COLUMN FAILURE COUNTS
# ============================================================

failure_counts = []

for col in hard_failure_columns:

    failure_counts.append(
        (
            col,
            int(
                qc[col].sum()
            )
        )
    )

failure_text = "\n".join(
    f"{name:35s}: {count}"
    for name, count
    in failure_counts
)


# ============================================================
# REPORT
# ============================================================

report = f"""
============================================================
STAGE 8 - FINAL GLOBAL QC
============================================================

Overall status                  : {overall}

------------------------------------------------------------
MASTER DATASET IDENTITY
------------------------------------------------------------

Rows                            : {len(df)}
Expected rows                   : {EXPECTED_ILS}

Unique IL IDs                   : {unique_IL}
IL ID sequence IL0001-IL4617    : {IL_sequence_ok}

Unique cations                  : {unique_cations}
Unique anions                   : {unique_anions}
Unique cation-anion pairs       : {unique_pairs}

Master SHA256:
{master_sha256}

------------------------------------------------------------
STOICHIOMETRY
------------------------------------------------------------

1:1 ILs                         : {count_11}
Expected 1:1                    : {EXPECTED_11}

2:1 ILs                         : {count_21}
Expected 2:1                    : {EXPECTED_21}

A057 combinations               : {count_A057}

Non-neutral formula units       : {int(qc["nonneutral_formula_unit"].sum())}

------------------------------------------------------------
FINAL ROW STATUS
------------------------------------------------------------

PASS                            : {n_pass}
PASS_WITH_SOURCE_WARNING        : {n_warn}
FAIL                            : {n_fail}

Expected source warnings        : {EXPECTED_WARNINGS}
Actual C071 warning rows        : {warning_count}

------------------------------------------------------------
THERMODYNAMIC QC
------------------------------------------------------------

Invalid gamma_inf_H2O           : {int(qc["bad_gamma_H2O"].sum())}
Invalid gamma_inf_H2            : {int(qc["bad_gamma_H2"].sum())}

Invalid kHcp_H2                 : {int(qc["bad_kHcp_H2"].sum())}
Invalid kHcc_H2                 : {int(qc["bad_kHcc_H2"].sum())}

ln(gamma_H2O) inconsistencies   : {int(qc["ln_gamma_H2O_inconsistent"].sum())}
ln(gamma_H2) inconsistencies    : {int(qc["ln_gamma_H2_inconsistent"].sum())}

Maximum |ln H2O consistency err|:
{np.max(np.abs(df_H2O_log_error))}

Maximum |ln H2 consistency err|:
{np.max(np.abs(df_H2_log_error))}

------------------------------------------------------------
DESCRIPTOR QC
------------------------------------------------------------

Area consistency failures       : {int(qc["area_consistency_fail"].sum())}

Maximum |area error| %          :
{np.max(np.abs(area_error_pct_recalc))}

MOM2/A inconsistencies          : {int(qc["MOM2_A_inconsistent"].sum())}
MOM3/A inconsistencies          : {int(qc["MOM3_A_inconsistent"].sum())}

HBACC2/A inconsistencies        : {int(qc["HBACC2_A_inconsistent"].sum())}
HBDON2/A inconsistencies        : {int(qc["HBDON2_A_inconsistent"].sum())}

Negative HB descriptors         : {int(qc["negative_HB_descriptor"].sum())}

------------------------------------------------------------
STATUS / WARNING AUDIT
------------------------------------------------------------

Stage 4 failures                : {int(qc["stage4_FAIL"].sum())}
Stage 5 non-PASS                : {int(qc["stage5_not_PASS"].sum())}
Stage 6 non-PASS                : {int(qc["stage6_not_PASS"].sum())}

Warning-flag mismatches         : {int(qc["warning_flag_mismatch"].sum())}
Master-status mismatches        : {int(qc["master_status_mismatch"].sum())}

------------------------------------------------------------
HARD FAILURE COUNTS BY TEST
------------------------------------------------------------

{failure_text}

------------------------------------------------------------
FINAL PROPERTY RANGES
------------------------------------------------------------

A_IL [A^2]:
    {df["A_IL_A2"].min()}
    to
    {df["A_IL_A2"].max()}

V_IL [A^3]:
    {df["V_IL_A3"].min()}
    to
    {df["V_IL_A3"].max()}

ln gamma_inf H2O:
    {df["ln_gamma_inf_H2O"].min()}
    to
    {df["ln_gamma_inf_H2O"].max()}

ln gamma_inf H2:
    {df["ln_gamma_inf_H2"].min()}
    to
    {df["ln_gamma_inf_H2"].max()}

kHcp H2 [mol/L/atm]:
    {df["kHcp_H2_mol_L_atm"].min()}
    to
    {df["kHcp_H2_mol_L_atm"].max()}

------------------------------------------------------------
OUTPUTS
------------------------------------------------------------

Master dataset:
{MASTER_FILE}

Row-level QC:
{ROW_QC_FILE}

Final QC report:
{REPORT_FILE}

------------------------------------------------------------
INTERPRETATION
------------------------------------------------------------

PASS_WITH_SOURCE_WARNING means:

- no IL failed the final numerical/data QC;
- the 57 flagged rows are exactly the ILs containing C071;
- C071 carries the previously documented ion-level MOM1
  charge warning;
- its direct 1:1 IL validation and H2O/H2 thermodynamic
  calculations passed;
- these rows remain in the dataset but retain transparent
  provenance/QC flags.

No screening or ranking criterion is applied in Stage 8.

============================================================
"""

REPORT_FILE.write_text(
    report
)

print(report)

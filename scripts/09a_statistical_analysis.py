from pathlib import Path
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

OUT = ROOT / "09_statistics"
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

DESC_FILE = OUT / "descriptive_statistics.csv"
PEARSON_FILE = OUT / "correlation_pearson_all.csv"
SPEARMAN_FILE = OUT / "correlation_spearman_all.csv"

PEARSON_CLEAN_FILE = (
    OUT / "correlation_pearson_clean.csv"
)

SPEARMAN_CLEAN_FILE = (
    OUT / "correlation_spearman_clean.csv"
)

CATION_FILE = (
    OUT / "cation_marginal_effects.csv"
)

ANION_FILE = (
    OUT / "anion_marginal_effects.csv"
)

VARIANCE_FILE = (
    OUT / "cation_anion_variance_decomposition.csv"
)

TARGET_CORR_FILE = (
    OUT / "target_correlations.csv"
)

REPORT_FILE = (
    OUT / "stage9a_statistics_report.txt"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    MASTER_FILE
)

if len(df) != 4617:
    raise RuntimeError(
        f"Expected 4617 rows, found {len(df)}"
    )


# ============================================================
# DERIVED VARIABLES
# ============================================================

if (
    df["kHcp_H2_mol_L_atm"] <= 0
).any():

    raise RuntimeError(
        "Nonpositive kHcp encountered."
    )


df["log10_kHcp_H2"] = np.log10(
    df["kHcp_H2_mol_L_atm"]
)


# ============================================================
# CORE ANALYSIS VARIABLES
#
# Avoid deliberately duplicating gamma + ln(gamma),
# and avoid using both kHcp and kHcc in the core matrix.
# ============================================================

features = [

    # size
    "A_IL_A2",
    "V_IL_A3",

    # normalized intrinsic descriptors
    "MOM2_A",
    "MOM3_A",
    "HBACC2_A",
    "HBDON2_A",

    # water thermodynamics
    "ln_gamma_inf_H2O",

    # hydrogen thermodynamics
    "ln_gamma_inf_H2",
    "log10_kHcp_H2",
]


targets = [
    "ln_gamma_inf_H2O",
    "ln_gamma_inf_H2",
    "log10_kHcp_H2",
]


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

desc = (
    df[features]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
    .T
)

desc["skew"] = (
    df[features]
    .skew()
)

desc["kurtosis"] = (
    df[features]
    .kurt()
)

desc.to_csv(
    DESC_FILE
)


# ============================================================
# CORRELATIONS — ALL 4617
# ============================================================

pearson = (
    df[features]
    .corr(
        method="pearson"
    )
)

spearman = (
    df[features]
    .corr(
        method="spearman"
    )
)

pearson.to_csv(
    PEARSON_FILE
)

spearman.to_csv(
    SPEARMAN_FILE
)


# ============================================================
# SENSITIVITY ANALYSIS:
# EXCLUDE C071 WARNING ROWS
# ============================================================

clean = df[
    df["master_status"]
    == "PASS"
].copy()

pearson_clean = (
    clean[features]
    .corr(
        method="pearson"
    )
)

spearman_clean = (
    clean[features]
    .corr(
        method="spearman"
    )
)

pearson_clean.to_csv(
    PEARSON_CLEAN_FILE
)

spearman_clean.to_csv(
    SPEARMAN_CLEAN_FILE
)


# ============================================================
# CORRELATIONS WITH KEY TARGETS
# ============================================================

rows = []

for target in targets:

    for variable in features:

        if variable == target:
            continue

        rows.append({

            "target":
                target,

            "variable":
                variable,

            "pearson_r":
                pearson.loc[
                    target,
                    variable
                ],

            "spearman_rho":
                spearman.loc[
                    target,
                    variable
                ],

            "abs_pearson_r":
                abs(
                    pearson.loc[
                        target,
                        variable
                    ]
                ),

            "abs_spearman_rho":
                abs(
                    spearman.loc[
                        target,
                        variable
                    ]
                ),
        })


target_corr = pd.DataFrame(
    rows
)

target_corr = target_corr.sort_values(
    [
        "target",
        "abs_spearman_rho",
    ],
    ascending=[
        True,
        False,
    ],
)

target_corr.to_csv(
    TARGET_CORR_FILE,
    index=False,
)


# ============================================================
# CATION / ANION MARGINAL EFFECTS
#
# Because every cation is paired with every anion,
# simple marginal means are scientifically interpretable.
# ============================================================

cation = (
    df
    .groupby(
        [
            "cation_id",
            "cation_abbreviation",
        ],
        dropna=False,
    )[targets]
    .agg(
        [
            "mean",
            "std",
            "min",
            "max",
        ]
    )
)

cation.columns = [
    "_".join(col)
    for col in cation.columns
]

cation = (
    cation
    .reset_index()
)

cation.to_csv(
    CATION_FILE,
    index=False,
)


anion = (
    df
    .groupby(
        [
            "anion_id",
            "anion_abbreviation",
        ],
        dropna=False,
    )[targets]
    .agg(
        [
            "mean",
            "std",
            "min",
            "max",
        ]
    )
)

anion.columns = [
    "_".join(col)
    for col in anion.columns
]

anion = (
    anion
    .reset_index()
)

anion.to_csv(
    ANION_FILE,
    index=False,
)


# ============================================================
# TWO-WAY FACTORIAL VARIANCE DECOMPOSITION
#
# y_ij =
# grand mean
# + cation effect
# + anion effect
# + cation-anion residual/interaction
#
# Complete 81 × 57 design allows orthogonal decomposition.
# ============================================================

variance_rows = []

n_cations = (
    df["cation_id"].nunique()
)

n_anions = (
    df["anion_id"].nunique()
)


for target in targets:

    y = df[target].astype(
        float
    )

    grand = float(
        y.mean()
    )

    cmean = (
        df.groupby(
            "cation_id"
        )[target]
        .mean()
    )

    amean = (
        df.groupby(
            "anion_id"
        )[target]
        .mean()
    )


    pred = (
        grand
        +
        df["cation_id"].map(
            cmean
        ).to_numpy()
        - grand
        +
        df["anion_id"].map(
            amean
        ).to_numpy()
        - grand
    )


    resid = (
        y.to_numpy()
        - pred
    )


    SST = float(
        np.sum(
            (
                y.to_numpy()
                - grand
            )**2
        )
    )


    SS_cation = float(
        n_anions
        *
        np.sum(
            (
                cmean.to_numpy()
                - grand
            )**2
        )
    )


    SS_anion = float(
        n_cations
        *
        np.sum(
            (
                amean.to_numpy()
                - grand
            )**2
        )
    )


    SS_interaction = float(
        np.sum(
            resid**2
        )
    )


    variance_rows.append({

        "property":
            target,

        "grand_mean":
            grand,

        "SS_total":
            SST,

        "SS_cation":
            SS_cation,

        "SS_anion":
            SS_anion,

        "SS_interaction_residual":
            SS_interaction,

        "fraction_cation":
            SS_cation / SST,

        "fraction_anion":
            SS_anion / SST,

        "fraction_interaction":
            SS_interaction / SST,

        "additive_R2":
            1.0
            - SS_interaction / SST,

        "closure_error":
            SST
            - (
                SS_cation
                + SS_anion
                + SS_interaction
            ),
    })


variance = pd.DataFrame(
    variance_rows
)

variance.to_csv(
    VARIANCE_FILE,
    index=False,
)


# ============================================================
# C071 SENSITIVITY
# ============================================================

max_corr_change = float(
    np.max(
        np.abs(
            pearson.to_numpy()
            -
            pearson_clean.to_numpy()
        )
    )
)


# ============================================================
# REPORT
# ============================================================

report = f"""
============================================================
STAGE 9A - STATISTICAL AUDIT
============================================================

Dataset rows                     : {len(df)}
Clean PASS rows                  : {len(clean)}
Source-warning rows              : {len(df) - len(clean)}

Unique cations                   : {df["cation_id"].nunique()}
Unique anions                    : {df["anion_id"].nunique()}

------------------------------------------------------------
CORE VARIABLES
------------------------------------------------------------

{chr(10).join(features)}

------------------------------------------------------------
THERMODYNAMIC TARGETS
------------------------------------------------------------

{chr(10).join(targets)}

------------------------------------------------------------
C071 SENSITIVITY
------------------------------------------------------------

Maximum absolute change in any
Pearson correlation after
excluding the 57 C071 rows       : {max_corr_change}

------------------------------------------------------------
CATION / ANION VARIANCE DECOMPOSITION
------------------------------------------------------------

{variance.to_string(index=False)}

------------------------------------------------------------
TOP CORRELATIONS WITH TARGETS
------------------------------------------------------------

{target_corr.groupby("target").head(5).to_string(index=False)}

------------------------------------------------------------
OUTPUTS
------------------------------------------------------------

Descriptive statistics:
{DESC_FILE}

Pearson matrix:
{PEARSON_FILE}

Spearman matrix:
{SPEARMAN_FILE}

Pearson matrix, clean-only:
{PEARSON_CLEAN_FILE}

Spearman matrix, clean-only:
{SPEARMAN_CLEAN_FILE}

Target correlations:
{TARGET_CORR_FILE}

Cation marginal effects:
{CATION_FILE}

Anion marginal effects:
{ANION_FILE}

Cation/anion variance decomposition:
{VARIANCE_FILE}

------------------------------------------------------------
IMPORTANT
------------------------------------------------------------

No ranking, Pareto filtering, PCA, or clustering has been
performed in Stage 9A.

This stage is intended to identify descriptor redundancy,
thermodynamic relationships, and relative cation/anion
control before selecting variables for later analyses.

============================================================
"""

REPORT_FILE.write_text(
    report
)

print(report)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Repository default: numerical outputs only.
SAVE_FIGURES = False


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MASTER_FILE = (
    ROOT
    / "07_master_table"
    / "IL_master_table_4617.csv"
)

OUT = (
    ROOT
    / "10_pareto"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PARETO_FILE = (
    OUT
    / "pareto_front_primary.csv"
)

ALL_FLAGGED_FILE = (
    OUT
    / "IL_master_with_pareto_flag.csv"
)

REPORT_FILE = (
    OUT
    / "stage10a_pareto_report.txt"
)

FIG_PNG = (
    OUT
    / "pareto_H2O_H2_primary.png"
)

FIG_PDF = (
    OUT
    / "pareto_H2O_H2_primary.pdf"
)


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({

    "font.family": "serif",

    "font.serif": [
        "Times New Roman",
        "Times",
        "DejaVu Serif",
    ],

    "font.size": 11,

    "axes.labelsize": 12,

    "axes.titlesize": 13,

    "xtick.labelsize": 10,

    "ytick.labelsize": 10,

    "legend.fontsize": 10,

    "figure.dpi": 150,

    "savefig.dpi": 600,

    "axes.linewidth": 1.0,
})


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    MASTER_FILE
)

if len(df) != 4617:

    raise RuntimeError(
        f"Expected 4617 rows, "
        f"found {len(df)}"
    )


# ============================================================
# DERIVED H2 SOLUBILITY VARIABLE
# ============================================================

if (
    df["kHcp_H2_mol_L_atm"]
    <= 0
).any():

    raise RuntimeError(
        "Nonpositive kHcp detected."
    )


df["log10_kHcp_H2"] = np.log10(
    df["kHcp_H2_mol_L_atm"]
)


# ============================================================
# OBJECTIVES
#
# Objective 1:
#   minimize ln gamma infinity H2O
#
# Objective 2:
#   maximize log10 kHcp H2
# ============================================================

X = "ln_gamma_inf_H2O"

Y = "log10_kHcp_H2"


if (
    df[[X, Y]]
    .isna()
    .any()
    .any()
):

    raise RuntimeError(
        "Missing Pareto objective values."
    )


# ============================================================
# PARETO FRONT
#
# Point i is dominated if another point has:
#
# x_j <= x_i
# y_j >= y_i
#
# with at least one strict inequality.
#
# Lower X = better
# Higher Y = better
# ============================================================

df["pareto_primary"] = False


# Small numerical tolerance
TOL = 1.0e-12


# Group by identical X values.
# This gives a robust O(n log n) Pareto calculation.

ordered_x = np.sort(
    df[X].unique()
)

best_y_so_far = -np.inf


for xval in ordered_x:

    mask = np.isclose(
        df[X].to_numpy(),
        xval,
        atol=TOL,
        rtol=0.0,
    )

    group_y = df.loc[
        mask,
        Y
    ]

    group_max_y = float(
        group_y.max()
    )


    # Current x-position contributes to Pareto front only
    # if its best Y exceeds everything already encountered
    # at smaller X.

    if (
        group_max_y
        > best_y_so_far + TOL
    ):

        best_mask = (
            mask
            &
            np.isclose(
                df[Y].to_numpy(),
                group_max_y,
                atol=TOL,
                rtol=0.0,
            )
        )

        df.loc[
            best_mask,
            "pareto_primary"
        ] = True

        best_y_so_far = (
            group_max_y
        )


# ============================================================
# EXTRACT PARETO FRONT
# ============================================================

pareto = (
    df[
        df[
            "pareto_primary"
        ]
    ]
    .copy()
    .sort_values(
        [
            X,
            Y,
        ],
        ascending=[
            True,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)


pareto[
    "pareto_order"
] = np.arange(
    1,
    len(pareto) + 1
)


# ============================================================
# CHECK NON-DOMINATION EXPLICITLY
# ============================================================

def is_dominated(
    row,
    data,
):

    better_or_equal_x = (
        data[X]
        <= row[X] + TOL
    )

    better_or_equal_y = (
        data[Y]
        >= row[Y] - TOL
    )

    strictly_better = (
        (
            data[X]
            < row[X] - TOL
        )
        |
        (
            data[Y]
            > row[Y] + TOL
        )
    )

    return bool(
        (
            better_or_equal_x
            &
            better_or_equal_y
            &
            strictly_better
        ).any()
    )


validation_failures = 0

for _, row in pareto.iterrows():

    if is_dominated(
        row,
        df
    ):

        validation_failures += 1


# ============================================================
# SAVE TABLES
# ============================================================

important_columns = [

    "pareto_order",

    "IL_ID",

    "cation_id",
    "cation_name",
    "cation_abbreviation",

    "anion_id",
    "anion_name",
    "anion_abbreviation",

    "stoichiometry",

    "ln_gamma_inf_H2O",

    "gamma_inf_H2O",

    "log10_kHcp_H2",

    "kHcp_H2_mol_L_atm",

    "ln_gamma_inf_H2",

    "gamma_inf_H2",

    "MOM2_A",
    "MOM3_A",

    "HBACC2_A",
    "HBDON2_A",

    "A_IL_A2",
    "V_IL_A3",

    "master_status",
]


pareto[
    important_columns
].to_csv(
    PARETO_FILE,
    index=False,
)


df.to_csv(
    ALL_FLAGGED_FILE,
    index=False,
)


# ============================================================
# EXTREME PARETO MEMBERS
# ============================================================

water_extreme = pareto.loc[
    pareto[X].idxmin()
]

h2_extreme = pareto.loc[
    pareto[Y].idxmax()
]


# ============================================================
# PARETO CHEMICAL DIVERSITY COUNTS
# ============================================================

pareto_cations = (
    pareto[
        "cation_id"
    ].nunique()
)

pareto_anions = (
    pareto[
        "anion_id"
    ].nunique()
)

pareto_11 = int(
    (
        pareto[
            "stoichiometry"
        ]
        == "1:1"
    ).sum()
)

pareto_21 = int(
    (
        pareto[
            "stoichiometry"
        ]
        == "2:1"
    ).sum()
)


# ============================================================
# MOST FREQUENT PARETO CATIONS / ANIONS
# ============================================================

top_cations = (
    pareto[
        [
            "cation_id",
            "cation_abbreviation",
        ]
    ]
    .value_counts()
    .reset_index(
        name="count"
    )
    .head(10)
)


top_anions = (
    pareto[
        [
            "anion_id",
            "anion_abbreviation",
        ]
    ]
    .value_counts()
    .reset_index(
        name="count"
    )
    .head(10)
)


# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.0, 5.5)
)


# All ILs
ax.scatter(
    df[X],
    df[Y],
    s=11,
    alpha=0.30,
    label="All ionic liquids",
)


# Pareto front
ax.scatter(
    pareto[X],
    pareto[Y],
    s=55,
    facecolors="none",
    linewidths=1.4,
    label="Pareto front",
)


# Join Pareto front in objective order
ax.plot(
    pareto[X],
    pareto[Y],
    linewidth=1.0,
)


# Extreme labels only
ax.annotate(

    str(
        water_extreme[
            "IL_ID"
        ]
    ),

    (
        water_extreme[X],
        water_extreme[Y],
    ),

    xytext=(8, 8),

    textcoords="offset points",

    fontsize=9,
)


ax.annotate(

    str(
        h2_extreme[
            "IL_ID"
        ]
    ),

    (
        h2_extreme[X],
        h2_extreme[Y],
    ),

    xytext=(8, -14),

    textcoords="offset points",

    fontsize=9,
)


ax.set_xlabel(
    r"$\ln\gamma^\infty_{\mathrm{H_2O}}$"
)

ax.set_ylabel(
    r"$\log_{10}k_H^{cp}(\mathrm{H_2})$"
)


ax.set_title(
    "Primary Pareto screening of 4,617 ionic liquids"
)


ax.text(

    0.03,
    0.04,

    "Preferred direction:\n"
    r"$\leftarrow$ lower "
    r"$\ln\gamma^\infty_{\mathrm{H_2O}}$"
    "\n"
    r"$\uparrow$ higher "
    r"$k_H^{cp}(\mathrm{H_2})$",

    transform=ax.transAxes,

    va="bottom",
)


ax.legend(
    frameon=False
)


fig.tight_layout()


if SAVE_FIGURES:
    fig.savefig(
        FIG_PNG,
        dpi=600,
        bbox_inches="tight",
    )

if SAVE_FIGURES:
    fig.savefig(
        FIG_PDF,
        bbox_inches="tight",
    )

plt.close(
    fig
)


# ============================================================
# REPORT
# ============================================================

front_fraction = (
    100.0
    * len(pareto)
    / len(df)
)


report = f"""
============================================================
STAGE 10A - PRIMARY PARETO SCREENING
============================================================

Dataset                          : {len(df)} ILs

------------------------------------------------------------
OBJECTIVES
------------------------------------------------------------

Objective 1:
MINIMIZE ln_gamma_inf_H2O

Interpretation:
Lower values correspond to stronger H2O affinity at
infinite dilution.

Objective 2:
MAXIMIZE log10_kHcp_H2

Interpretation:
Higher kHcp corresponds to greater dissolved H2
concentration per unit H2 partial pressure.

No descriptor such as MOM2/A, MOM3/A, HBACC2/A,
HBDON2/A, area, or volume was included directly as a
Pareto objective.

------------------------------------------------------------
PARETO FRONT
------------------------------------------------------------

Pareto ILs                       : {len(pareto)}
Fraction of total dataset (%)    : {front_fraction}

Direct non-domination failures   : {validation_failures}

Unique Pareto cations            : {pareto_cations}
Unique Pareto anions             : {pareto_anions}

Pareto 1:1 ILs                   : {pareto_11}
Pareto 2:1 ILs                   : {pareto_21}

------------------------------------------------------------
WATER-AFFINITY EXTREME
------------------------------------------------------------

IL_ID:
{water_extreme["IL_ID"]}

Cation:
{water_extreme["cation_abbreviation"]}

Anion:
{water_extreme["anion_abbreviation"]}

Stoichiometry:
{water_extreme["stoichiometry"]}

ln_gamma_inf_H2O:
{water_extreme["ln_gamma_inf_H2O"]}

kHcp_H2 [mol/L/atm]:
{water_extreme["kHcp_H2_mol_L_atm"]}

------------------------------------------------------------
H2-SOLUBILITY EXTREME
------------------------------------------------------------

IL_ID:
{h2_extreme["IL_ID"]}

Cation:
{h2_extreme["cation_abbreviation"]}

Anion:
{h2_extreme["anion_abbreviation"]}

Stoichiometry:
{h2_extreme["stoichiometry"]}

ln_gamma_inf_H2O:
{h2_extreme["ln_gamma_inf_H2O"]}

kHcp_H2 [mol/L/atm]:
{h2_extreme["kHcp_H2_mol_L_atm"]}

------------------------------------------------------------
MOST FREQUENT CATIONS ON PARETO FRONT
------------------------------------------------------------

{top_cations.to_string(index=False)}

------------------------------------------------------------
MOST FREQUENT ANIONS ON PARETO FRONT
------------------------------------------------------------

{top_anions.to_string(index=False)}

------------------------------------------------------------
OUTPUTS
------------------------------------------------------------

Pareto table:
{PARETO_FILE}

Master table with Pareto flag:
{ALL_FLAGGED_FILE}

Figure PNG:
{FIG_PNG}

Figure PDF:
{FIG_PDF}

------------------------------------------------------------
IMPORTANT INTERPRETATION
------------------------------------------------------------

This is a thermodynamic Pareto screen.

It identifies ionic liquids that are non-dominated with
respect to the selected H2O-affinity and H2-solubility
objectives.

It is NOT an HER activity ranking.

No arbitrary weighted score has been introduced.

============================================================
"""


REPORT_FILE.write_text(
    report
)


print(
    report
)

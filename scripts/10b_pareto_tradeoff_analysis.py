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

PARETO_FILE = (
    ROOT
    / "10_pareto"
    / "pareto_front_primary.csv"
)

OUT = ROOT / "10_pareto"

METRICS_FILE = (
    OUT
    / "pareto_tradeoff_metrics.csv"
)

REPORT_FILE = (
    OUT
    / "stage10b_tradeoff_report.txt"
)

FIG1_PNG = (
    OUT
    / "pareto_tradeoff_knee.png"
)

FIG1_PDF = (
    OUT
    / "pareto_tradeoff_knee.pdf"
)

FIG2_PNG = (
    OUT
    / "pareto_adjacent_tradeoff_efficiency.png"
)

FIG2_PDF = (
    OUT
    / "pareto_adjacent_tradeoff_efficiency.pdf"
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

master = pd.read_csv(
    MASTER_FILE
)

pareto = pd.read_csv(
    PARETO_FILE
).sort_values(
    "pareto_order"
).reset_index(
    drop=True
)


if len(pareto) < 3:
    raise RuntimeError(
        "Need at least 3 Pareto points."
    )


# ============================================================
# VARIABLES
# ============================================================

X = "ln_gamma_inf_H2O"
K = "kHcp_H2_mol_L_atm"

pareto[
    "log10_kHcp_H2"
] = np.log10(
    pareto[K]
)

Y = "log10_kHcp_H2"


# ============================================================
# STEPWISE TRADE-OFFS
# ============================================================

pareto[
    "delta_ln_gamma_H2O"
] = pareto[X].diff()

pareto[
    "delta_kHcp_H2"
] = pareto[K].diff()

pareto[
    "percent_gain_kHcp_H2"
] = (
    pareto[K].pct_change()
    * 100.0
)

pareto[
    "delta_log10_kHcp_H2"
] = pareto[Y].diff()


# H2 gain in log-space per unit loss in H2O objective.
# Larger = more efficient Pareto movement.
pareto[
    "tradeoff_efficiency"
] = (
    pareto[
        "delta_log10_kHcp_H2"
    ]
    /
    pareto[
        "delta_ln_gamma_H2O"
    ]
)


# Water penalty for each 1% H2-solubility gain.
# Larger = more expensive transition.
pareto[
    "water_penalty_per_1pct_H2_gain"
] = (
    pareto[
        "delta_ln_gamma_H2O"
    ]
    /
    pareto[
        "percent_gain_kHcp_H2"
    ]
)


# ============================================================
# TRANSITION LABELS
# ============================================================

pareto[
    "transition"
] = ""

for i in range(
    1,
    len(pareto)
):

    pareto.loc[
        i,
        "transition"
    ] = (
        str(
            pareto.loc[
                i - 1,
                "IL_ID"
            ]
        )
        +
        " → "
        +
        str(
            pareto.loc[
                i,
                "IL_ID"
            ]
        )
    )


# ============================================================
# NORMALIZED OBJECTIVE SPACE
#
# Both transformed so larger = better.
#
# water_benefit:
# 1 = strongest water-affinity end
# 0 = weakest water-affinity end
#
# H2_benefit:
# 0 = lowest H2 solubility on front
# 1 = highest H2 solubility on front
# ============================================================

x_min = float(
    pareto[X].min()
)

x_max = float(
    pareto[X].max()
)

y_min = float(
    pareto[Y].min()
)

y_max = float(
    pareto[Y].max()
)


pareto[
    "water_objective_normalized"
] = (
    x_max
    - pareto[X]
) / (
    x_max
    - x_min
)


pareto[
    "H2_objective_normalized"
] = (
    pareto[Y]
    - y_min
) / (
    y_max
    - y_min
)


# ============================================================
# GEOMETRIC KNEE
#
# The line between the normalized extreme points is:
#
# H2_benefit = 1 - water_benefit
#
# or:
#
# water_benefit + H2_benefit - 1 = 0
#
# Perpendicular distance:
#
# |w + h - 1| / sqrt(2)
#
# A larger positive departure means the point achieves more
# of both objectives than a simple linear extreme-to-extreme
# compromise.
# ============================================================

pareto[
    "knee_distance"
] = (
    pareto[
        "water_objective_normalized"
    ]
    +
    pareto[
        "H2_objective_normalized"
    ]
    - 1.0
) / np.sqrt(2.0)


knee_idx = (
    pareto[
        "knee_distance"
    ]
    .idxmax()
)

knee = pareto.loc[
    knee_idx
]


# ============================================================
# LARGEST ADJACENT TRADE-OFF CLIFF
# ============================================================

steps = pareto.iloc[
    1:
].copy()


cliff_idx = (
    steps[
        "water_penalty_per_1pct_H2_gain"
    ]
    .idxmax()
)


cliff_destination = pareto.loc[
    cliff_idx
]

cliff_source = pareto.loc[
    cliff_idx - 1
]


# ============================================================
# CHECK AGREEMENT
# ============================================================

knee_cliff_agree = (
    str(
        knee["IL_ID"]
    )
    ==
    str(
        cliff_source["IL_ID"]
    )
)


# ============================================================
# SAVE METRICS
# ============================================================

output_cols = [

    "pareto_order",

    "IL_ID",

    "cation_abbreviation",

    "anion_abbreviation",

    "stoichiometry",

    "ln_gamma_inf_H2O",

    "kHcp_H2_mol_L_atm",

    "log10_kHcp_H2",

    "delta_ln_gamma_H2O",

    "delta_kHcp_H2",

    "percent_gain_kHcp_H2",

    "delta_log10_kHcp_H2",

    "tradeoff_efficiency",

    "water_penalty_per_1pct_H2_gain",

    "water_objective_normalized",

    "H2_objective_normalized",

    "knee_distance",

    "transition",

    "MOM2_A",

    "MOM3_A",

    "HBACC2_A",

    "HBDON2_A",

    "V_IL_A3",
]


pareto[
    output_cols
].to_csv(
    METRICS_FILE,
    index=False
)


# ============================================================
# FIGURE 1:
# FULL DATASET + PARETO + KNEE
# ============================================================

master[
    "log10_kHcp_H2"
] = np.log10(
    master[
        "kHcp_H2_mol_L_atm"
    ]
)


fig, ax = plt.subplots(
    figsize=(7.2, 5.6)
)


# All 4617 ILs
ax.scatter(

    master[
        "ln_gamma_inf_H2O"
    ],

    master[
        "log10_kHcp_H2"
    ],

    s=10,

    alpha=0.25,

    label="All ionic liquids",
)


# Pareto front
ax.plot(

    pareto[X],

    pareto[Y],

    marker="o",

    markersize=5,

    linewidth=1.4,

    label="Pareto front",
)


# Knee
ax.scatter(

    [knee[X]],

    [knee[Y]],

    marker="*",

    s=180,

    zorder=10,

    label="Geometric knee",
)


ax.annotate(

    str(
        knee[
            "IL_ID"
        ]
    ),

    (
        knee[X],
        knee[Y],
    ),

    xytext=(10, 8),

    textcoords="offset points",

    fontsize=10,
)


# Extremes
water_extreme = pareto.iloc[0]
h2_extreme = pareto.iloc[-1]


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

    xytext=(8, -15),

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

    xytext=(8, 7),

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
    "Pareto front and geometric knee"
)

ax.legend(
    frameon=False
)

fig.tight_layout()

if SAVE_FIGURES:
    fig.savefig(
        FIG1_PNG,
        dpi=600,
        bbox_inches="tight",
    )

if SAVE_FIGURES:
    fig.savefig(
        FIG1_PDF,
        bbox_inches="tight",
    )

plt.close(fig)


# ============================================================
# FIGURE 2:
# ADJACENT WATER PENALTY PER 1% H2 GAIN
# ============================================================

step_plot = pareto.iloc[
    1:
].copy()


fig, ax = plt.subplots(
    figsize=(8.0, 5.0)
)


xpos = np.arange(
    len(step_plot)
)


ax.bar(

    xpos,

    step_plot[
        "water_penalty_per_1pct_H2_gain"
    ],
)


ax.set_xticks(
    xpos
)


ax.set_xticklabels(

    step_plot[
        "transition"
    ],

    rotation=45,

    ha="right",
)


ax.set_ylabel(
    r"$\Delta\ln\gamma^\infty_{\mathrm{H_2O}}$ "
    "per 1% H$_2$ solubility gain"
)


ax.set_title(
    "Adjacent Pareto trade-off penalty"
)


# Label largest cliff
max_pos = int(
    np.argmax(
        step_plot[
            "water_penalty_per_1pct_H2_gain"
        ].to_numpy()
    )
)


max_row = step_plot.iloc[
    max_pos
]


ax.annotate(

    "Largest trade-off cliff",

    xy=(
        max_pos,
        max_row[
            "water_penalty_per_1pct_H2_gain"
        ],
    ),

    xytext=(
        max_pos,
        max_row[
            "water_penalty_per_1pct_H2_gain"
        ] * 0.75,
    ),

    ha="center",

    arrowprops={
        "arrowstyle": "->",
    },
)


fig.tight_layout()

if SAVE_FIGURES:
    fig.savefig(
        FIG2_PNG,
        dpi=600,
        bbox_inches="tight",
    )

if SAVE_FIGURES:
    fig.savefig(
        FIG2_PDF,
        bbox_inches="tight",
    )

plt.close(fig)


# ============================================================
# REPORT TABLE
# ============================================================

report_table_cols = [

    "pareto_order",

    "IL_ID",

    "cation_abbreviation",

    "anion_abbreviation",

    "ln_gamma_inf_H2O",

    "kHcp_H2_mol_L_atm",

    "percent_gain_kHcp_H2",

    "delta_ln_gamma_H2O",

    "tradeoff_efficiency",

    "water_penalty_per_1pct_H2_gain",

    "knee_distance",
]


report_table = pareto[
    report_table_cols
].copy()


# ============================================================
# REPORT
# ============================================================

report = f"""
============================================================
STAGE 10B - PARETO TRADE-OFF / KNEE ANALYSIS
============================================================

Pareto candidates                 : {len(pareto)}

------------------------------------------------------------
GEOMETRIC KNEE
------------------------------------------------------------

IL_ID:
{knee["IL_ID"]}

Cation:
{knee["cation_abbreviation"]}

Anion:
{knee["anion_abbreviation"]}

Stoichiometry:
{knee["stoichiometry"]}

ln_gamma_inf_H2O:
{knee["ln_gamma_inf_H2O"]}

kHcp_H2 [mol/L/atm]:
{knee["kHcp_H2_mol_L_atm"]}

log10_kHcp_H2:
{knee["log10_kHcp_H2"]}

Normalized water objective:
{knee["water_objective_normalized"]}

Normalized H2 objective:
{knee["H2_objective_normalized"]}

Geometric knee distance:
{knee["knee_distance"]}

------------------------------------------------------------
LARGEST ADJACENT TRADE-OFF CLIFF
------------------------------------------------------------

Transition:
{cliff_source["IL_ID"]} -> {cliff_destination["IL_ID"]}

Source candidate:
{cliff_source["cation_abbreviation"]} /
{cliff_source["anion_abbreviation"]}

Destination candidate:
{cliff_destination["cation_abbreviation"]} /
{cliff_destination["anion_abbreviation"]}

Loss in water objective:
+{cliff_destination["delta_ln_gamma_H2O"]}

Relative H2-solubility gain (%):
{cliff_destination["percent_gain_kHcp_H2"]}

Water penalty per 1% H2 gain:
{cliff_destination["water_penalty_per_1pct_H2_gain"]}

------------------------------------------------------------
AGREEMENT
------------------------------------------------------------

Geometric knee candidate equals the point immediately
before the largest adjacent trade-off cliff:

{knee_cliff_agree}

------------------------------------------------------------
FULL PARETO TRADE-OFF TABLE
------------------------------------------------------------

{report_table.to_string(index=False)}

------------------------------------------------------------
OUTPUTS
------------------------------------------------------------

Trade-off metrics:
{METRICS_FILE}

Pareto + knee figure:
{FIG1_PNG}
{FIG1_PDF}

Adjacent trade-off figure:
{FIG2_PNG}
{FIG2_PDF}

------------------------------------------------------------
INTERPRETATION RULE
------------------------------------------------------------

The geometric knee is NOT a weighted ranking score.

It is the Pareto point with the largest perpendicular
departure from the straight line connecting the two
objective extremes after min-max normalization.

The adjacent-cliff analysis independently identifies where
additional H2-solubility improvement begins to require a
disproportionately large loss in the H2O objective.

Agreement between these diagnostics strengthens the case
for a balanced Pareto region but does not convert the
result into an HER activity ranking.

============================================================
"""


REPORT_FILE.write_text(
    report
)

print(report)

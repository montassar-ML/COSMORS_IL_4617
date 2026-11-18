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

TRADEOFF_FILE = (
    ROOT
    / "10_pareto"
    / "pareto_tradeoff_metrics.csv"
)

CLUSTER_FILE = (
    ROOT
    / "11_PCA_clustering"
    / "Pareto_cluster_membership.csv"
)

ALL_CLUSTER_FILE = (
    ROOT
    / "11_PCA_clustering"
    / "PCA_clusters_all_4617.csv"
)

OUT = (
    ROOT
    / "12_shortlist"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PARETO9_FILE = (
    OUT
    / "final_Pareto_9.csv"
)

CORE5_FILE = (
    OUT
    / "core_shortlist_5.csv"
)

SECONDARY_FILE = (
    OUT
    / "secondary_Pareto_4.csv"
)

REPORT_FILE = (
    OUT
    / "stage12a_shortlist_report.txt"
)

FIG_PNG = (
    OUT
    / "final_shortlist_Pareto_PCA.png"
)

FIG_PDF = (
    OUT
    / "final_shortlist_Pareto_PCA.pdf"
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

    "legend.fontsize": 9,

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
    TRADEOFF_FILE
)

cluster = pd.read_csv(
    CLUSTER_FILE
)

all_clusters = pd.read_csv(
    ALL_CLUSTER_FILE
)


if len(pareto) != 9:
    raise RuntimeError(
        f"Expected 9 Pareto candidates, found {len(pareto)}"
    )


# ============================================================
# MERGE ALL INFORMATION FOR THE 9 PARETO CANDIDATES
# ============================================================

# Keep clustering columns
cluster_keep = cluster[
    [
        "IL_ID",
        "cluster",
        "PC1",
        "PC2",
        "PC3",
    ]
].copy()


# Remove PCA columns from pareto table first if already present
for col in [
    "cluster",
    "PC1",
    "PC2",
    "PC3",
]:

    if col in pareto.columns:
        pareto = pareto.drop(
            columns=[col]
        )


p = (
    pareto
    .merge(
        cluster_keep,
        on="IL_ID",
        how="left",
        validate="one_to_one",
    )
)


# Add names and thermodynamic properties
master_keep = master[
    [
        "IL_ID",

        "cation_id",
        "cation_name",
        "cation_abbreviation",

        "anion_id",
        "anion_name",
        "anion_abbreviation",

        "stoichiometry",

        "gamma_inf_H2O",
        "ln_gamma_inf_H2O",

        "gamma_inf_H2",
        "ln_gamma_inf_H2",

        "kHcp_H2_mol_L_atm",

        "A_IL_A2",
        "V_IL_A3",

        "MOM2_A",
        "MOM3_A",

        "HBACC2_A",
        "HBDON2_A",
    ]
]


# Avoid duplicate columns already present
drop_from_master = [
    c for c in master_keep.columns
    if c != "IL_ID"
    and c in p.columns
]

master_add = master_keep.drop(
    columns=drop_from_master
)


p = p.merge(
    master_add,
    on="IL_ID",
    how="left",
    validate="one_to_one",
)


# ============================================================
# SELECTION ROLE 1:
# WATER-AFFINITY EXTREME
# ============================================================

water_extreme_idx = (
    p[
        "ln_gamma_inf_H2O"
    ].idxmin()
)

water_extreme = p.loc[
    water_extreme_idx
]


# ============================================================
# SELECTION ROLE 2:
# H2-SOLUBILITY EXTREME
# ============================================================

h2_extreme_idx = (
    p[
        "kHcp_H2_mol_L_atm"
    ].idxmax()
)

h2_extreme = p.loc[
    h2_extreme_idx
]


# ============================================================
# SELECTION ROLE 3:
# GEOMETRIC KNEE
# ============================================================

knee_idx = (
    p[
        "knee_distance"
    ].idxmax()
)

knee = p.loc[
    knee_idx
]


# ============================================================
# SELECTION ROLE 4:
# HIGHEST ADJACENT TRADE-OFF EFFICIENCY
#
# Exclude the two extremes and the knee.
# ============================================================

mandatory_ids = {
    water_extreme["IL_ID"],
    h2_extreme["IL_ID"],
    knee["IL_ID"],
}


eff_candidates = p[
    ~p["IL_ID"].isin(
        mandatory_ids
    )
].copy()


eff_candidates = eff_candidates[
    np.isfinite(
        eff_candidates[
            "tradeoff_efficiency"
        ]
    )
]


if len(eff_candidates) == 0:

    raise RuntimeError(
        "No candidate available for trade-off-efficiency role."
    )


eff_idx = (
    eff_candidates[
        "tradeoff_efficiency"
    ].idxmax()
)

eff_candidate = p.loc[
    eff_idx
]


selected_ids = {
    water_extreme["IL_ID"],
    h2_extreme["IL_ID"],
    knee["IL_ID"],
    eff_candidate["IL_ID"],
}


# ============================================================
# SELECTION ROLE 5:
# CHEMICALLY DIVERSE 1:1 REPRESENTATIVE
#
# Preference:
# 1. same cluster as Pareto knee
# 2. 1:1 stoichiometry
# 3. maximize introduction of new cation/anion identities
# 4. maximize minimum PC1-PC3 distance from already selected
# ============================================================

knee_cluster = int(
    knee["cluster"]
)


remaining = p[
    ~p["IL_ID"].isin(
        selected_ids
    )
].copy()


diverse_pool = remaining[
    (
        remaining[
            "cluster"
        ]
        == knee_cluster
    )
    &
    (
        remaining[
            "stoichiometry"
        ]
        == "1:1"
    )
].copy()


# Fallback if constrained pool is empty
if len(diverse_pool) == 0:

    diverse_pool = remaining.copy()


selected_rows = p[
    p["IL_ID"].isin(
        selected_ids
    )
].copy()


existing_cations = set(
    selected_rows[
        "cation_id"
    ]
)

existing_anions = set(
    selected_rows[
        "anion_id"
    ]
)


def novelty_score(row):

    new_cation = int(
        row["cation_id"]
        not in existing_cations
    )

    new_anion = int(
        row["anion_id"]
        not in existing_anions
    )

    return (
        new_cation
        + new_anion
    )


def min_pc_distance(row):

    q = np.array(
        [
            row["PC1"],
            row["PC2"],
            row["PC3"],
        ],
        dtype=float,
    )

    distances = []

    for _, r in selected_rows.iterrows():

        z = np.array(
            [
                r["PC1"],
                r["PC2"],
                r["PC3"],
            ],
            dtype=float,
        )

        distances.append(
            np.linalg.norm(
                q - z
            )
        )

    return float(
        min(distances)
    )


diverse_pool[
    "novelty_score"
] = diverse_pool.apply(
    novelty_score,
    axis=1,
)

diverse_pool[
    "minimum_PC123_distance"
] = diverse_pool.apply(
    min_pc_distance,
    axis=1,
)


diverse_pool = diverse_pool.sort_values(
    [
        "novelty_score",
        "minimum_PC123_distance",
    ],
    ascending=[
        False,
        False,
    ],
)


diverse_candidate = (
    diverse_pool.iloc[0]
)


selected_ids.add(
    diverse_candidate["IL_ID"]
)


# ============================================================
# ASSIGN ROLES
# ============================================================

roles = {

    water_extreme["IL_ID"]:
        "Water-affinity extreme",

    eff_candidate["IL_ID"]:
        "Highest Pareto step efficiency",

    knee["IL_ID"]:
        "Geometric Pareto knee",

    diverse_candidate["IL_ID"]:
        "Descriptor-diverse 1:1 representative",

    h2_extreme["IL_ID"]:
        "H2-solubility extreme",
}


p[
    "core_shortlist"
] = p[
    "IL_ID"
].isin(
    selected_ids
)


p[
    "selection_role"
] = p[
    "IL_ID"
].map(
    roles
).fillna(
    "Secondary Pareto candidate"
)


# ============================================================
# ORDER TABLES
# ============================================================

p = p.sort_values(
    "pareto_order"
).reset_index(
    drop=True
)


core = p[
    p[
        "core_shortlist"
    ]
].copy()


secondary = p[
    ~p[
        "core_shortlist"
    ]
].copy()


# Preserve Pareto order
core = core.sort_values(
    "pareto_order"
)

secondary = secondary.sort_values(
    "pareto_order"
)


# ============================================================
# OUTPUT COLUMNS
# ============================================================

output_cols = [

    "pareto_order",

    "IL_ID",

    "selection_role",

    "core_shortlist",

    "cation_id",
    "cation_name",
    "cation_abbreviation",

    "anion_id",
    "anion_name",
    "anion_abbreviation",

    "stoichiometry",

    "cluster",

    "ln_gamma_inf_H2O",

    "gamma_inf_H2O",

    "kHcp_H2_mol_L_atm",

    "ln_gamma_inf_H2",

    "gamma_inf_H2",

    "MOM2_A",
    "MOM3_A",

    "HBACC2_A",
    "HBDON2_A",

    "V_IL_A3",

    "PC1",
    "PC2",
    "PC3",

    "knee_distance",

    "tradeoff_efficiency",

    "water_penalty_per_1pct_H2_gain",
]


p[
    output_cols
].to_csv(
    PARETO9_FILE,
    index=False,
)


core[
    output_cols
].to_csv(
    CORE5_FILE,
    index=False,
)


secondary[
    output_cols
].to_csv(
    SECONDARY_FILE,
    index=False,
)


# ============================================================
# DIVERSITY SUMMARY
# ============================================================

core_unique_cations = (
    core[
        "cation_id"
    ].nunique()
)

core_unique_anions = (
    core[
        "anion_id"
    ].nunique()
)

core_clusters = (
    core[
        "cluster"
    ].nunique()
)

core_11 = int(
    (
        core[
            "stoichiometry"
        ]
        == "1:1"
    ).sum()
)

core_21 = int(
    (
        core[
            "stoichiometry"
        ]
        == "2:1"
    ).sum()
)


# ============================================================
# FIGURE
# TWO-PANEL SUMMARY
# ============================================================

master_plot = master.copy()

master_plot[
    "log10_kHcp_H2"
] = np.log10(
    master_plot[
        "kHcp_H2_mol_L_atm"
    ]
)

p[
    "log10_kHcp_H2"
] = np.log10(
    p[
        "kHcp_H2_mol_L_atm"
    ]
)

core[
    "log10_kHcp_H2"
] = np.log10(
    core[
        "kHcp_H2_mol_L_atm"
    ]
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.0, 5.6),
)


# ============================================================
# PANEL A — PARETO PERFORMANCE
# ============================================================

ax = axes[0]


ax.scatter(

    master_plot[
        "ln_gamma_inf_H2O"
    ],

    master_plot[
        "log10_kHcp_H2"
    ],

    s=9,

    alpha=0.18,

    label="All ionic liquids",
)


ax.plot(

    p[
        "ln_gamma_inf_H2O"
    ],

    p[
        "log10_kHcp_H2"
    ],

    marker="o",

    markersize=4,

    linewidth=1.2,

    label="Pareto front",
)


ax.scatter(

    core[
        "ln_gamma_inf_H2O"
    ],

    core[
        "log10_kHcp_H2"
    ],

    s=65,

    zorder=6,

    label="Core shortlist",
)


for _, row in core.iterrows():

    ax.annotate(

        row[
            "IL_ID"
        ],

        (
            row[
                "ln_gamma_inf_H2O"
            ],

            row[
                "log10_kHcp_H2"
            ],
        ),

        xytext=(
            6,
            5,
        ),

        textcoords="offset points",

        fontsize=8,
    )


ax.set_xlabel(
    r"$\ln\gamma^\infty_{\mathrm{H_2O}}$"
)

ax.set_ylabel(
    r"$\log_{10}k_H^{cp}(\mathrm{H_2})$"
)

ax.set_title(
    "(a) Thermodynamic Pareto performance"
)

ax.legend(
    frameon=False
)


# ============================================================
# PANEL B — PCA / CLUSTER DIVERSITY
# ============================================================

ax = axes[1]


for cl in sorted(
    all_clusters[
        "cluster"
    ].unique()
):

    temp = all_clusters[
        all_clusters[
            "cluster"
        ]
        == cl
    ]

    ax.scatter(

        temp[
            "PC1"
        ],

        temp[
            "PC2"
        ],

        s=8,

        alpha=0.20,

        label=f"Cluster {cl}",
    )


# All Pareto candidates
ax.scatter(

    p[
        "PC1"
    ],

    p[
        "PC2"
    ],

    s=55,

    facecolors="none",

    linewidths=1.2,

    label="Pareto candidates",
)


# Core shortlist
ax.scatter(

    core[
        "PC1"
    ],

    core[
        "PC2"
    ],

    s=70,

    zorder=7,

    label="Core shortlist",
)


# Knee star
knee_plot = core[
    core[
        "IL_ID"
    ]
    == knee[
        "IL_ID"
    ]
].iloc[0]


ax.scatter(

    [
        knee_plot[
            "PC1"
        ]
    ],

    [
        knee_plot[
            "PC2"
        ]
    ],

    marker="*",

    s=180,

    zorder=10,

    label="Pareto knee",
)


for _, row in core.iterrows():

    ax.annotate(

        row[
            "IL_ID"
        ],

        (
            row[
                "PC1"
            ],

            row[
                "PC2"
            ],
        ),

        xytext=(
            5,
            5,
        ),

        textcoords="offset points",

        fontsize=8,
    )


ax.set_xlabel(
    "PC1 (53.7% variance)"
)

ax.set_ylabel(
    "PC2 (29.3% variance)"
)

ax.set_title(
    "(b) Descriptor-space diversity"
)

ax.legend(
    frameon=False,
    fontsize=8,
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

plt.close(fig)


# ============================================================
# REPORT
# ============================================================

report_cols = [

    "pareto_order",

    "IL_ID",

    "cation_abbreviation",

    "anion_abbreviation",

    "stoichiometry",

    "cluster",

    "ln_gamma_inf_H2O",

    "kHcp_H2_mol_L_atm",

    "selection_role",
]


report = f"""
============================================================
STAGE 12A - FINAL PARETO / CORE SHORTLIST
============================================================

Validated starting dataset        : {len(master)} ILs
Formal Pareto-optimal set          : {len(p)} ILs
Core downstream shortlist          : {len(core)} ILs
Secondary Pareto candidates        : {len(secondary)} ILs

------------------------------------------------------------
SELECTION PRINCIPLE
------------------------------------------------------------

All 9 non-dominated ILs remain part of the formal Pareto
set.

The 5-member core shortlist is a representative subset for
downstream expensive calculations or experimental
prioritization.

No weighted global ranking score was used.

The five roles are:

1. strongest H2O-affinity Pareto extreme;
2. highest adjacent Pareto trade-off efficiency;
3. geometric Pareto knee;
4. descriptor-diverse 1:1 candidate from the knee-side
   descriptor cluster;
5. strongest H2-solubility Pareto extreme.

------------------------------------------------------------
CORE SHORTLIST
------------------------------------------------------------

{core[
    report_cols
].to_string(index=False)}

------------------------------------------------------------
CORE DIVERSITY
------------------------------------------------------------

Unique cations                    : {core_unique_cations}
Unique anions                     : {core_unique_anions}
Descriptor-space clusters         : {core_clusters}

1:1 systems                       : {core_11}
2:1 systems                       : {core_21}

------------------------------------------------------------
SECONDARY PARETO CANDIDATES
------------------------------------------------------------

{secondary[
    report_cols
].to_string(index=False)}

------------------------------------------------------------
SPECIAL CANDIDATE
------------------------------------------------------------

Pareto knee:

{knee["IL_ID"]}

{knee["cation_abbreviation"]} / {knee["anion_abbreviation"]}

ln_gamma_inf_H2O:
{knee["ln_gamma_inf_H2O"]}

kHcp_H2 [mol/L/atm]:
{knee["kHcp_H2_mol_L_atm"]}

This candidate is the balanced knee of the two-objective
thermodynamic Pareto front. It is not an HER activity
ranking.

------------------------------------------------------------
OUTPUTS
------------------------------------------------------------

Full formal Pareto set:
{PARETO9_FILE}

Core shortlist:
{CORE5_FILE}

Secondary Pareto set:
{SECONDARY_FILE}

Summary figure:
{FIG_PNG}
{FIG_PDF}

============================================================
"""


REPORT_FILE.write_text(
    report
)

print(
    report
)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Repository default: numerical outputs only.
SAVE_FIGURES = False

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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

OUT = (
    ROOT
    / "11_PCA_clustering"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SCORES_FILE = (
    OUT
    / "PCA_scores_all_4617.csv"
)

LOADINGS_FILE = (
    OUT
    / "PCA_loadings.csv"
)

VARIANCE_FILE = (
    OUT
    / "PCA_explained_variance.csv"
)

PARETO_PCA_FILE = (
    OUT
    / "PCA_pareto_candidates.csv"
)

REPORT_FILE = (
    OUT
    / "stage11a_PCA_report.txt"
)

FIG1_PNG = (
    OUT
    / "PCA_PC1_PC2_all_with_Pareto.png"
)

FIG1_PDF = (
    OUT
    / "PCA_PC1_PC2_all_with_Pareto.pdf"
)

FIG2_PNG = (
    OUT
    / "PCA_loadings_PC1_PC2.png"
)

FIG2_PDF = (
    OUT
    / "PCA_loadings_PC1_PC2.pdf"
)

FIG3_PNG = (
    OUT
    / "PCA_explained_variance.png"
)

FIG3_PDF = (
    OUT
    / "PCA_explained_variance.pdf"
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

pareto = pd.read_csv(
    PARETO_FILE
)


if len(df) != 4617:
    raise RuntimeError(
        f"Expected 4617 ILs, found {len(df)}"
    )


# ============================================================
# PCA FEATURES
#
# Intrinsic descriptor space only.
# No H2O/H2 thermodynamic target is included here.
# ============================================================

features = [

    "V_IL_A3",

    "MOM2_A",

    "MOM3_A",

    "HBACC2_A",

    "HBDON2_A",
]


feature_labels = {

    "V_IL_A3":
        r"$V_{\mathrm{IL}}$",

    "MOM2_A":
        r"$\mathrm{MOM2}/A$",

    "MOM3_A":
        r"$\mathrm{MOM3}/A$",

    "HBACC2_A":
        r"$\mathrm{HBACC2}/A$",

    "HBDON2_A":
        r"$\mathrm{HBDON2}/A$",
}


# ============================================================
# VALIDATE
# ============================================================

X = df[
    features
].copy()

if X.isna().any().any():
    raise RuntimeError(
        "Missing PCA descriptor values detected."
    )

if not np.isfinite(
    X.to_numpy()
).all():

    raise RuntimeError(
        "Non-finite PCA descriptor values detected."
    )


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# ============================================================
# PCA
# ============================================================

pca = PCA()

scores = pca.fit_transform(
    X_scaled
)

explained = (
    pca.explained_variance_ratio_
)

cumulative = np.cumsum(
    explained
)


# ============================================================
# SCORE TABLE
# ============================================================

score_df = df[
    [
        "IL_ID",

        "cation_id",
        "cation_abbreviation",

        "anion_id",
        "anion_abbreviation",

        "stoichiometry",

        "ln_gamma_inf_H2O",

        "ln_gamma_inf_H2",

        "kHcp_H2_mol_L_atm",

        "master_status",
    ]
].copy()


for i in range(
    scores.shape[1]
):

    score_df[
        f"PC{i+1}"
    ] = scores[:, i]


score_df.to_csv(
    SCORES_FILE,
    index=False,
)


# ============================================================
# LOADINGS
#
# component coefficients for standardized variables
# ============================================================

loadings = pd.DataFrame(

    pca.components_.T,

    index=features,

    columns=[
        f"PC{i+1}"
        for i in range(
            len(features)
        )
    ],
)

loadings.index.name = (
    "descriptor"
)

loadings.to_csv(
    LOADINGS_FILE
)


# ============================================================
# EXPLAINED VARIANCE
# ============================================================

variance = pd.DataFrame({

    "PC": [
        f"PC{i+1}"
        for i in range(
            len(features)
        )
    ],

    "explained_variance_ratio":
        explained,

    "explained_variance_percent":
        100.0 * explained,

    "cumulative_variance_percent":
        100.0 * cumulative,
})

variance.to_csv(
    VARIANCE_FILE,
    index=False,
)


# ============================================================
# PARETO OVERLAY TABLE
# ============================================================

pareto_ids = set(
    pareto["IL_ID"]
)

pareto_scores = (
    score_df[
        score_df["IL_ID"].isin(
            pareto_ids
        )
    ]
    .merge(
        pareto[
            [
                "IL_ID",
                "pareto_order",
            ]
        ],
        on="IL_ID",
        how="left",
    )
    .sort_values(
        "pareto_order"
    )
    .reset_index(
        drop=True
    )
)


pareto_scores.to_csv(
    PARETO_PCA_FILE,
    index=False,
)


# ============================================================
# KNEE
# ============================================================

KNEE_ID = "IL0114"

if KNEE_ID not in set(
    pareto_scores["IL_ID"]
):

    raise RuntimeError(
        "IL0114 not found in Pareto PCA table."
    )


knee = pareto_scores[
    pareto_scores[
        "IL_ID"
    ]
    == KNEE_ID
].iloc[0]


# ============================================================
# FIGURE 1
# PC1 vs PC2
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.3, 5.8)
)


# All ILs
ax.scatter(

    score_df["PC1"],

    score_df["PC2"],

    s=10,

    alpha=0.25,

    label="All ionic liquids",
)


# Pareto candidates
ax.scatter(

    pareto_scores["PC1"],

    pareto_scores["PC2"],

    s=65,

    facecolors="none",

    linewidths=1.5,

    label="Pareto candidates",
)


# Knee
ax.scatter(

    [knee["PC1"]],

    [knee["PC2"]],

    marker="*",

    s=190,

    zorder=10,

    label="Pareto knee",
)


# Label all 9 Pareto candidates
for _, row in pareto_scores.iterrows():

    ax.annotate(

        row["IL_ID"],

        (
            row["PC1"],
            row["PC2"],
        ),

        xytext=(
            6,
            5,
        ),

        textcoords="offset points",

        fontsize=8,
    )


pc1_pct = (
    variance.loc[
        0,
        "explained_variance_percent"
    ]
)

pc2_pct = (
    variance.loc[
        1,
        "explained_variance_percent"
    ]
)


ax.set_xlabel(
    f"PC1 ({pc1_pct:.1f}% variance)"
)

ax.set_ylabel(
    f"PC2 ({pc2_pct:.1f}% variance)"
)

ax.set_title(
    "PCA of intrinsic ionic-liquid descriptor space"
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
# FIGURE 2
# LOADINGS PC1 / PC2
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.5, 6.0)
)


ax.axhline(
    0,
    linewidth=0.8,
)

ax.axvline(
    0,
    linewidth=0.8,
)


for feature in features:

    x = loadings.loc[
        feature,
        "PC1"
    ]

    y = loadings.loc[
        feature,
        "PC2"
    ]


    ax.arrow(

        0,
        0,

        x,
        y,

        length_includes_head=True,

        head_width=0.025,

        head_length=0.035,

        linewidth=1.2,
    )


    ax.text(

        x * 1.10,

        y * 1.10,

        feature_labels[
            feature
        ],

        ha="center",

        va="center",

        fontsize=11,
    )


max_loading = float(
    np.max(
        np.abs(
            loadings[
                [
                    "PC1",
                    "PC2",
                ]
            ].to_numpy()
        )
    )
)

lim = max(
    0.8,
    max_loading * 1.35,
)


ax.set_xlim(
    -lim,
    lim,
)

ax.set_ylim(
    -lim,
    lim,
)


ax.set_xlabel(
    f"PC1 loading ({pc1_pct:.1f}%)"
)

ax.set_ylabel(
    f"PC2 loading ({pc2_pct:.1f}%)"
)

ax.set_title(
    "PCA descriptor loadings"
)

ax.set_aspect(
    "equal",
    adjustable="box",
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
# FIGURE 3
# EXPLAINED VARIANCE
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.5, 4.8)
)


x = np.arange(
    1,
    len(features) + 1
)


ax.bar(

    x,

    100.0 * explained,

    label="Individual",
)


ax.plot(

    x,

    100.0 * cumulative,

    marker="o",

    label="Cumulative",
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        f"PC{i}"
        for i in x
    ]
)

ax.set_ylabel(
    "Explained variance (%)"
)

ax.set_ylim(
    0,
    105,
)

ax.set_title(
    "PCA explained variance"
)

ax.legend(
    frameon=False
)

fig.tight_layout()

if SAVE_FIGURES:
    fig.savefig(
        FIG3_PNG,
        dpi=600,
        bbox_inches="tight",
    )

if SAVE_FIGURES:
    fig.savefig(
        FIG3_PDF,
        bbox_inches="tight",
    )

plt.close(fig)


# ============================================================
# TOP LOADINGS
# ============================================================

loading_summary = []

for pc in [
    "PC1",
    "PC2",
    "PC3",
]:

    tmp = (
        loadings[
            [pc]
        ]
        .copy()
    )

    tmp[
        "abs_loading"
    ] = (
        tmp[pc]
        .abs()
    )

    tmp = (
        tmp
        .sort_values(
            "abs_loading",
            ascending=False,
        )
        .reset_index()
    )

    top = tmp.head(
        5
    )

    for rank, row in top.iterrows():

        loading_summary.append({

            "PC":
                pc,

            "rank":
                rank + 1,

            "descriptor":
                row[
                    "descriptor"
                ],

            "loading":
                row[
                    pc
                ],

            "abs_loading":
                row[
                    "abs_loading"
                ],
        })


loading_summary = pd.DataFrame(
    loading_summary
)


# ============================================================
# REPORT
# ============================================================

report = f"""
============================================================
STAGE 11A - PCA CHEMICAL-SPACE ANALYSIS
============================================================

Dataset                          : {len(df)} ILs

------------------------------------------------------------
PCA DESIGN
------------------------------------------------------------

Input variables:

{chr(10).join(features)}

All variables were standardized to:
mean = 0
standard deviation = 1

Thermodynamic screening targets were NOT used as PCA
input variables.

A_IL_A2 was excluded to avoid duplicating the primary
molecular-size information represented by V_IL_A3.

------------------------------------------------------------
EXPLAINED VARIANCE
------------------------------------------------------------

{variance.to_string(index=False)}

PC1 + PC2 cumulative variance (%):
{100.0 * cumulative[1]}

PC1 + PC2 + PC3 cumulative variance (%):
{100.0 * cumulative[2]}

------------------------------------------------------------
TOP LOADINGS
------------------------------------------------------------

{loading_summary.to_string(index=False)}

------------------------------------------------------------
PARETO CANDIDATES IN PCA SPACE
------------------------------------------------------------

{pareto_scores[
    [
        "pareto_order",
        "IL_ID",
        "cation_abbreviation",
        "anion_abbreviation",
        "stoichiometry",
        "PC1",
        "PC2",
        "PC3",
    ]
].to_string(index=False)}

------------------------------------------------------------
PARETO KNEE
------------------------------------------------------------

IL_ID:
{knee["IL_ID"]}

Cation:
{knee["cation_abbreviation"]}

Anion:
{knee["anion_abbreviation"]}

PC1:
{knee["PC1"]}

PC2:
{knee["PC2"]}

PC3:
{knee["PC3"]}

------------------------------------------------------------
OUTPUTS
------------------------------------------------------------

PCA scores:
{SCORES_FILE}

PCA loadings:
{LOADINGS_FILE}

Explained variance:
{VARIANCE_FILE}

Pareto candidates in PCA space:
{PARETO_PCA_FILE}

PC1/PC2 figure:
{FIG1_PNG}
{FIG1_PDF}

Loadings figure:
{FIG2_PNG}
{FIG2_PDF}

Explained-variance figure:
{FIG3_PNG}
{FIG3_PDF}

------------------------------------------------------------
IMPORTANT
------------------------------------------------------------

PCA is used here to visualize intrinsic descriptor-space
structure and chemical diversity.

It is NOT used to rank ionic liquids.

The Pareto thermodynamic result remains independent of
the PCA transformation.

============================================================
"""


REPORT_FILE.write_text(
    report
)

print(
    report
)

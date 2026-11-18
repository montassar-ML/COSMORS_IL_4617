from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Repository default: numerical outputs only.
SAVE_FIGURES = False

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SCORES_FILE = (
    ROOT
    / "11_PCA_clustering"
    / "PCA_scores_all_4617.csv"
)

PARETO_FILE = (
    ROOT
    / "11_PCA_clustering"
    / "PCA_pareto_candidates.csv"
)

MASTER_FILE = (
    ROOT
    / "07_master_table"
    / "IL_master_table_4617.csv"
)

OUT = (
    ROOT
    / "11_PCA_clustering"
)

METRICS_FILE = (
    OUT
    / "clustering_model_selection.csv"
)

CLUSTERED_FILE = (
    OUT
    / "PCA_clusters_all_4617.csv"
)

PARETO_CLUSTER_FILE = (
    OUT
    / "Pareto_cluster_membership.csv"
)

PROFILE_FILE = (
    OUT
    / "cluster_descriptor_profiles.csv"
)

REPORT_FILE = (
    OUT
    / "stage11b_clustering_report.txt"
)

FIG_METRICS = (
    OUT
    / "cluster_model_selection.png"
)

FIG_CLUSTERS = (
    OUT
    / "PCA_clusters_PC1_PC2.png"
)

FIG_PARETO = (
    OUT
    / "PCA_clusters_with_Pareto.png"
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

scores = pd.read_csv(
    SCORES_FILE
)

pareto = pd.read_csv(
    PARETO_FILE
)

master = pd.read_csv(
    MASTER_FILE
)


if len(scores) != 4617:
    raise RuntimeError(
        f"Expected 4617 PCA rows, found {len(scores)}"
    )


# ============================================================
# CLUSTERING SPACE
#
# PC1 + PC2 + PC3 = 94.85% variance
# ============================================================

pc_cols = [
    "PC1",
    "PC2",
    "PC3",
]

X = scores[
    pc_cols
].to_numpy()


# ============================================================
# TEST K = 2 ... 10
# ============================================================

metric_rows = []

models = {}


for k in range(
    2,
    11,
):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=50,
    )

    labels = model.fit_predict(
        X
    )

    sil = silhouette_score(
        X,
        labels,
    )

    ch = calinski_harabasz_score(
        X,
        labels,
    )

    db = davies_bouldin_score(
        X,
        labels,
    )

    inertia = model.inertia_

    metric_rows.append({
        "k": k,
        "silhouette_score": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "inertia": inertia,
    })

    models[k] = model


metrics = pd.DataFrame(
    metric_rows
)

metrics.to_csv(
    METRICS_FILE,
    index=False,
)


# ============================================================
# PRIMARY K SELECTION
#
# Use maximum silhouette score.
# Other metrics retained for independent inspection.
# ============================================================

best_k = int(
    metrics.loc[
        metrics[
            "silhouette_score"
        ].idxmax(),
        "k"
    ]
)

best_model = models[
    best_k
]

cluster_labels = (
    best_model.labels_
)


# Convert zero-based sklearn labels to 1-based labels
scores[
    "cluster"
] = (
    cluster_labels
    + 1
)


scores.to_csv(
    CLUSTERED_FILE,
    index=False,
)


# ============================================================
# CLUSTER CENTERS IN PCA SPACE
# ============================================================

centers = pd.DataFrame(
    best_model.cluster_centers_,
    columns=pc_cols,
)

centers[
    "cluster"
] = (
    np.arange(
        best_k
    )
    + 1
)


# ============================================================
# JOIN CLUSTERS WITH MASTER DESCRIPTORS
# ============================================================

descriptor_cols = [

    "IL_ID",

    "V_IL_A3",

    "MOM2_A",

    "MOM3_A",

    "HBACC2_A",

    "HBDON2_A",

    "ln_gamma_inf_H2O",

    "ln_gamma_inf_H2",

    "kHcp_H2_mol_L_atm",
]


joined = (
    scores[
        [
            "IL_ID",
            "cluster",
        ]
    ]
    .merge(
        master[
            descriptor_cols
        ],
        on="IL_ID",
        how="left",
        validate="one_to_one",
    )
)


# ============================================================
# CLUSTER PROFILES
# ============================================================

profile_features = [

    "V_IL_A3",

    "MOM2_A",

    "MOM3_A",

    "HBACC2_A",

    "HBDON2_A",

    "ln_gamma_inf_H2O",

    "ln_gamma_inf_H2",

    "kHcp_H2_mol_L_atm",
]


profile = (
    joined
    .groupby(
        "cluster"
    )[profile_features]
    .agg(
        [
            "mean",
            "median",
            "std",
        ]
    )
)


profile.columns = [
    "_".join(col)
    for col in profile.columns
]

profile = profile.reset_index()


counts = (
    joined[
        "cluster"
    ]
    .value_counts()
    .sort_index()
)


profile.insert(
    1,
    "n_ILs",
    profile[
        "cluster"
    ].map(
        counts
    ),
)


profile.to_csv(
    PROFILE_FILE,
    index=False,
)


# ============================================================
# PARETO MEMBERSHIP
# ============================================================

pareto_cluster = (
    pareto
    .merge(
        scores[
            [
                "IL_ID",
                "cluster",
                "PC1",
                "PC2",
                "PC3",
            ]
        ],
        on="IL_ID",
        how="left",
        suffixes=(
            "",
            "_clustered",
        ),
    )
)


# Avoid duplicated PCA columns if already present
for pc in [
    "PC1",
    "PC2",
    "PC3",
]:

    alternate = (
        pc
        + "_clustered"
    )

    if alternate in pareto_cluster.columns:

        pareto_cluster[
            pc
        ] = pareto_cluster[
            alternate
        ]

        pareto_cluster.drop(
            columns=[
                alternate
            ],
            inplace=True,
        )


pareto_cluster.to_csv(
    PARETO_CLUSTER_FILE,
    index=False,
)


# ============================================================
# PARETO CLUSTER COUNTS
# ============================================================

pareto_cluster_counts = (
    pareto_cluster[
        "cluster"
    ]
    .value_counts()
    .sort_index()
    .rename_axis(
        "cluster"
    )
    .reset_index(
        name="Pareto_count"
    )
)


# ============================================================
# PAIRWISE DISTANCES BETWEEN PARETO POINTS
# IN PC1-PC3 SPACE
# ============================================================

distance_rows = []

for i in range(
    len(pareto_cluster)
):

    for j in range(
        i + 1,
        len(pareto_cluster),
    ):

        r1 = pareto_cluster.iloc[
            i
        ]

        r2 = pareto_cluster.iloc[
            j
        ]

        p1 = np.array([
            r1["PC1"],
            r1["PC2"],
            r1["PC3"],
        ])

        p2 = np.array([
            r2["PC1"],
            r2["PC2"],
            r2["PC3"],
        ])

        d = float(
            np.linalg.norm(
                p1 - p2
            )
        )

        distance_rows.append({
            "IL1": r1["IL_ID"],
            "IL2": r2["IL_ID"],
            "distance_PC123": d,
        })


distances = pd.DataFrame(
    distance_rows
)


closest_pair = (
    distances.loc[
        distances[
            "distance_PC123"
        ].idxmin()
    ]
)

most_distant_pair = (
    distances.loc[
        distances[
            "distance_PC123"
        ].idxmax()
    ]
)


# ============================================================
# FIGURE 1:
# MODEL SELECTION
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.6, 4.8)
)

ax.plot(
    metrics["k"],
    metrics[
        "silhouette_score"
    ],
    marker="o",
)

ax.axvline(
    best_k,
    linestyle="--",
    linewidth=1.0,
)

ax.set_xlabel(
    "Number of clusters, k"
)

ax.set_ylabel(
    "Silhouette score"
)

ax.set_title(
    "K-means cluster-number selection"
)

fig.tight_layout()

if SAVE_FIGURES:
    fig.savefig(
        FIG_METRICS,
        dpi=600,
        bbox_inches="tight",
    )

plt.close(fig)


# ============================================================
# FIGURE 2:
# ALL CLUSTERS
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.2, 5.8)
)


for cluster in sorted(
    scores[
        "cluster"
    ].unique()
):

    x = scores[
        scores[
            "cluster"
        ]
        == cluster
    ]

    ax.scatter(
        x["PC1"],
        x["PC2"],
        s=10,
        alpha=0.35,
        label=f"Cluster {cluster}",
    )


ax.set_xlabel(
    "PC1 (53.7% variance)"
)

ax.set_ylabel(
    "PC2 (29.3% variance)"
)

ax.set_title(
    f"K-means clustering in PCA space (k={best_k})"
)

ax.legend(
    frameon=False,
    markerscale=1.5,
)

fig.tight_layout()

if SAVE_FIGURES:
    fig.savefig(
        FIG_CLUSTERS,
        dpi=600,
        bbox_inches="tight",
    )

plt.close(fig)


# ============================================================
# FIGURE 3:
# CLUSTERS + PARETO
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.4, 5.9)
)


for cluster in sorted(
    scores[
        "cluster"
    ].unique()
):

    x = scores[
        scores[
            "cluster"
        ]
        == cluster
    ]

    ax.scatter(
        x["PC1"],
        x["PC2"],
        s=9,
        alpha=0.25,
        label=f"Cluster {cluster}",
    )


# Pareto candidates
ax.scatter(
    pareto_cluster["PC1"],
    pareto_cluster["PC2"],
    s=75,
    facecolors="none",
    linewidths=1.5,
    label="Pareto candidates",
)


# Knee
knee = pareto_cluster[
    pareto_cluster[
        "IL_ID"
    ]
    == "IL0114"
].iloc[0]


ax.scatter(
    [knee["PC1"]],
    [knee["PC2"]],
    marker="*",
    s=200,
    zorder=10,
    label="Pareto knee",
)


for _, row in pareto_cluster.iterrows():

    ax.annotate(
        row["IL_ID"],
        (
            row["PC1"],
            row["PC2"],
        ),
        xytext=(5, 5),
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
    "Pareto candidates across descriptor-space clusters"
)

ax.legend(
    frameon=False,
    markerscale=1.3,
)

fig.tight_layout()

if SAVE_FIGURES:
    fig.savefig(
        FIG_PARETO,
        dpi=600,
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

    "PC1",

    "PC2",

    "PC3",
]


report = f"""
============================================================
STAGE 11B - PCA-SPACE CLUSTERING
============================================================

Dataset                          : {len(scores)} ILs

Clustering variables:
PC1
PC2
PC3

Variance retained by PC1-PC3:
94.8489%

------------------------------------------------------------
CLUSTER-NUMBER SELECTION
------------------------------------------------------------

{metrics.to_string(index=False)}

Selected k by maximum silhouette:
{best_k}

Best silhouette score:
{metrics.loc[
    metrics["k"] == best_k,
    "silhouette_score"
].iloc[0]}

Calinski-Harabasz at selected k:
{metrics.loc[
    metrics["k"] == best_k,
    "calinski_harabasz"
].iloc[0]}

Davies-Bouldin at selected k:
{metrics.loc[
    metrics["k"] == best_k,
    "davies_bouldin"
].iloc[0]}

------------------------------------------------------------
CLUSTER SIZES
------------------------------------------------------------

{counts.to_string()}

------------------------------------------------------------
PARETO CANDIDATE CLUSTER MEMBERSHIP
------------------------------------------------------------

{pareto_cluster[
    report_cols
].to_string(index=False)}

------------------------------------------------------------
PARETO COUNTS BY CLUSTER
------------------------------------------------------------

{pareto_cluster_counts.to_string(index=False)}

------------------------------------------------------------
PARETO DESCRIPTOR-SPACE DISTANCES
------------------------------------------------------------

Closest Pareto pair:

{closest_pair["IL1"]}
{closest_pair["IL2"]}

Euclidean distance in PC1-PC3:
{closest_pair["distance_PC123"]}

Most distant Pareto pair:

{most_distant_pair["IL1"]}
{most_distant_pair["IL2"]}

Euclidean distance in PC1-PC3:
{most_distant_pair["distance_PC123"]}

------------------------------------------------------------
OUTPUTS
------------------------------------------------------------

Model-selection metrics:
{METRICS_FILE}

All cluster assignments:
{CLUSTERED_FILE}

Cluster descriptor profiles:
{PROFILE_FILE}

Pareto cluster membership:
{PARETO_CLUSTER_FILE}

Cluster-number figure:
{FIG_METRICS}

Cluster map:
{FIG_CLUSTERS}

Cluster map with Pareto candidates:
{FIG_PARETO}

------------------------------------------------------------
IMPORTANT
------------------------------------------------------------

Clustering is used to characterize descriptor-space
families and candidate diversity.

It does NOT alter the Pareto dominance calculation and
does NOT constitute a thermodynamic ranking.

============================================================
"""


REPORT_FILE.write_text(
    report
)

print(
    report
)

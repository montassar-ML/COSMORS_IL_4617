from pathlib import Path
from math import gcd
import re

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES_FILE = (
    ROOT
    / "02_candidates"
    / "IL_candidates_master.csv"
)

ION_DESC_FILE = (
    ROOT
    / "04_descriptors"
    / "ions"
    / "ion_descriptors_final.csv"
)

ION_PROFILE_FILE = (
    ROOT
    / "04_descriptors"
    / "ions"
    / "ion_sigma_profiles_final.csv"
)

OUT = (
    ROOT
    / "04_descriptors"
    / "ILs"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

IL_DESC_FILE = (
    OUT
    / "IL_descriptors_master.csv"
)

IL_PROFILE_FILE = (
    OUT
    / "IL_sigma_profiles_master.csv"
)

REPORT_FILE = (
    OUT
    / "stage4c_validation.txt"
)


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_ILS = 4617
EXPECTED_CATIONS = 81
EXPECTED_ANIONS = 57

SIGMA_MIN = -0.060
SIGMA_MAX = +0.060
DELTA_SIGMA = 0.001
POINTS = 121

HB_CUTOFF = 0.010

AREA_TOL_PCT = 0.05
LINEARITY_TOL = 1e-9


# ============================================================
# HELPERS
# ============================================================

def norm(s):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(s).lower(),
    )


def find_named_column(df, aliases):

    lookup = {
        norm(c): c
        for c in df.columns
    }

    for alias in aliases:

        key = norm(alias)

        if key in lookup:
            return lookup[key]

    return None


def find_pattern_column(df, pattern):

    hits = []

    rx = re.compile(pattern)

    for col in df.columns:

        vals = (
            df[col]
            .dropna()
            .astype(str)
            .str.strip()
        )

        if len(vals) == 0:
            continue

        frac = vals.map(
            lambda x: bool(
                rx.fullmatch(x)
            )
        ).mean()

        if frac > 0.99:
            hits.append(col)

    if len(hits) == 1:
        return hits[0]

    return None


# ============================================================
# LOAD DATABASES
# ============================================================

cand = pd.read_csv(
    CANDIDATES_FILE
)

ions = pd.read_csv(
    ION_DESC_FILE
)

profiles = pd.read_csv(
    ION_PROFILE_FILE
)


# ============================================================
# IDENTIFY CANDIDATE COLUMNS ROBUSTLY
# ============================================================

il_col = find_named_column(
    cand,
    [
        "IL_ID",
        "il_id",
        "ILID",
    ]
)

cation_col = find_named_column(
    cand,
    [
        "cation_id",
        "cation_ion_id",
        "cationID",
        "cation",
    ]
)

anion_col = find_named_column(
    cand,
    [
        "anion_id",
        "anion_ion_id",
        "anionID",
        "anion",
    ]
)


if il_col is None:
    il_col = find_pattern_column(
        cand,
        r"IL\d{4}",
    )

if cation_col is None:
    cation_col = find_pattern_column(
        cand,
        r"C\d{3}",
    )

if anion_col is None:
    anion_col = find_pattern_column(
        cand,
        r"A\d{3}",
    )


if (
    il_col is None
    or cation_col is None
    or anion_col is None
):

    print("Candidate table columns:")
    print(list(cand.columns))

    raise RuntimeError(
        "Could not identify IL_ID, "
        "cation ID, or anion ID column."
    )


print("============================================")
print("STAGE 4C / 4D")
print("============================================")
print("Candidate table :", CANDIDATES_FILE)
print("IL ID column    :", il_col)
print("Cation column   :", cation_col)
print("Anion column    :", anion_col)
print("Candidates      :", len(cand))
print("============================================")


# ============================================================
# BASIC DATABASE QC
# ============================================================

if len(cand) != EXPECTED_ILS:
    raise RuntimeError(
        f"Expected {EXPECTED_ILS} ILs; "
        f"found {len(cand)}"
    )

if len(ions) != 138:
    raise RuntimeError(
        f"Expected 138 ions; "
        f"found {len(ions)}"
    )

if profiles["ion_id"].nunique() != 138:
    raise RuntimeError(
        "Ion profile database does not "
        "contain 138 unique ions"
    )

if len(profiles) != 138 * POINTS:
    raise RuntimeError(
        "Ion profile table has incorrect "
        "number of rows"
    )


# ============================================================
# LOOKUP TABLES
# ============================================================

ion_lookup = ions.set_index(
    "ion_id"
)

profile_lookup = {}

for ion_id, grp in profiles.groupby(
    "ion_id"
):

    g = grp.sort_values(
        "sigma_e_A2"
    )

    sigma = g[
        "sigma_e_A2"
    ].to_numpy(dtype=float)

    total = g[
        "total_profile_A2"
    ].to_numpy(dtype=float)

    hb = g[
        "hb_profile_A2"
    ].to_numpy(dtype=float)

    if len(sigma) != POINTS:
        raise RuntimeError(
            f"{ion_id}: profile has "
            f"{len(sigma)} points"
        )

    if not np.allclose(
        sigma,
        np.round(
            np.arange(
                SIGMA_MIN,
                SIGMA_MAX
                + DELTA_SIGMA / 2,
                DELTA_SIGMA,
            ),
            3,
        ),
        atol=1e-12,
    ):
        raise RuntimeError(
            f"{ion_id}: unexpected sigma grid"
        )

    profile_lookup[ion_id] = (
        sigma,
        total,
        hb,
    )


# ============================================================
# BUILD ALL IONIC LIQUIDS
# ============================================================

descriptor_rows = []
profile_rows = []

count_11 = 0
count_21 = 0

for idx, row in cand.iterrows():

    IL_ID = str(
        row[il_col]
    ).strip()

    cid = str(
        row[cation_col]
    ).strip()

    aid = str(
        row[anion_col]
    ).strip()


    if cid not in ion_lookup.index:
        raise RuntimeError(
            f"{IL_ID}: unknown cation {cid}"
        )

    if aid not in ion_lookup.index:
        raise RuntimeError(
            f"{IL_ID}: unknown anion {aid}"
        )


    c = ion_lookup.loc[cid]
    a = ion_lookup.loc[aid]

    q_c = int(c["charge"])
    q_a = int(a["charge"])


    # --------------------------------------------------------
    # CHARGE-BALANCED STOICHIOMETRY
    # --------------------------------------------------------

    g = gcd(
        abs(q_c),
        abs(q_a),
    )

    n_c = (
        abs(q_a)
        // g
    )

    n_a = (
        abs(q_c)
        // g
    )

    net_charge = (
        n_c * q_c
        + n_a * q_a
    )

    if net_charge != 0:
        raise RuntimeError(
            f"{IL_ID}: non-neutral "
            f"stoichiometry generated"
        )

    if (
        n_c == 1
        and n_a == 1
    ):
        count_11 += 1

    elif (
        n_c == 2
        and n_a == 1
    ):
        count_21 += 1


    # --------------------------------------------------------
    # ION PROFILES
    # --------------------------------------------------------

    sigma_c, p_c, hb_c = (
        profile_lookup[cid]
    )

    sigma_a, p_a, hb_a = (
        profile_lookup[aid]
    )

    if not np.allclose(
        sigma_c,
        sigma_a,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"{IL_ID}: ion sigma grids differ"
        )

    sigma = sigma_c


    # --------------------------------------------------------
    # FORMULA-UNIT IL PROFILE
    #
    # IMPORTANT:
    # stoichiometric SUM, NOT average
    # --------------------------------------------------------

    p_IL = (
        n_c * p_c
        + n_a * p_a
    )

    hb_IL = (
        n_c * hb_c
        + n_a * hb_a
    )


    # --------------------------------------------------------
    # SIZE DESCRIPTORS
    # --------------------------------------------------------

    A_expected = (
        n_c
        * float(c["area_cosmo_A2"])
        +
        n_a
        * float(a["area_cosmo_A2"])
    )

    V_IL = (
        n_c
        * float(c["volume_cosmo_A3"])
        +
        n_a
        * float(a["volume_cosmo_A3"])
    )


    # --------------------------------------------------------
    # SIGMA MOMENTS FROM COMBINED PROFILE
    # --------------------------------------------------------

    MOM0 = float(
        np.sum(p_IL)
    )

    MOM1 = float(
        np.sum(
            p_IL
            * sigma
        )
    )

    MOM2 = float(
        np.sum(
            p_IL
            * sigma**2
        )
    )

    MOM3 = float(
        np.sum(
            p_IL
            * sigma**3
        )
    )


    HBACC2 = float(
        np.sum(
            hb_IL
            * np.maximum(
                0.0,
                sigma - HB_CUTOFF,
            )
        )
    )

    HBDON2 = float(
        np.sum(
            hb_IL
            * np.maximum(
                0.0,
                -sigma - HB_CUTOFF,
            )
        )
    )


    # --------------------------------------------------------
    # LINEARITY CROSS-CHECK
    # --------------------------------------------------------

    MOM2_expected = (
        n_c * float(c["MOM2"])
        +
        n_a * float(a["MOM2"])
    )

    MOM3_expected = (
        n_c * float(c["MOM3"])
        +
        n_a * float(a["MOM3"])
    )

    HBACC2_expected = (
        n_c * float(c["HBACC2"])
        +
        n_a * float(a["HBACC2"])
    )

    HBDON2_expected = (
        n_c * float(c["HBDON2"])
        +
        n_a * float(a["HBDON2"])
    )


    lin_err = max(
        abs(
            MOM2
            - MOM2_expected
        ),
        abs(
            MOM3
            - MOM3_expected
        ),
        abs(
            HBACC2
            - HBACC2_expected
        ),
        abs(
            HBDON2
            - HBDON2_expected
        ),
    )


    # --------------------------------------------------------
    # QC
    # --------------------------------------------------------

    area_error_pct = (
        100.0
        * (
            MOM0
            - A_expected
        )
        / A_expected
    )


    charge_from_profile = (
        -MOM1
    )


    endpoint = max(
        abs(float(p_IL[0])),
        abs(float(p_IL[-1])),
        abs(float(hb_IL[0])),
        abs(float(hb_IL[-1])),
    )


    total_nonnegative = bool(
        np.all(
            p_IL >= -1e-12
        )
    )

    hb_nonnegative = bool(
        np.all(
            hb_IL >= -1e-12
        )
    )

    hb_le_total = bool(
        np.all(
            hb_IL
            <= p_IL + 2e-6
        )
    )


    contains_warning_ion = (
        c["stage4b_final_status"]
        != "PASS"
        or
        a["stage4b_final_status"]
        != "PASS"
    )


    basic_ok = all([
        abs(area_error_pct)
        <= AREA_TOL_PCT,

        endpoint <= 1e-6,

        total_nonnegative,

        hb_nonnegative,

        hb_le_total,

        lin_err
        <= LINEARITY_TOL,
    ])


    if not basic_ok:

        status = "FAIL"

    elif contains_warning_ion:

        status = (
            "PASS_WITH_SOURCE_WARNING"
        )

    else:

        status = "PASS"


    notes = ""

    if contains_warning_ion:

        warning_ids = []

        if (
            c["stage4b_final_status"]
            != "PASS"
        ):
            warning_ids.append(cid)

        if (
            a["stage4b_final_status"]
            != "PASS"
        ):
            warning_ids.append(aid)

        notes = (
            "Contains accepted ion-level "
            "QC warning: "
            + ",".join(warning_ids)
        )


    # --------------------------------------------------------
    # DESCRIPTOR ROW
    # --------------------------------------------------------

    descriptor_rows.append({

        "IL_ID": IL_ID,

        "cation_id": cid,
        "cation_name": c["name"],
        "cation_abbreviation": (
            c["abbreviation"]
        ),

        "anion_id": aid,
        "anion_name": a["name"],
        "anion_abbreviation": (
            a["abbreviation"]
        ),

        "cation_charge": q_c,
        "anion_charge": q_a,

        "cation_count": n_c,
        "anion_count": n_a,

        "formula_unit_net_charge": (
            net_charge
        ),

        "stoichiometry": (
            f"{n_c}:{n_a}"
        ),

        "A_IL_A2": A_expected,
        "V_IL_A3": V_IL,

        "MOM0_A2": MOM0,
        "MOM1": MOM1,

        "charge_from_profile": (
            charge_from_profile
        ),

        "MOM2": MOM2,
        "MOM2_A": MOM2 / MOM0,

        "MOM3": MOM3,
        "MOM3_A": MOM3 / MOM0,

        "HBACC2": HBACC2,
        "HBACC2_A": (
            HBACC2 / MOM0
        ),

        "HBDON2": HBDON2,
        "HBDON2_A": (
            HBDON2 / MOM0
        ),

        "area_error_pct": (
            area_error_pct
        ),

        "linearity_error_max": (
            lin_err
        ),

        "max_endpoint": endpoint,

        "total_nonnegative": (
            total_nonnegative
        ),

        "hb_nonnegative": (
            hb_nonnegative
        ),

        "hb_le_total": (
            hb_le_total
        ),

        "contains_warning_ion": (
            contains_warning_ion
        ),

        "stage4c_status": status,

        "notes": notes,
    })


    # --------------------------------------------------------
    # FULL 121-POINT IL PROFILE
    # --------------------------------------------------------

    for s, pv, hv in zip(
        sigma,
        p_IL,
        hb_IL,
    ):

        profile_rows.append({

            "IL_ID": IL_ID,

            "sigma_e_A2": s,

            "total_profile_A2": pv,

            "hb_profile_A2": hv,
        })


    if (
        (idx + 1) % 250 == 0
        or
        idx + 1 == EXPECTED_ILS
    ):

        print(
            f"[{idx+1:4d}/{EXPECTED_ILS}] "
            f"{IL_ID}"
        )


# ============================================================
# SAVE
# ============================================================

desc = pd.DataFrame(
    descriptor_rows
)

il_profiles = pd.DataFrame(
    profile_rows
)


desc.to_csv(
    IL_DESC_FILE,
    index=False,
)

il_profiles.to_csv(
    IL_PROFILE_FILE,
    index=False,
)


# ============================================================
# GLOBAL QC
# ============================================================

expected_profile_rows = (
    EXPECTED_ILS
    * POINTS
)

n_pass = int(
    (
        desc["stage4c_status"]
        == "PASS"
    ).sum()
)

n_warn = int(
    (
        desc["stage4c_status"]
        == "PASS_WITH_SOURCE_WARNING"
    ).sum()
)

n_fail = int(
    (
        desc["stage4c_status"]
        == "FAIL"
    ).sum()
)


duplicate_IL_IDs = int(
    desc["IL_ID"]
    .duplicated()
    .sum()
)


unique_pairs = int(
    desc[
        [
            "cation_id",
            "anion_id",
        ]
    ]
    .drop_duplicates()
    .shape[0]
)


max_area_error = float(
    desc["area_error_pct"]
    .abs()
    .max()
)

max_linearity_error = float(
    desc["linearity_error_max"]
    .abs()
    .max()
)

max_endpoint = float(
    desc["max_endpoint"]
    .abs()
    .max()
)


if (
    len(desc) == EXPECTED_ILS
    and
    duplicate_IL_IDs == 0
    and
    unique_pairs == EXPECTED_ILS
    and
    len(il_profiles)
    == expected_profile_rows
    and
    il_profiles["IL_ID"]
    .nunique()
    == EXPECTED_ILS
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


warnings = desc[
    desc["stage4c_status"]
    != "PASS"
][
    [
        "IL_ID",
        "cation_id",
        "anion_id",
        "stoichiometry",
        "charge_from_profile",
        "stage4c_status",
        "notes",
    ]
]


report = f"""
============================================================
STAGE 4C/4D - 4617 IL INTRINSIC DESCRIPTOR DATABASE
============================================================

Overall status                  : {overall}

------------------------------------------------------------
CANDIDATE SPACE
------------------------------------------------------------

Expected ILs                    : {EXPECTED_ILS}
Generated ILs                   : {len(desc)}

Unique IL IDs                   : {desc["IL_ID"].nunique()}
Duplicate IL IDs                : {duplicate_IL_IDs}

Unique cation-anion pairs       : {unique_pairs}

------------------------------------------------------------
STOICHIOMETRY
------------------------------------------------------------

1:1 ILs                         : {count_11}
2:1 ILs                         : {count_21}

Non-neutral formula units       : {(desc["formula_unit_net_charge"] != 0).sum()}

------------------------------------------------------------
COMMON SIGMA GRID
------------------------------------------------------------

Sigma minimum                   : {SIGMA_MIN}
Sigma maximum                   : {SIGMA_MAX}
Delta sigma                     : {DELTA_SIGMA}
Points per IL                   : {POINTS}

Expected profile rows           : {expected_profile_rows}
Actual profile rows             : {len(il_profiles)}
Unique IL profiles              : {il_profiles["IL_ID"].nunique()}

------------------------------------------------------------
STATUS
------------------------------------------------------------

PASS                            : {n_pass}
PASS_WITH_SOURCE_WARNING        : {n_warn}
FAIL                            : {n_fail}

------------------------------------------------------------
NUMERICAL QC
------------------------------------------------------------

Maximum absolute area error %   : {max_area_error}
Maximum descriptor linearity err: {max_linearity_error}
Maximum endpoint                : {max_endpoint}

Negative total profiles         : {(~desc["total_nonnegative"]).sum()}
Negative HB profiles            : {(~desc["hb_nonnegative"]).sum()}
HB > total violations           : {(~desc["hb_le_total"]).sum()}

------------------------------------------------------------
SOURCE WARNING
------------------------------------------------------------

Accepted ion-level warning:
C071 = methyltrioctylphosphonium (P1,8,8,8)

Number of ILs containing an
accepted source warning          : {n_warn}

These ILs are retained but flagged.
They are NOT treated as failed ILs.

------------------------------------------------------------
OUTPUTS
------------------------------------------------------------

{IL_DESC_FILE}

{IL_PROFILE_FILE}

------------------------------------------------------------
NEXT VALIDATION
------------------------------------------------------------

The algebraically constructed IL profiles/descriptors are
PROVISIONAL until Stage 4E direct multi-species COSMO-RS
validation confirms the stoichiometric-sum construction.

============================================================
"""

REPORT_FILE.write_text(
    report
)

print(report)

from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

BOHR = 0.529177210903

COMMON_SIGMA_MAX = 0.060
STEP = 0.001
EXPECTED_POINTS = 121

HB_CUTOFF = 0.010

AREA_TOL_PCT = 0.05
CHARGE_TOL_E = 0.05

# Warning threshold below the formal 0.05 e failure limit.
CHARGE_WARNING_E = 0.02

SOURCE_EDGE_TOL = 0.001


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ION_DB = ROOT / "01_database" / "ions_master.csv"

ION_DIR = ROOT / "04_descriptors" / "ions"
RECOVERY = ROOT / "04_descriptors" / "recovery"

OLD_DESC = ION_DIR / "ion_descriptors.csv"
OLD_PROFILES = ION_DIR / "ion_sigma_profiles.csv"
JOB_STATUS = ION_DIR / "ion_job_status.csv"

FINAL_DESC = ION_DIR / "ion_descriptors_final.csv"
FINAL_PROFILES = ION_DIR / "ion_sigma_profiles_final.csv"
FINAL_REPORT = ION_DIR / "stage4b_final_validation.txt"


# ============================================================
# COMMON GRID
# ============================================================

COMMON_GRID = np.round(
    np.arange(
        -COMMON_SIGMA_MAX,
        COMMON_SIGMA_MAX + STEP / 2,
        STEP,
    ),
    3,
)

assert len(COMMON_GRID) == EXPECTED_POINTS


# ============================================================
# HELPERS
# ============================================================

BASH = shutil.which("bash")

if not BASH:
    raise RuntimeError("bash not found")


def to_msys_path(path):
    s = str(path)

    if len(s) >= 2 and s[1] == ":":
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/")
        return "/" + drive + rest

    return s.replace("\\", "/")


def amsreport_numbers(crskf, key):

    r = subprocess.run(
        [
            BASH,
            "-lc",
            '"$AMSBIN/amsreport" "$1" "$2"',
            "ams",
            to_msys_path(crskf),
            key,
        ],
        capture_output=True,
        text=True,
        errors="ignore",
    )

    if r.returncode != 0:
        raise RuntimeError(
            f"amsreport failed:\n"
            f"file={crskf}\n"
            f"key={key}\n"
            f"{r.stderr}"
        )

    values = []

    for line in r.stdout.splitlines():

        if line.strip().startswith("Property"):
            continue

        values.extend(
            float(x)
            for x in re.findall(
                r"[-+]?(?:\d+\.\d+|\d+)(?:[Ee][-+]?\d+)?",
                line
            )
        )

    return np.array(values, dtype=float)


def parse_cosmo_area_volume(cosmo):

    text = Path(cosmo).read_text(errors="ignore")

    ma = re.search(
        r"^\s*area\s*=\s*([-+0-9.Ee]+)",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    mv = re.search(
        r"^\s*volume\s*=\s*([-+0-9.Ee]+)",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    if not ma or not mv:
        raise RuntimeError(
            f"Could not parse area/volume from {cosmo}"
        )

    area_au2 = float(ma.group(1))
    volume_au3 = float(mv.group(1))

    area_A2 = area_au2 * BOHR**2
    volume_A3 = volume_au3 * BOHR**3

    return (
        area_au2,
        volume_au3,
        area_A2,
        volume_A3,
    )


def validate_raw_profile(sigma, total, hb, expected_points):

    if not (
        len(sigma)
        == len(total)
        == len(hb)
        == expected_points
    ):
        raise RuntimeError(
            f"Unexpected profile lengths: "
            f"{len(sigma)}, {len(total)}, {len(hb)}"
        )

    if len(sigma) > 1:

        if not np.allclose(
            np.diff(sigma),
            STEP,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Non-uniform sigma grid"
            )

    if np.any(total < -1e-12):
        raise RuntimeError(
            "Negative total profile value detected"
        )

    if np.any(hb < -1e-12):
        raise RuntimeError(
            "Negative HB profile value detected"
        )

    if np.any(hb > total + 2e-6):
        raise RuntimeError(
            "HB profile exceeds total profile"
        )


def place_on_common_grid(sigma, total, hb):

    p = np.zeros(
        len(COMMON_GRID),
        dtype=float,
    )

    h = np.zeros(
        len(COMMON_GRID),
        dtype=float,
    )

    for s, pv, hv in zip(
        sigma,
        total,
        hb,
    ):

        idx = int(
            round(
                (s + COMMON_SIGMA_MAX)
                / STEP
            )
        )

        if idx < 0 or idx >= len(COMMON_GRID):
            raise RuntimeError(
                f"Sigma {s} is outside common grid"
            )

        if not np.isclose(
            COMMON_GRID[idx],
            s,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Grid mapping error for sigma={s}"
            )

        p[idx] = pv
        h[idx] = hv

    return p, h


# ============================================================
# LOAD DATA
# ============================================================

ions = (
    pd.read_csv(ION_DB)
    .sort_values("ion_id")
    .reset_index(drop=True)
)

old_desc = pd.read_csv(OLD_DESC)
old_profiles = pd.read_csv(OLD_PROFILES)
jobs = pd.read_csv(JOB_STATUS)

if len(ions) != 138:
    raise RuntimeError(
        f"Expected 138 ions, found {len(ions)}"
    )

ion_lookup = ions.set_index("ion_id")
desc_lookup = old_desc.set_index("ion_id")
job_lookup = jobs.set_index("ion_id")


# ============================================================
# RECOVER C060 / C081
# ============================================================

special_profiles = {}

for ion_id in ["C060", "C081"]:

    crskf = Path(
        job_lookup.loc[
            ion_id,
            "crskf_path"
        ]
    )

    sigma = amsreport_numbers(
        crskf,
        "sigma",
    )

    total = amsreport_numbers(
        crskf,
        "sigma-profile",
    )

    hb = amsreport_numbers(
        crskf,
        "sigma-profile-hb",
    )

    validate_raw_profile(
        sigma,
        total,
        hb,
        61,
    )

    special_profiles[ion_id] = (
        sigma,
        total,
        hb,
    )


# ============================================================
# C071 FINAL ±0.060 PROFILE
# ============================================================

c071_candidates = list(
    (
        RECOVERY
        / "C071_range_sweep"
    ).rglob(
        "C071_060.crskf"
    )
)

if len(c071_candidates) != 1:

    raise RuntimeError(
        "Could not uniquely locate "
        "C071_060.crskf. Found: "
        f"{c071_candidates}"
    )

c071_crskf = c071_candidates[0]

sigma = amsreport_numbers(
    c071_crskf,
    "sigma",
)

total = amsreport_numbers(
    c071_crskf,
    "sigma-profile",
)

hb = amsreport_numbers(
    c071_crskf,
    "sigma-profile-hb",
)

validate_raw_profile(
    sigma,
    total,
    hb,
    121,
)

special_profiles["C071"] = (
    sigma,
    total,
    hb,
)


# ============================================================
# AREA / VOLUME PATCHES
# ============================================================

area_volume_patch = {}

for ion_id in [
    "C060",
    "C071",
    "C081",
]:

    cosmo = (
        RECOVERY
        / f"{ion_id}.cosmo"
    )

    if not cosmo.exists():

        raise RuntimeError(
            f"Required recovery file missing: {cosmo}"
        )

    area_volume_patch[ion_id] = (
        parse_cosmo_area_volume(cosmo)
    )


# ============================================================
# PROCESS ALL 138 IONS
# ============================================================

descriptor_rows = []
profile_rows = []

for n, row in ions.iterrows():

    ion_id = row["ion_id"]

    # --------------------------------------------------------
    # GET SOURCE PROFILE
    # --------------------------------------------------------

    if ion_id in special_profiles:

        sigma, total, hb = (
            special_profiles[ion_id]
        )

        if ion_id == "C071":
            source = "direct_0060"
            source_sigma_max = 0.060

        else:
            source = (
                "recovered_0030_zero_padded"
            )
            source_sigma_max = 0.030

    else:

        x = (
            old_profiles[
                old_profiles["ion_id"]
                == ion_id
            ]
            .sort_values(
                "sigma_e_A2"
            )
        )

        if len(x) != 61:

            raise RuntimeError(
                f"{ion_id}: expected 61 old "
                f"profile rows, got {len(x)}"
            )

        sigma = (
            x["sigma_e_A2"]
            .to_numpy(dtype=float)
        )

        total = (
            x["total_profile_A2"]
            .to_numpy(dtype=float)
        )

        hb = (
            x["hb_profile_A2"]
            .to_numpy(dtype=float)
        )

        validate_raw_profile(
            sigma,
            total,
            hb,
            61,
        )

        source = (
            "validated_0030_zero_padded"
        )

        source_sigma_max = 0.030

    # --------------------------------------------------------
    # SOURCE EDGE QC
    # --------------------------------------------------------

    source_max_edge = max(
        abs(float(total[0])),
        abs(float(total[-1])),
        abs(float(hb[0])),
        abs(float(hb[-1])),
    )

    # --------------------------------------------------------
    # PUT ON COMMON ±0.060 GRID
    # --------------------------------------------------------

    total_common, hb_common = (
        place_on_common_grid(
            sigma,
            total,
            hb,
        )
    )

    # --------------------------------------------------------
    # AREA / VOLUME
    # --------------------------------------------------------

    if ion_id in area_volume_patch:

        (
            area_au2,
            volume_au3,
            area_A2,
            volume_A3,
        ) = area_volume_patch[ion_id]

    else:

        d = desc_lookup.loc[ion_id]

        area_au2 = float(
            d["area_cosmo_au2"]
        )

        volume_au3 = float(
            d["volume_cosmo_au3"]
        )

        area_A2 = float(
            d["area_cosmo_A2"]
        )

        volume_A3 = float(
            d["volume_cosmo_A3"]
        )

    # --------------------------------------------------------
    # DESCRIPTORS ON COMMON GRID
    # --------------------------------------------------------

    MOM0 = float(
        np.sum(total_common)
    )

    MOM1 = float(
        np.sum(
            total_common
            * COMMON_GRID
        )
    )

    MOM2 = float(
        np.sum(
            total_common
            * COMMON_GRID**2
        )
    )

    MOM3 = float(
        np.sum(
            total_common
            * COMMON_GRID**3
        )
    )

    HBACC2 = float(
        np.sum(
            hb_common
            * np.maximum(
                0.0,
                COMMON_GRID - HB_CUTOFF,
            )
        )
    )

    HBDON2 = float(
        np.sum(
            hb_common
            * np.maximum(
                0.0,
                -COMMON_GRID - HB_CUTOFF,
            )
        )
    )

    charge_expected = int(
        row["charge"]
    )

    charge_from_profile = -MOM1

    charge_error = (
        charge_from_profile
        - charge_expected
    )

    area_error_pct = (
        100.0
        * (MOM0 - area_A2)
        / area_A2
    )

    # --------------------------------------------------------
    # FINAL PROFILE QC
    # --------------------------------------------------------

    common_endpoint = max(
        abs(total_common[0]),
        abs(total_common[-1]),
        abs(hb_common[0]),
        abs(hb_common[-1]),
    )

    total_nonnegative = bool(
        np.all(
            total_common >= -1e-12
        )
    )

    hb_nonnegative = bool(
        np.all(
            hb_common >= -1e-12
        )
    )

    hb_le_total = bool(
        np.all(
            hb_common
            <= total_common + 2e-6
        )
    )

    # Source endpoint criterion does not apply to C071's
    # rejected ±0.030 profile; its accepted source is ±0.060.
    source_edge_ok = (
        source_max_edge
        <= SOURCE_EDGE_TOL
    )

    area_ok = (
        abs(area_error_pct)
        <= AREA_TOL_PCT
    )

    charge_ok = (
        abs(charge_error)
        <= CHARGE_TOL_E
    )

    common_edge_ok = (
        common_endpoint
        <= 1e-6
    )

    basic_ok = all([
        source_edge_ok,
        area_ok,
        charge_ok,
        common_edge_ok,
        total_nonnegative,
        hb_nonnegative,
        hb_le_total,
    ])

    if not basic_ok:

        status = "FAIL"

    elif (
        abs(charge_error)
        > CHARGE_WARNING_E
    ):

        status = "PASS_WITH_WARNING"

    else:

        status = "PASS"

    note = ""

    if status == "PASS_WITH_WARNING":

        note = (
            "Charge from MOM1 differs from "
            f"formal charge by "
            f"{abs(charge_error):.6f} e; "
            "within predefined 0.05 e "
            "acceptance tolerance."
        )

    # --------------------------------------------------------
    # SAVE DESCRIPTOR ROW
    # --------------------------------------------------------

    descriptor_rows.append({
        "ion_id": ion_id,
        "name": row["name"],
        "abbreviation": row[
            "abbreviation_primary"
        ],
        "type": row["type"],
        "charge": charge_expected,

        "profile_source": source,
        "source_sigma_max": source_sigma_max,
        "common_sigma_max": COMMON_SIGMA_MAX,
        "delta_sigma": STEP,
        "common_points": EXPECTED_POINTS,

        "area_cosmo_au2": area_au2,
        "volume_cosmo_au3": volume_au3,
        "area_cosmo_A2": area_A2,
        "volume_cosmo_A3": volume_A3,

        "MOM0_A2": MOM0,
        "MOM1": MOM1,
        "MOM2": MOM2,
        "MOM3": MOM3,

        "MOM2_A": MOM2 / MOM0,
        "MOM3_A": MOM3 / MOM0,

        "HBACC2": HBACC2,
        "HBDON2": HBDON2,

        "HBACC2_A": HBACC2 / MOM0,
        "HBDON2_A": HBDON2 / MOM0,

        "charge_from_profile": (
            charge_from_profile
        ),
        "charge_error": charge_error,

        "area_error_pct": area_error_pct,

        "source_max_endpoint": (
            source_max_edge
        ),

        "common_max_endpoint": (
            common_endpoint
        ),

        "total_nonnegative": (
            total_nonnegative
        ),

        "hb_nonnegative": (
            hb_nonnegative
        ),

        "hb_le_total": hb_le_total,

        "stage4b_final_status": status,
        "notes": note,
    })

    # --------------------------------------------------------
    # SAVE 121-POINT COMMON PROFILE
    # --------------------------------------------------------

    for s, pv, hv in zip(
        COMMON_GRID,
        total_common,
        hb_common,
    ):

        profile_rows.append({
            "ion_id": ion_id,
            "sigma_e_A2": s,
            "total_profile_A2": pv,
            "hb_profile_A2": hv,
        })

    print(
        f"[{n+1:3d}/138] "
        f"{ion_id:4s} "
        f"{status:17s} "
        f"Aerr={area_error_pct:+.5f}% "
        f"Q={charge_from_profile:+.6f} "
        f"Qerr={charge_error:+.6f} "
        f"edge={common_endpoint:.6g}"
    )


# ============================================================
# SAVE
# ============================================================

desc = pd.DataFrame(
    descriptor_rows
)

profiles = pd.DataFrame(
    profile_rows
)

desc.to_csv(
    FINAL_DESC,
    index=False,
)

profiles.to_csv(
    FINAL_PROFILES,
    index=False,
)


# ============================================================
# FINAL VALIDATION
# ============================================================

n_pass = int(
    (
        desc["stage4b_final_status"]
        == "PASS"
    ).sum()
)

n_warn = int(
    (
        desc["stage4b_final_status"]
        == "PASS_WITH_WARNING"
    ).sum()
)

n_fail = int(
    (
        desc["stage4b_final_status"]
        == "FAIL"
    ).sum()
)

profile_ions = (
    profiles["ion_id"].nunique()
)

profile_rows_count = len(profiles)

expected_rows = (
    138 * EXPECTED_POINTS
)

max_area_error = (
    desc["area_error_pct"]
    .abs()
    .max()
)

max_charge_error = (
    desc["charge_error"]
    .abs()
    .max()
)

max_common_edge = (
    desc["common_max_endpoint"]
    .max()
)

if (
    n_fail == 0
    and profile_ions == 138
    and profile_rows_count == expected_rows
):

    if n_warn == 0:
        overall = "PASS"
    else:
        overall = "PASS_WITH_WARNING"

else:
    overall = "FAIL"


warnings = desc[
    desc["stage4b_final_status"]
    != "PASS"
][
    [
        "ion_id",
        "abbreviation",
        "charge",
        "profile_source",
        "charge_from_profile",
        "charge_error",
        "area_error_pct",
        "common_max_endpoint",
        "stage4b_final_status",
        "notes",
    ]
]


report = f"""
============================================================
STAGE 4B - FINAL CONSOLIDATED ION DATABASE
============================================================

Overall status                  : {overall}

------------------------------------------------------------
DATABASE
------------------------------------------------------------

Total ions                      : {len(desc)}
Cations                         : {(desc["type"] == "cation").sum()}
Anions                          : {(desc["type"] == "anion").sum()}

------------------------------------------------------------
COMMON SIGMA GRID
------------------------------------------------------------

Sigma minimum                   : -0.060
Sigma maximum                   : +0.060
Delta sigma                     : 0.001
Points per ion                  : 121

Expected profile rows           : {expected_rows}
Actual profile rows             : {profile_rows_count}
Unique ions in profile table    : {profile_ions}

------------------------------------------------------------
FINAL STATUS
------------------------------------------------------------

PASS                            : {n_pass}
PASS_WITH_WARNING               : {n_warn}
FAIL                            : {n_fail}

------------------------------------------------------------
GLOBAL QC
------------------------------------------------------------

Maximum absolute area error %   : {max_area_error}
Maximum absolute charge error e : {max_charge_error}
Maximum common endpoint         : {max_common_edge}

Area tolerance %                : {AREA_TOL_PCT}
Charge tolerance e              : {CHARGE_TOL_E}

------------------------------------------------------------
NON-PASS / WARNING ROWS
------------------------------------------------------------

{warnings.to_string(index=False) if len(warnings) else "NONE"}

------------------------------------------------------------
SPECIAL RECOVERY
------------------------------------------------------------

C060:
  Existing successful COSMO-RS job reused.
  Area/volume recovered from safe-name COSMO file.

C081:
  Existing successful COSMO-RS job reused.
  Area/volume recovered from safe-name COSMO file.

C071:
  Final direct sigma profile:
  SigmaMax = 0.060 e/A^2
  Delta sigma = 0.001 e/A^2
  121 points.
  No boundary truncation.

------------------------------------------------------------
OUTPUTS
------------------------------------------------------------

{FINAL_DESC}

{FINAL_PROFILES}

============================================================
"""

print(report)

FINAL_REPORT.write_text(
    report
)

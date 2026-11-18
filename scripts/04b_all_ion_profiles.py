from pathlib import Path
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd

from scm.plams import (
    Settings,
    CRSJob,
    JobRunner,
    config,
    init,
    finish,
)

# ============================================================
# CONSTANTS
# ============================================================

BOHR = 0.529177210903

SIGMA_MAX = 0.030
NPROFILE = 60
EXPECTED_POINTS = 61
EXPECTED_STEP = 0.001

HB_CUTOFF = 0.010

AREA_TOL_PCT = 0.05
CHARGE_TOL_E = 0.02

# We report two endpoint thresholds.
ENDPOINT_STRICT = 1.0e-6
ENDPOINT_PRACTICAL = 1.0e-3

MAXJOBS = 8

# ============================================================
# PATHS
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]

ION_DB = PROJECT / "01_database" / "ions_master.csv"

OUT = PROJECT / "04_descriptors" / "ions"
JOBS = OUT / "jobs"
COSMO_ASCII = OUT / "cosmo_ascii"

OUT.mkdir(parents=True, exist_ok=True)
COSMO_ASCII.mkdir(parents=True, exist_ok=True)

# ============================================================
# EXECUTABLE DISCOVERY
# ============================================================

def find_ams_executable(name):
    p = shutil.which(name)

    if p:
        return p

    amsbin = Path(os.environ.get("AMSBIN", ""))

    candidates = [
        amsbin / name,
        amsbin / f"{name}.exe",
    ]

    for x in candidates:
        if x.exists():
            return str(x)

    raise RuntimeError(f"Cannot locate AMS executable: {name}")


AMSREPORT = find_ams_executable("amsreport")
KF2COSMO = find_ams_executable("kf2cosmo")

# ============================================================
# HELPERS
# ============================================================

def report_numbers(crskf, key):
    p = subprocess.run(
        [AMSREPORT, str(crskf), key],
        capture_output=True,
        text=True,
        errors="ignore",
        check=True,
    )

    values = []

    for line in p.stdout.splitlines():

        if line.strip().startswith("Property"):
            continue

        values.extend(
            float(x)
            for x in re.findall(
                r"[-+]?(?:\d+\.\d+|\d+)(?:[Ee][-+]?\d+)?",
                line,
            )
        )

    return np.array(values, dtype=float)


def get_cosmo_area_volume(coskf, ion_id):

    cosmo = COSMO_ASCII / f"{ion_id}.cosmo"

    if cosmo.exists():
        cosmo.unlink()

    subprocess.run(
        [KF2COSMO, str(coskf), str(cosmo)],
        check=True,
        capture_output=True,
        text=True,
        errors="ignore",
    )

    text = cosmo.read_text(errors="ignore")

    m_area = re.search(
        r"^\s*area\s*=\s*([-+0-9.Ee]+)",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    m_volume = re.search(
        r"^\s*volume\s*=\s*([-+0-9.Ee]+)",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    if not m_area or not m_volume:
        raise RuntimeError(
            f"Could not parse COSMO area/volume for {ion_id}"
        )

    area_au2 = float(m_area.group(1))
    volume_au3 = float(m_volume.group(1))

    area_A2 = area_au2 * BOHR**2
    volume_A3 = volume_au3 * BOHR**3

    return area_au2, volume_au3, area_A2, volume_A3


# ============================================================
# INPUT
# ============================================================

ions = (
    pd.read_csv(ION_DB)
    .sort_values("ion_id")
    .reset_index(drop=True)
)

if len(ions) != 138:
    raise RuntimeError(
        f"Expected 138 ions, found {len(ions)}"
    )

print("============================================================")
print("STAGE 4B - ALL ION SIGMA PROFILES")
print("============================================================")
print("Ions       :", len(ions))
print("SigmaMax   :", SIGMA_MAX)
print("Nprofile   :", NPROFILE)
print("Points     :", EXPECTED_POINTS)
print("Delta sigma:", EXPECTED_STEP)
print("Parallel jobs:", MAXJOBS)
print("============================================================")

# ============================================================
# INITIALIZE PLAMS
# ============================================================

init(
    path=str(OUT),
    folder="jobs"
)

config.default_jobrunner = JobRunner(
    parallel=True,
    maxjobs=MAXJOBS,
)

config.default_jobmanager.settings.hashing = None
config.job.runscript.nproc = 1
config.log.stdout = 1

# ============================================================
# SUBMIT 138 JOBS
# ============================================================

outputs = []
metadata = []

for _, row in ions.iterrows():

    ion_id = row["ion_id"]

    s = Settings()

    s.input.property._h = "puresigmaprofile"
    s.input.property.Nprofile = NPROFILE
    s.input.property.SigmaMax = SIGMA_MAX

    s.input.method = "COSMORS"

    compound = Settings()
    compound._h = str(Path(row["coskf_path"]))

    s.input.compound = [compound]

    job = CRSJob(settings=s)
    job.name = f"{ion_id}_sigma"

    out = job.run()

    outputs.append(out)
    metadata.append(row)

# ============================================================
# WAIT FOR ALL JOBS
# ============================================================

job_records = []

for row, out in zip(metadata, outputs):

    ion_id = row["ion_id"]

    try:
        out.get_results()

        job_path = Path(out.job.path)

        crskf = job_path / f"{out.job.name}.crskf"

        success = crskf.exists()

        job_records.append({
            "ion_id": ion_id,
            "job_success": success,
            "job_path": str(job_path),
            "crskf_path": str(crskf),
            "job_error": "",
        })

    except Exception as exc:

        job_records.append({
            "ion_id": ion_id,
            "job_success": False,
            "job_path": str(getattr(out.job, "path", "")),
            "crskf_path": "",
            "job_error": str(exc),
        })

finish()

job_df = pd.DataFrame(job_records)

# ============================================================
# EXTRACT PROFILES + DESCRIPTORS
# ============================================================

profile_rows = []
descriptor_rows = []

ion_lookup = ions.set_index("ion_id")

for _, jr in job_df.iterrows():

    ion_id = jr["ion_id"]

    row = ion_lookup.loc[ion_id]

    base = {
        "ion_id": ion_id,
        "name": row["name"],
        "abbreviation": row["abbreviation_primary"],
        "type": row["type"],
        "charge": int(row["charge"]),
    }

    if not jr["job_success"]:

        descriptor_rows.append({
            **base,
            "stage4b_status": "JOB_FAIL",
            "notes": jr["job_error"],
        })

        continue

    try:

        crskf = Path(jr["crskf_path"])

        sigma = report_numbers(
            crskf,
            "sigma",
        )

        total = report_numbers(
            crskf,
            "sigma-profile",
        )

        hb = report_numbers(
            crskf,
            "sigma-profile-hb",
        )

        # ----------------------------------------------------
        # Raw COSMO area + volume
        # ----------------------------------------------------

        (
            area_au2,
            volume_au3,
            area_A2,
            volume_A3,
        ) = get_cosmo_area_volume(
            Path(row["coskf_path"]),
            ion_id,
        )

        # ----------------------------------------------------
        # Profile validation
        # ----------------------------------------------------

        same_length = (
            len(sigma)
            == len(total)
            == len(hb)
        )

        points_ok = (
            same_length
            and len(sigma) == EXPECTED_POINTS
        )

        if len(sigma) > 1:
            ds = np.diff(sigma)
            uniform_grid = bool(
                np.allclose(
                    ds,
                    EXPECTED_STEP,
                    atol=1e-12,
                )
            )
        else:
            uniform_grid = False

        sigma_range_ok = bool(
            len(sigma) > 0
            and np.isclose(
                sigma[0],
                -SIGMA_MAX,
                atol=1e-12,
            )
            and np.isclose(
                sigma[-1],
                SIGMA_MAX,
                atol=1e-12,
            )
        )

        total_nonnegative = bool(
            np.all(total >= -1e-12)
        )

        hb_nonnegative = bool(
            np.all(hb >= -1e-12)
        )

        hb_le_total = bool(
            np.all(hb <= total + 1e-8)
        )

        # ----------------------------------------------------
        # Sigma moments
        # ----------------------------------------------------

        MOM0 = np.sum(total)

        MOM1 = np.sum(
            total * sigma
        )

        MOM2 = np.sum(
            total * sigma**2
        )

        MOM3 = np.sum(
            total * sigma**3
        )

        HBACC2 = np.sum(
            hb * np.maximum(
                0.0,
                sigma - HB_CUTOFF,
            )
        )

        HBDON2 = np.sum(
            hb * np.maximum(
                0.0,
                -sigma - HB_CUTOFF,
            )
        )

        # MOM1 is negative of molecular charge
        charge_from_profile = -MOM1

        charge_error = (
            charge_from_profile
            - int(row["charge"])
        )

        area_error_pct = (
            100.0
            * (MOM0 - area_A2)
            / area_A2
        )

        # ----------------------------------------------------
        # Boundary diagnostics
        # ----------------------------------------------------

        endpoint_total_left = total[0]
        endpoint_total_right = total[-1]

        endpoint_hb_left = hb[0]
        endpoint_hb_right = hb[-1]

        max_endpoint = max(
            abs(endpoint_total_left),
            abs(endpoint_total_right),
            abs(endpoint_hb_left),
            abs(endpoint_hb_right),
        )

        strict_boundary_ok = (
            max_endpoint <= ENDPOINT_STRICT
        )

        practical_boundary_ok = (
            max_endpoint <= ENDPOINT_PRACTICAL
        )

        # ----------------------------------------------------
        # QC status
        # ----------------------------------------------------

        basic_qc = all([
            points_ok,
            uniform_grid,
            sigma_range_ok,
            total_nonnegative,
            hb_nonnegative,
            hb_le_total,
            abs(area_error_pct) <= AREA_TOL_PCT,
            abs(charge_error) <= CHARGE_TOL_E,
        ])

        if not basic_qc:
            status = "FAIL"

        elif not practical_boundary_ok:
            status = "WIDER_GRID"

        else:
            status = "PASS"

        # ----------------------------------------------------
        # Store complete profiles
        # ----------------------------------------------------

        for s, p, h in zip(
            sigma,
            total,
            hb,
        ):

            profile_rows.append({
                **base,
                "sigma_e_A2": s,
                "total_profile_A2": p,
                "hb_profile_A2": h,
            })

        # ----------------------------------------------------
        # Store descriptors
        # ----------------------------------------------------

        descriptor_rows.append({

            **base,

            "area_cosmo_au2": area_au2,
            "volume_cosmo_au3": volume_au3,

            "area_cosmo_A2": area_A2,
            "volume_cosmo_A3": volume_A3,

            "MOM0_A2": MOM0,

            "area_error_pct": area_error_pct,

            "MOM1": MOM1,

            "charge_from_profile": charge_from_profile,
            "charge_error": charge_error,

            "MOM2": MOM2,
            "MOM2_A": MOM2 / MOM0,

            "MOM3": MOM3,
            "MOM3_A": MOM3 / MOM0,

            "HBACC2": HBACC2,
            "HBACC2_A": HBACC2 / MOM0,

            "HBDON2": HBDON2,
            "HBDON2_A": HBDON2 / MOM0,

            "endpoint_total_left": endpoint_total_left,
            "endpoint_total_right": endpoint_total_right,

            "endpoint_hb_left": endpoint_hb_left,
            "endpoint_hb_right": endpoint_hb_right,

            "max_endpoint": max_endpoint,

            "strict_boundary_ok": strict_boundary_ok,
            "practical_boundary_ok": practical_boundary_ok,

            "points": len(sigma),
            "sigma_min": sigma[0],
            "sigma_max": sigma[-1],

            "uniform_grid": uniform_grid,
            "total_nonnegative": total_nonnegative,
            "hb_nonnegative": hb_nonnegative,
            "hb_le_total": hb_le_total,

            "stage4b_status": status,
            "notes": "",
        })

    except Exception as exc:

        descriptor_rows.append({
            **base,
            "stage4b_status": "PARSE_FAIL",
            "notes": str(exc),
        })

# ============================================================
# SAVE
# ============================================================

profiles = pd.DataFrame(profile_rows)
desc = pd.DataFrame(descriptor_rows)

profiles.to_csv(
    OUT / "ion_sigma_profiles.csv",
    index=False,
)

desc.to_csv(
    OUT / "ion_descriptors.csv",
    index=False,
)

job_df.to_csv(
    OUT / "ion_job_status.csv",
    index=False,
)

# ============================================================
# GLOBAL QC
# ============================================================

n_job_success = int(
    job_df["job_success"].sum()
)

n_job_fail = (
    len(job_df) - n_job_success
)

status_counts = (
    desc["stage4b_status"]
    .value_counts(dropna=False)
)

n_pass = int(
    (desc["stage4b_status"] == "PASS").sum()
)

n_wider = int(
    (desc["stage4b_status"] == "WIDER_GRID").sum()
)

n_fail = int(
    desc["stage4b_status"]
    .isin(["FAIL", "JOB_FAIL", "PARSE_FAIL"])
    .sum()
)

valid = desc[
    desc["MOM0_A2"].notna()
].copy()

strict_offenders = valid[
    valid["max_endpoint"] > ENDPOINT_STRICT
].copy()

practical_offenders = valid[
    valid["max_endpoint"] > ENDPOINT_PRACTICAL
].copy()

strict_offenders = strict_offenders.sort_values(
    "max_endpoint",
    ascending=False,
)

practical_offenders = practical_offenders.sort_values(
    "max_endpoint",
    ascending=False,
)

if len(valid):

    max_area_error = valid[
        "area_error_pct"
    ].abs().max()

    max_charge_error = valid[
        "charge_error"
    ].abs().max()

    max_boundary = valid[
        "max_endpoint"
    ].max()

else:

    max_area_error = np.nan
    max_charge_error = np.nan
    max_boundary = np.nan

if n_fail == 0 and n_wider == 0:
    overall = "PASS"
elif n_fail == 0:
    overall = "NEEDS_WIDER_GRID_FOR_SOME_IONS"
else:
    overall = "FAIL"

# ============================================================
# REPORT
# ============================================================

report = f"""
============================================================
STAGE 4B - ALL ION PROFILE VALIDATION
============================================================

Overall status                  : {overall}

------------------------------------------------------------
CALCULATIONS
------------------------------------------------------------

Expected ions                   : 138
Jobs submitted                  : {len(job_df)}
Jobs successful                 : {n_job_success}
Jobs failed                     : {n_job_fail}

Parallel jobs                   : {MAXJOBS}
Cores per job                   : 1

------------------------------------------------------------
PRODUCTION SIGMA GRID
------------------------------------------------------------

Sigma minimum                   : {-SIGMA_MAX}
Sigma maximum                   : {SIGMA_MAX}
Nprofile                        : {NPROFILE}
Expected numerical points       : {EXPECTED_POINTS}
Delta sigma                     : {EXPECTED_STEP}

------------------------------------------------------------
ION QC
------------------------------------------------------------

PASS                            : {n_pass}
WIDER_GRID                      : {n_wider}
FAIL                            : {n_fail}

Status distribution:
{status_counts.to_string()}

------------------------------------------------------------
BOUNDARY QC
------------------------------------------------------------

Strict threshold                : {ENDPOINT_STRICT}
Practical threshold             : {ENDPOINT_PRACTICAL}

Ions > strict threshold         : {len(strict_offenders)}
Ions > practical threshold      : {len(practical_offenders)}

Largest endpoint value          : {max_boundary}

------------------------------------------------------------
AREA / CHARGE QC
------------------------------------------------------------

Maximum absolute area error %   : {max_area_error}
Maximum absolute charge error e : {max_charge_error}

------------------------------------------------------------
TOP BOUNDARY OFFENDERS
------------------------------------------------------------

{strict_offenders[
    [
        "ion_id",
        "abbreviation",
        "charge",
        "endpoint_total_left",
        "endpoint_total_right",
        "endpoint_hb_left",
        "endpoint_hb_right",
        "max_endpoint",
    ]
].head(20).to_string(index=False)}

------------------------------------------------------------
OUTPUTS
------------------------------------------------------------

{OUT / 'ion_sigma_profiles.csv'}
{OUT / 'ion_descriptors.csv'}
{OUT / 'ion_job_status.csv'}

============================================================
"""

print(report)

(OUT / "stage4b_validation.txt").write_text(
    report
)


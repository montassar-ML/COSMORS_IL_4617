from pathlib import Path
import os
import math
import traceback

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
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ION_FILE = (
    ROOT
    / "01_database"
    / "ions_master.csv"
)

IL_FILE = (
    ROOT
    / "04_descriptors"
    / "ILs"
    / "IL_descriptors_master.csv"
)

PILOT_FILE = (
    ROOT
    / "05_H2O"
    / "pilot"
    / "H2O_pilot_results.csv"
)

OUT = (
    ROOT
    / "05_H2O"
    / "production"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_FILE = (
    OUT
    / "H2O_infinite_dilution_298K.csv"
)

REPORT_FILE = (
    OUT
    / "stage5_H2O_validation.txt"
)

# ============================================================
# SETTINGS
# ============================================================

TEMPERATURE = 298.15

EXPECTED_ILS = 4617

MAXJOBS = 8
BATCH_SIZE = 200

# ============================================================
# DATABASES
# ============================================================

ions = pd.read_csv(
    ION_FILE
).set_index("ion_id")

ils = pd.read_csv(
    IL_FILE
).sort_values(
    "IL_ID"
).reset_index(drop=True)

if len(ils) != EXPECTED_ILS:
    raise RuntimeError(
        f"Expected {EXPECTED_ILS} ILs, "
        f"found {len(ils)}"
    )

# ============================================================
# SOLUTE
# ============================================================

WATER = Path(os.environ["COSMORS_SOLUTE_DB"]) / "Water.coskf"

if not WATER.exists():
    raise RuntimeError(
        f"Water.coskf not found: {WATER}"
    )

# ============================================================
# SAFE ION PATHS
# ============================================================

SAFE = {}

for ion_id in [
    "C060",
    "C071",
    "C081",
]:

    p = (
        ROOT
        / "04_descriptors"
        / "recovery"
        / f"{ion_id}.coskf"
    )

    if p.exists():
        SAFE[ion_id] = p


def ion_path(ion_id):

    if ion_id in SAFE:
        return SAFE[ion_id]

    return Path(
        ions.loc[
            ion_id,
            "coskf_path"
        ]
    )


# ============================================================
# JOB CONSTRUCTION
# ============================================================

def make_job(row):

    IL_ID = row["IL_ID"]

    cid = row["cation_id"]
    aid = row["anion_id"]

    nc = int(
        row["cation_count"]
    )

    na = int(
        row["anion_count"]
    )

    cpath = ion_path(cid)
    apath = ion_path(aid)

    if not cpath.exists():
        raise RuntimeError(
            f"{IL_ID}: missing cation file "
            f"{cpath}"
        )

    if not apath.exists():
        raise RuntimeError(
            f"{IL_ID}: missing anion file "
            f"{apath}"
        )

    s = Settings()

    s.input.property._h = (
        "activitycoef"
    )

    s.input.method = "COSMORS"

    s.input.temperature = (
        TEMPERATURE
    )

    # --------------------------------------------------------
    # Compound 1 = ionic liquid
    # --------------------------------------------------------

    IL = Settings()

    form = [
        Settings()
    ]

    species = [
        Settings(),
        Settings(),
    ]

    # Cation
    cs = [
        Settings()
    ]

    cs[0]._h = str(
        cpath
    )

    cs[0].count = nc

    species[0] = cs

    # Anion
    ans = [
        Settings()
    ]

    ans[0]._h = str(
        apath
    )

    ans[0].count = na

    species[1] = ans

    form[0].species = species

    IL.form = form
    IL.name = IL_ID
    IL.frac1 = 1.0

    # --------------------------------------------------------
    # Compound 2 = water at infinite dilution
    # --------------------------------------------------------

    H2O = Settings()

    H2O._h = str(
        WATER
    )

    H2O.name = "H2O"
    H2O.frac1 = 0.0

    s.input.compound = [
        IL,
        H2O,
    ]

    # C071 has an unsuitable cached sigma profile.
    # Generate from its COSMO surface instead.
    if cid == "C071":

        s.input.REUSE_SIGMA_PROFILE = (
            "No"
        )

        s.input.UPDATE_SIGMA_PROFILE = (
            "No"
        )

    job = CRSJob(
        settings=s,
        name=f"H2O_{IL_ID}",
    )

    return job


# ============================================================
# RESUME EXISTING RESULTS
# ============================================================

results = {}

if RESULT_FILE.exists():

    previous = pd.read_csv(
        RESULT_FILE
    )

    for _, r in previous.iterrows():

        if (
            str(r.get("status", ""))
            == "PASS"
        ):

            results[
                r["IL_ID"]
            ] = r.to_dict()

    print(
        f"Resume mode: "
        f"{len(results)} previous PASS results loaded."
    )

else:

    print(
        "Fresh production run."
    )


# ============================================================
# PENDING CANDIDATES
# ============================================================

pending = ils[
    ~ils["IL_ID"].isin(
        results.keys()
    )
].copy()

print()
print(
    "============================================================"
)

print(
    "STAGE 5B - FULL H2O SCREEN"
)

print(
    "============================================================"
)

print(
    "Total ILs        :",
    len(ils),
)

print(
    "Already complete :",
    len(results),
)

print(
    "Pending          :",
    len(pending),
)

print(
    "Parallel jobs    :",
    MAXJOBS,
)

print(
    "Batch size       :",
    BATCH_SIZE,
)

print(
    "Temperature      :",
    TEMPERATURE,
    "K",
)

print(
    "============================================================"
)


# ============================================================
# PLAMS
# ============================================================

init(
    path=str(OUT),
    folder="jobs",
)

config.default_jobrunner = (
    JobRunner(
        parallel=True,
        maxjobs=MAXJOBS,
    )
)

config.default_jobmanager.settings.hashing = (
    None
)

config.job.runscript.nproc = 1

config.log.stdout = 1


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint():

    if not results:
        return

    df = pd.DataFrame(
        list(
            results.values()
        )
    )

    order = {
        x: i
        for i, x in enumerate(
            ils["IL_ID"]
        )
    }

    df["_order"] = (
        df["IL_ID"]
        .map(order)
    )

    df = (
        df
        .sort_values("_order")
        .drop(
            columns=[
                "_order"
            ]
        )
    )

    temp = RESULT_FILE.with_suffix(
        ".tmp.csv"
    )

    df.to_csv(
        temp,
        index=False,
    )

    temp.replace(
        RESULT_FILE
    )


# ============================================================
# BATCH CALCULATIONS
# ============================================================

pending_records = (
    pending.to_dict(
        "records"
    )
)

total_pending = len(
    pending_records
)

for start in range(
    0,
    total_pending,
    BATCH_SIZE,
):

    batch = pending_records[
        start:
        start + BATCH_SIZE
    ]

    batch_number = (
        start // BATCH_SIZE
        + 1
    )

    print()
    print(
        "------------------------------------------------------------"
    )

    print(
        f"BATCH {batch_number}"
    )

    print(
        f"Candidates "
        f"{start+1} - "
        f"{min(start+len(batch), total_pending)} "
        f"of {total_pending}"
    )

    print(
        "------------------------------------------------------------"
    )

    launched = []

    # --------------------------------------------------------
    # SUBMIT
    # --------------------------------------------------------

    for row in batch:

        IL_ID = row[
            "IL_ID"
        ]

        try:

            job = make_job(
                row
            )

            out = job.run()

            launched.append(
                (
                    row,
                    job,
                    out,
                )
            )

        except Exception as exc:

            results[
                IL_ID
            ] = {
                "IL_ID": IL_ID,

                "cation_id":
                    row["cation_id"],

                "anion_id":
                    row["anion_id"],

                "stoichiometry":
                    row["stoichiometry"],

                "temperature_K":
                    TEMPERATURE,

                "gamma_inf_H2O":
                    np.nan,

                "ln_gamma_inf_H2O":
                    np.nan,

                "status":
                    "FAIL",

                "error":
                    f"LAUNCH: {exc}",
            }

    # --------------------------------------------------------
    # COLLECT
    # --------------------------------------------------------

    for row, job, out in launched:

        IL_ID = row[
            "IL_ID"
        ]

        try:

            res = (
                out.get_results()
            )

            gamma = float(
                res[
                    "gamma"
                ][-1][0]
            )

            if (
                not np.isfinite(
                    gamma
                )
                or
                gamma <= 0.0
            ):

                raise ValueError(
                    f"Invalid gamma: "
                    f"{gamma}"
                )

            lngamma = (
                math.log(
                    gamma
                )
            )

            results[
                IL_ID
            ] = {
                "IL_ID": IL_ID,

                "cation_id":
                    row["cation_id"],

                "anion_id":
                    row["anion_id"],

                "cation_abbreviation":
                    row[
                        "cation_abbreviation"
                    ],

                "anion_abbreviation":
                    row[
                        "anion_abbreviation"
                    ],

                "cation_count":
                    int(
                        row[
                            "cation_count"
                        ]
                    ),

                "anion_count":
                    int(
                        row[
                            "anion_count"
                        ]
                    ),

                "stoichiometry":
                    row[
                        "stoichiometry"
                    ],

                "temperature_K":
                    TEMPERATURE,

                "gamma_inf_H2O":
                    gamma,

                "ln_gamma_inf_H2O":
                    lngamma,

                "descriptor_source_warning":
                    bool(
                        row[
                            "contains_warning_ion"
                        ]
                    ),

                "status":
                    "PASS",

                "error":
                    "",
            }

        except Exception as exc:

            results[
                IL_ID
            ] = {
                "IL_ID": IL_ID,

                "cation_id":
                    row["cation_id"],

                "anion_id":
                    row["anion_id"],

                "stoichiometry":
                    row[
                        "stoichiometry"
                    ],

                "temperature_K":
                    TEMPERATURE,

                "gamma_inf_H2O":
                    np.nan,

                "ln_gamma_inf_H2O":
                    np.nan,

                "status":
                    "FAIL",

                "error":
                    str(exc),
            }

            print(
                f"FAIL {IL_ID}: "
                f"{exc}"
            )

    save_checkpoint()

    current_pass = sum(
        1
        for x in results.values()
        if x.get(
            "status"
        ) == "PASS"
    )

    current_fail = sum(
        1
        for x in results.values()
        if x.get(
            "status"
        ) == "FAIL"
    )

    print(
        f"Checkpoint: "
        f"PASS={current_pass}, "
        f"FAIL={current_fail}"
    )


finish()


# ============================================================
# FINAL RESULT TABLE
# ============================================================

save_checkpoint()

df = pd.read_csv(
    RESULT_FILE
)

# Include any failures not already written
# in case the last checkpoint was interrupted.
all_ids = set(
    ils["IL_ID"]
)

result_ids = set(
    df["IL_ID"]
)

missing_ids = sorted(
    all_ids
    - result_ids
)

pass_df = df[
    df["status"]
    == "PASS"
].copy()

fail_df = df[
    df["status"]
    != "PASS"
].copy()


# ============================================================
# PILOT REPRODUCIBILITY
# ============================================================

pilot_max_gamma_diff = np.nan

if PILOT_FILE.exists():

    pilot = pd.read_csv(
        PILOT_FILE
    )

    m = pilot.merge(
        pass_df[
            [
                "IL_ID",
                "gamma_inf_H2O",
            ]
        ],
        on="IL_ID",
        suffixes=(
            "_pilot",
            "_production",
        ),
    )

    if len(m):

        pilot_max_gamma_diff = float(
            np.max(
                np.abs(
                    m[
                        "gamma_inf_H2O_pilot"
                    ]
                    -
                    m[
                        "gamma_inf_H2O_production"
                    ]
                )
            )
        )


# ============================================================
# QC
# ============================================================

duplicates = int(
    df["IL_ID"]
    .duplicated()
    .sum()
)

invalid_positive = int(
    (
        ~np.isfinite(
            pass_df[
                "gamma_inf_H2O"
            ]
        )
        |
        (
            pass_df[
                "gamma_inf_H2O"
            ]
            <= 0.0
        )
    ).sum()
)

invalid_log = int(
    (
        ~np.isfinite(
            pass_df[
                "ln_gamma_inf_H2O"
            ]
        )
    ).sum()
)


if (
    len(df)
    == EXPECTED_ILS
    and
    len(pass_df)
    == EXPECTED_ILS
    and
    len(fail_df)
    == 0
    and
    len(missing_ids)
    == 0
    and
    duplicates == 0
    and
    invalid_positive == 0
    and
    invalid_log == 0
):

    overall = "PASS"

else:

    overall = "FAIL"


# ============================================================
# REPORT
# ============================================================

if len(pass_df):

    gamma_min = float(
        pass_df[
            "gamma_inf_H2O"
        ].min()
    )

    gamma_max = float(
        pass_df[
            "gamma_inf_H2O"
        ].max()
    )

    lngamma_min = float(
        pass_df[
            "ln_gamma_inf_H2O"
        ].min()
    )

    lngamma_max = float(
        pass_df[
            "ln_gamma_inf_H2O"
        ].max()
    )

else:

    gamma_min = np.nan
    gamma_max = np.nan
    lngamma_min = np.nan
    lngamma_max = np.nan


report = f"""
============================================================
STAGE 5 - H2O INFINITE-DILUTION SCREENING
============================================================

Overall status                  : {overall}

------------------------------------------------------------
CALCULATION
------------------------------------------------------------

Temperature                     : {TEMPERATURE} K
Expected ILs                    : {EXPECTED_ILS}
Results                         : {len(df)}

PASS                            : {len(pass_df)}
FAIL                            : {len(fail_df)}

Missing IL IDs                  : {len(missing_ids)}
Duplicate IL IDs                : {duplicates}

------------------------------------------------------------
STOICHIOMETRY COVERAGE
------------------------------------------------------------

1:1 PASS                        : {(pass_df["stoichiometry"] == "1:1").sum()}
2:1 PASS                        : {(pass_df["stoichiometry"] == "2:1").sum()}

------------------------------------------------------------
NUMERICAL QC
------------------------------------------------------------

Nonpositive/nonfinite gamma     : {invalid_positive}
Nonfinite ln(gamma)             : {invalid_log}

gamma_inf_H2O minimum           : {gamma_min}
gamma_inf_H2O maximum           : {gamma_max}

ln_gamma_inf_H2O minimum        : {lngamma_min}
ln_gamma_inf_H2O maximum        : {lngamma_max}

Pilot max |gamma difference|    : {pilot_max_gamma_diff}

------------------------------------------------------------
OUTPUT
------------------------------------------------------------

{RESULT_FILE}

------------------------------------------------------------
FAILURES
------------------------------------------------------------

{fail_df[["IL_ID","cation_id","anion_id","error"]].to_string(index=False) if len(fail_df) else "NONE"}

============================================================
"""

REPORT_FILE.write_text(
    report
)

print(report)

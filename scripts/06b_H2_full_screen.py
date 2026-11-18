from pathlib import Path
import os
import math
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

ROOT = Path(__file__).resolve().parents[1]

ION_FILE = ROOT / "01_database" / "ions_master.csv"

IL_FILE = (
    ROOT
    / "04_descriptors"
    / "ILs"
    / "IL_descriptors_master.csv"
)

PILOT_FILE = (
    ROOT
    / "06_H2"
    / "pilot"
    / "H2_pilot_results.csv"
)

OUT = (
    ROOT
    / "06_H2"
    / "production"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_FILE = (
    OUT
    / "H2_infinite_dilution_Henry_298K.csv"
)

REPORT_FILE = (
    OUT
    / "stage6_H2_validation.txt"
)

TEMPERATURE = 298.15
EXPECTED_ILS = 4617

MAXJOBS = 8
BATCH_SIZE = 200

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


H2 = Path(os.environ["COSMORS_SOLUTE_DB"]) / "Hydrogen.coskf"

if not H2.exists():
    raise RuntimeError(
        f"Hydrogen.coskf missing: {H2}"
    )


BASH = shutil.which("bash")

if not BASH:
    raise RuntimeError("bash not found")


def ion_path(ion_id):

    if ion_id in [
        "C060",
        "C071",
        "C081",
    ]:

        safe = (
            ROOT
            / "04_descriptors"
            / "recovery"
            / f"{ion_id}.coskf"
        )

        if safe.exists():
            return safe

    return Path(
        ions.loc[
            ion_id,
            "coskf_path"
        ]
    )


def to_msys_path(path):

    s = str(path)

    if len(s) >= 2 and s[1] == ":":
        return (
            "/"
            + s[0].lower()
            + s[2:].replace("\\", "/")
        )

    return s.replace("\\", "/")


def amsreport_numbers(
    crskf,
    key,
):

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
            f"amsreport failed for {key}: "
            f"{r.stderr}"
        )

    vals = []

    for line in r.stdout.splitlines():

        if line.strip().startswith(
            "Property"
        ):
            continue

        vals.extend(
            float(x)
            for x in re.findall(
                r"[-+]?(?:\d+\.\d+|\d+)"
                r"(?:[Ee][-+]?\d+)?",
                line
            )
        )

    return np.array(
        vals,
        dtype=float,
    )


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

    s = Settings()

    s.input.property._h = (
        "activitycoef"
    )

    s.input.method = "COSMORS"
    s.input.temperature = TEMPERATURE


    # ========================================================
    # Compound 1 = ionic liquid
    # ========================================================

    IL = Settings()

    form = [
        Settings()
    ]

    species = [
        Settings(),
        Settings(),
    ]

    cs = [
        Settings()
    ]

    cs[0]._h = str(
        cpath
    )

    cs[0].count = nc

    species[0] = cs


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


    # ========================================================
    # Compound 2 = H2
    # ========================================================

    solute = Settings()

    solute._h = str(
        H2
    )

    solute.name = "H2"
    solute.frac1 = 0.0


    s.input.compound = [
        IL,
        solute,
    ]


    if cid == "C071":

        s.input.REUSE_SIGMA_PROFILE = (
            "No"
        )

        s.input.UPDATE_SIGMA_PROFILE = (
            "No"
        )


    return CRSJob(
        settings=s,
        name=f"H2_{IL_ID}",
    )


# ============================================================
# RESUME
# ============================================================

results = {}

if RESULT_FILE.exists():

    previous = pd.read_csv(
        RESULT_FILE
    )

    for _, r in previous.iterrows():

        if str(
            r.get(
                "status",
                "",
            )
        ) == "PASS":

            results[
                r["IL_ID"]
            ] = r.to_dict()

    print(
        "Resume mode:",
        len(results),
        "previous PASS results loaded."
    )

else:

    print(
        "Fresh Stage 6 production run."
    )


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
    "STAGE 6B - FULL H2 SCREEN"
)
print(
    "============================================================"
)
print(
    "Total ILs        :",
    len(ils)
)
print(
    "Already complete :",
    len(results)
)
print(
    "Pending          :",
    len(pending)
)
print(
    "Parallel jobs    :",
    MAXJOBS
)
print(
    "Batch size       :",
    BATCH_SIZE
)
print(
    "Temperature      :",
    TEMPERATURE,
    "K"
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
# CHECKPOINT
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

    tmp = RESULT_FILE.with_suffix(
        ".tmp.csv"
    )

    df.to_csv(
        tmp,
        index=False,
    )

    tmp.replace(
        RESULT_FILE
    )


# ============================================================
# RUN BATCHES
# ============================================================

records = pending.to_dict(
    "records"
)

total_pending = len(
    records
)


for start in range(
    0,
    total_pending,
    BATCH_SIZE,
):

    batch = records[
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

                "IL_ID":
                    IL_ID,

                "cation_id":
                    row["cation_id"],

                "anion_id":
                    row["anion_id"],

                "stoichiometry":
                    row["stoichiometry"],

                "temperature_K":
                    TEMPERATURE,

                "gamma_inf_H2":
                    np.nan,

                "ln_gamma_inf_H2":
                    np.nan,

                "kHcp_H2_mol_L_atm":
                    np.nan,

                "kHcc_H2_dimensionless":
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


            # =================================================
            # gamma infinity
            # =================================================

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


            # =================================================
            # Henry constants
            # =================================================

            crskf = (
                Path(job.path)
                / f"{job.name}.crskf"
            )


            henry = (
                amsreport_numbers(
                    crskf,
                    "henry",
                )
            )


            henry_nodim = (
                amsreport_numbers(
                    crskf,
                    "henry-nodim",
                )
            )


            if len(henry) < 2:

                raise RuntimeError(
                    "Unexpected HENRY "
                    f"array: {henry}"
                )


            # Compound 2 = H2
            kHcp = float(
                henry[-1]
            )


            kHcc = (
                float(
                    henry_nodim[-1]
                )
                if len(
                    henry_nodim
                )
                else np.nan
            )


            if (
                not np.isfinite(
                    kHcp
                )
                or
                kHcp <= 0.0
            ):

                raise ValueError(
                    f"Invalid kHcp: "
                    f"{kHcp}"
                )


            results[
                IL_ID
            ] = {

                "IL_ID":
                    IL_ID,

                "cation_id":
                    row[
                        "cation_id"
                    ],

                "anion_id":
                    row[
                        "anion_id"
                    ],

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

                "gamma_inf_H2":
                    gamma,

                "ln_gamma_inf_H2":
                    lngamma,

                "kHcp_H2_mol_L_atm":
                    kHcp,

                "kHcc_H2_dimensionless":
                    kHcc,

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

                "IL_ID":
                    IL_ID,

                "cation_id":
                    row["cation_id"],

                "anion_id":
                    row["anion_id"],

                "stoichiometry":
                    row["stoichiometry"],

                "temperature_K":
                    TEMPERATURE,

                "gamma_inf_H2":
                    np.nan,

                "ln_gamma_inf_H2":
                    np.nan,

                "kHcp_H2_mol_L_atm":
                    np.nan,

                "kHcc_H2_dimensionless":
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
# FINAL TABLE
# ============================================================

save_checkpoint()

df = pd.read_csv(
    RESULT_FILE
)

all_ids = set(
    ils["IL_ID"]
)

result_ids = set(
    df["IL_ID"]
)

missing_ids = sorted(
    all_ids - result_ids
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

pilot_gamma_diff = np.nan
pilot_kH_diff = np.nan

if PILOT_FILE.exists():

    pilot = pd.read_csv(
        PILOT_FILE
    )

    m = pilot.merge(
        pass_df[
            [
                "IL_ID",
                "gamma_inf_H2",
                "kHcp_H2_mol_L_atm",
            ]
        ],
        on="IL_ID",
        suffixes=(
            "_pilot",
            "_production",
        ),
    )

    if len(m):

        pilot_gamma_diff = float(
            np.max(
                np.abs(
                    m[
                        "gamma_inf_H2_pilot"
                    ]
                    -
                    m[
                        "gamma_inf_H2_production"
                    ]
                )
            )
        )

        pilot_kH_diff = float(
            np.max(
                np.abs(
                    m[
                        "kHcp_H2_mol_L_atm_pilot"
                    ]
                    -
                    m[
                        "kHcp_H2_mol_L_atm_production"
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


bad_gamma = int(
    (
        ~np.isfinite(
            pass_df[
                "gamma_inf_H2"
            ]
        )
        |
        (
            pass_df[
                "gamma_inf_H2"
            ]
            <= 0.0
        )
    ).sum()
)


bad_log = int(
    (
        ~np.isfinite(
            pass_df[
                "ln_gamma_inf_H2"
            ]
        )
    ).sum()
)


bad_kH = int(
    (
        ~np.isfinite(
            pass_df[
                "kHcp_H2_mol_L_atm"
            ]
        )
        |
        (
            pass_df[
                "kHcp_H2_mol_L_atm"
            ]
            <= 0.0
        )
    ).sum()
)


if (
    len(df) == EXPECTED_ILS
    and
    len(pass_df) == EXPECTED_ILS
    and
    len(fail_df) == 0
    and
    len(missing_ids) == 0
    and
    duplicates == 0
    and
    bad_gamma == 0
    and
    bad_log == 0
    and
    bad_kH == 0
):

    overall = "PASS"

else:

    overall = "FAIL"


# ============================================================
# RANGES
# ============================================================

if len(pass_df):

    gamma_min = float(
        pass_df[
            "gamma_inf_H2"
        ].min()
    )

    gamma_max = float(
        pass_df[
            "gamma_inf_H2"
        ].max()
    )

    lngamma_min = float(
        pass_df[
            "ln_gamma_inf_H2"
        ].min()
    )

    lngamma_max = float(
        pass_df[
            "ln_gamma_inf_H2"
        ].max()
    )

    kH_min = float(
        pass_df[
            "kHcp_H2_mol_L_atm"
        ].min()
    )

    kH_max = float(
        pass_df[
            "kHcp_H2_mol_L_atm"
        ].max()
    )

else:

    gamma_min = np.nan
    gamma_max = np.nan
    lngamma_min = np.nan
    lngamma_max = np.nan
    kH_min = np.nan
    kH_max = np.nan


# ============================================================
# REPORT
# ============================================================

report = f"""
============================================================
STAGE 6 - H2 INFINITE-DILUTION + HENRY SCREENING
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

Invalid gamma_inf_H2            : {bad_gamma}
Invalid ln_gamma_inf_H2         : {bad_log}
Invalid kHcp                    : {bad_kH}

gamma_inf_H2 minimum            : {gamma_min}
gamma_inf_H2 maximum            : {gamma_max}

ln_gamma_inf_H2 minimum         : {lngamma_min}
ln_gamma_inf_H2 maximum         : {lngamma_max}

kHcp minimum [mol/L/atm]        : {kH_min}
kHcp maximum [mol/L/atm]        : {kH_max}

Pilot max |gamma difference|    : {pilot_gamma_diff}
Pilot max |kHcp difference|     : {pilot_kH_diff}

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

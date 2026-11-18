# COSMO-RS screening of 4,617 ionic-liquid candidates

Research data and computational workflow accompanying:

**Solute-Dependent Thermodynamics of Hydrogen and Water in 4,617 Ionic-Liquid Candidates: COSMO-RS Screening, Pareto Optimization, and Chemical-Space Analysis**

The library comprises 81 cations and 57 anions: 4,536 charge-balanced 1:1 formulations and 81 charge-balanced 2:1 formulations. Hydrogen and water were evaluated independently at infinite dilution at 298.15 K using AMS COSMO-RS 2026.103 (ADF Lei 2018 parameterization).

## Repository contents

- `07_master_table/IL_master_table_4617.csv`: complete numerical screening dataset, including ionic identities, COSMO-derived descriptors, thermodynamic responses and quality-control flags.
- `scripts/`: Python workflow for descriptor generation, COSMO-RS screening, data assembly, quality control, statistical analysis, Pareto optimization, PCA, K-means clustering and Henry-representation analysis.
- `requirements.txt`: open-source Python dependencies for analysis from the master dataset.

## Reproduce the reported numerical analysis

The following workflow uses the supplied master dataset; no AMS license is needed. Tested with Python 3.12 and the dependency versions in `requirements.txt`. Run from the repository root.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/08_final_global_QC.py
python scripts/09a_statistical_analysis.py
python scripts/10a_pareto_screening.py
python scripts/10b_pareto_tradeoff_analysis.py
python scripts/11a_PCA_chemical_space.py
python scripts/11b_cluster_PCA_space.py
python scripts/12a_final_shortlist.py
python scripts/13a_Henry_representation_analysis.py
```

The scripts write numerical CSV results and validation summaries to their corresponding stage directories. These generated files are not required as inputs to the initial analysis and are excluded from version control. Standalone manuscript-figure scripts and image exports are not included; figure export in the retained mixed-purpose analysis scripts is disabled by default.

### Expected numerical checks

- 4,617 unique formulations; 4,536 monovalent-anion formulations and 81 dianion-containing formulations.
- Final QC: no failed formulations; 57 rows retain the documented C071 ion-level warning.
- Water activity-coefficient variance: 84.85% anion contribution.
- Hydrogen activity-coefficient variance: 53.53% cation contribution.
- Concentration-based hydrogen Henry coefficient (log scale): 50.37% anion contribution.
- Nine primary Pareto-optimal candidates.
- First three PCs retain approximately 94.85% variance; silhouette-based K-means selection yields three clusters.
- Within the balanced 1:1 subset, IL4443 ranks 4,479th by descending concentration-based Henry coefficient and fifth by ascending reconstructed mole-fraction-based Henry constant. IL3808 ranks first under both conventions.

The Henry representation script reconstructs the pressure/mole-fraction Henry constant using `Hpx = 1 / (kHcp * Vm)`, where `Vm` is the estimated COSMO-based molar volume in L/mol. Factorial covariance decomposition is performed on logarithmic quantities for the 81 × 56 balanced subset. These reconstructed values are not independent experimental measurements.

## Dataset column conventions

Machine-readable column names are intentionally retained to preserve compatibility with the scripts.

| Column | Meaning | Unit |
|---|---|---|
| IL_ID | Unique candidate identifier | — |
| cation_count, anion_count | Formula-unit ion counts | — |
| A_IL_A2 | COSMO surface area | Å² |
| V_IL_A3 | COSMO volume per formula unit | Å³ |
| MOM2_A, MOM3_A | Area-normalized σ-profile moments | Derived from COSMO charge-density moments |
| HBACC2_A, HBDON2_A | Area-normalized hydrogen-bond descriptors | Derived from resolved σ-profiles |
| gamma_inf_H2O, gamma_inf_H2 | Infinite-dilution activity coefficients | Dimensionless |
| ln_gamma_inf_H2O, ln_gamma_inf_H2 | Natural logarithms of activity coefficients | Dimensionless |
| kHcp_H2_mol_L_atm | Concentration/pressure Henry coefficient for H₂ | mol L⁻¹ atm⁻¹ |
| kHcc_H2_dimensionless | Dimensionless concentration/concentration Henry coefficient for H₂ | Dimensionless |
| contains_warning_ion, notes, master_status | Quality-control provenance and status | — |

## Upstream COSMO-RS workflow

Stages 04b–07 are supplied as methodological source code. Full regeneration requires a licensed AMS 2026.103 Linux-compatible environment (including PLAMS, the command-line COSMO-RS utilities, the applicable ion database and the licensed COSMO files). These proprietary files are not distributed here. Upstream inputs are not all part of the initial repository package.

The original workflow uses these relative paths for numerical inputs:

| Stage | Input(s) |
|---|---|
| 04b | `01_database/ions_master.csv`, with locally valid COSKF paths; historical recovery files for finalization |
| 04c | `02_candidates/IL_candidates_master.csv`, finalized ion descriptors and ion σ-profiles under `04_descriptors/ions/` |
| 05b / 06b | `01_database/ions_master.csv`, `04_descriptors/ILs/IL_descriptors_master.csv`, licensed solute and ion COSKF files |
| 07 | IL descriptor master and full H₂O/H₂ production CSV results |

Before using stages 05b and 06b set the environment variable `COSMORS_SOLUTE_DB` to your local directory containing `Water.coskf` and `Hydrogen.coskf`. Ensure all ion-file paths point to valid files. The original workflow contains special handling for C060, C071 and C081; stage 04b finalization requires the original recovery outputs and conversion inputs. The full licensed screening has not been rerun as part of preparing this repository.

Where available, the intermediate numerical CSVs can be retained at their original relative paths to support upstream data assembly. Do not redistribute licensed binary databases or personal absolute file paths.

## Analysis scope

The PCA and K-means stages provide unsupervised chemical-space analysis. Pareto optimization uses independently calculated H₂O and H₂ infinite-dilution responses. This repository does not claim a supervised predictive model or simultaneous finite-composition H₂/H₂O simulations.

# COSMO-RS Screening of 4,617 Ionic Liquids



**Solute-Dependent Thermodynamics of Hydrogen and Water in 4,617 Ionic-Liquid Candidates: COSMO-RS Screening, Pareto Optimization, and Chemical-Space Analysis**
<img width="1536" height="1024" alt="Computational Workflow and Ionic Liquid Example (2)" src="https://github.com/user-attachments/assets/d43f0aad-20aa-4573-a336-69263fedd252" />

## Overview

The dataset comprises 4,617 ionic-liquid candidates constructed from 81 cations and 57 anions. Thermodynamic properties were calculated at infinite dilution and 298.15 K using AMS COSMO-RS 2026.103 (ADF Lei 2018).

The principal thermodynamic quantities are:

- Water activity coefficient: $\ln \gamma^{\infty}(\mathrm{H_2O})$
- Hydrogen activity coefficient: $\ln \gamma^{\infty}(\mathrm{H_2})$
- Hydrogen Henry coefficient: $k_H^{cp}(\mathrm{H_2})$

The reconstructed mole-fraction-based Henry constant is expressed as:

$$
H^{px}(\mathrm{H_2})=\frac{1}{k_H^{cp}(\mathrm{H_2})V_m}
$$

## Repository Contents

- `07_master_table/`  Complete dataset of 4,617 ionic liquids.
- `scripts/`  COSMO-RS screening, statistical analysis, variance decomposition, Pareto optimization, PCA, K-means clustering, and Henry representation analysis.
- `requirements.txt`  Python dependencies.

## Reproducibility

Install the required Python packages:

~~~bash
pip install -r requirements.txt
~~~

Execute scripts 08–13 in numerical order to reproduce the statistical and chemical-space analyses from the supplied dataset.

Regeneration of the original COSMO-RS calculations requires a licensed AMS installation and the corresponding input files.

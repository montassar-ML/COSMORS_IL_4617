"""Compare H2 Henry representations and their factorial contributions."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "07_master_table" / "IL_master_table_4617.csv"
OUT = ROOT / "13_henry_representation"
AVOGADRO = 6.02214076e23


def effects(matrix):
    """Orthogonal marginal and nonadditive effects of a balanced design."""
    grand = matrix.mean()
    cation = matrix.mean(axis=1, keepdims=True) - grand
    anion = matrix.mean(axis=0, keepdims=True) - grand
    interaction = matrix - grand - cation - anion
    return [np.broadcast_to(cation, matrix.shape),
            np.broadcast_to(anion, matrix.shape), interaction]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MASTER)
    assert len(df) == 4617 and df.IL_ID.is_unique
    d = df.loc[df.stoichiometry.eq("1:1")].copy()
    assert len(d) == 4536
    assert d.cation_id.nunique() == 81 and d.anion_id.nunique() == 56
    assert not d.duplicated(["cation_id", "anion_id"]).any()
    k = "kHcp_H2_mol_L_atm"
    v = "Vm_L_mol"
    h = "Hpx_reconstructed_atm"
    # 1 Å³ = 1e-27 L; COSMO molecular volume -> estimated molar volume.
    for frame in (df, d):
        assert np.isfinite(frame[[k, "V_IL_A3"]]).all().all()
        assert (frame[[k, "V_IL_A3"]] > 0).all().all()
        frame[v] = frame.V_IL_A3 * AVOGADRO * 1e-27
        frame[h] = 1 / (frame[k] * frame[v])
    # Higher kHcp and lower Hpx indicate more favorable H2 solubility
    # under their respective concentration and mole-fraction conventions.
    d["rank_kHcp"] = d[k].rank(ascending=False, method="min").astype(int)
    d["rank_Hpx"] = d[h].rank(ascending=True, method="min").astype(int)
    d["rank_difference_Hpx_minus_kHcp"] = d.rank_Hpx - d.rank_kHcp
    cols = ["IL_ID", "cation_id", "anion_id", "cation_abbreviation",
            "anion_abbreviation", "V_IL_A3", v, k, h,
            "rank_kHcp", "rank_Hpx", "rank_difference_Hpx_minus_kHcp"]
    ranks = d[cols].sort_values(["rank_kHcp", "IL_ID"])
    ranks.to_csv(OUT / "henry_rankings_4536.csv", index=False)
    ranks.loc[(ranks.rank_kHcp <= 5) | (ranks.rank_Hpx <= 5)].to_csv(
        OUT / "Table_S5_rank_comparison.csv", index=False)

    # Eq. (3): log10(kHcp) = -log10(Hpx) - log10(Vm),
    # using numerical values in mol L^-1 atm^-1, atm, and L mol^-1.
    d["U"] = -np.log10(d[h])
    d["B"] = -np.log10(d[v])
    d["Y"] = np.log10(d[k])
    def matrix(col):
        return d.pivot(index="cation_id", columns="anion_id", values=col).to_numpy()
    U, B, Y = map(matrix, ["U", "B", "Y"])
    np.testing.assert_allclose(U + B, Y, atol=1e-12)
    names = ["Cation", "Anion", "Nonadditive"]
    rows = []
    fractions = []
    for name, u, b, y in zip(names, effects(U), effects(B), effects(Y)):
        row = {"component": name, "var_U": np.mean(u*u), "var_B": np.mean(b*b),
               "twice_cov_U_B": 2*np.mean(u*b), "var_Y": np.mean(y*y)}
        np.testing.assert_allclose(row["var_U"]+row["var_B"]+row["twice_cov_U_B"],
                                   row["var_Y"], atol=1e-12)
        rows.append(row)
        fractions.append({"component": name,
                          "Hpx_log_variance_percent": 100*np.mean(u*u)/U.var(),
                          "Vm_log_variance_percent": 100*np.mean(b*b)/B.var(),
                          "kHcp_log_variance_percent": 100*np.mean(y*y)/Y.var()})
    covariance = pd.DataFrame(rows)
    covariance.loc[len(covariance)] = ["Total", U.var(), B.var(),
                                        2*np.mean((U-U.mean())*(B-B.mean())), Y.var()]
    # Raw log10 variances; multiply by 1000 for the manuscript's scaled table.
    covariance.to_csv(OUT / "Table_S4_factorial_covariance.csv", index=False)
    pd.DataFrame(fractions).to_csv(OUT / "henry_factorial_fractions.csv", index=False)
    correlations = []
    for name, frame in [("all_4617", df), ("monovalent_4536", d)]:
        for a, b in [("V_IL_A3", k), ("V_IL_A3", h), (k, h)]:
            correlations.append({"subset": name, "variable_1": a, "variable_2": b,
                                 "pearson": frame[a].corr(frame[b]),
                                 "spearman": frame[a].corr(frame[b], method="spearman")})
    pd.DataFrame(correlations).to_csv(OUT / "henry_correlations.csv", index=False)
    print(ranks.loc[ranks.IL_ID.isin(["IL3808", "IL4443"]),
                    ["IL_ID", "rank_kHcp", "rank_Hpx"]].to_string(index=False))
    print(covariance.to_string(index=False))


if __name__ == "__main__":
    main()

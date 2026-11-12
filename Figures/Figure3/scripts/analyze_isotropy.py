"""Export Figure 3 measurements using the shared fixed-coordinate sampler."""
import pandas as pd
from DEPICT.figure_reproduction.lipid_reference import measure


def build_tables(root, config, panel):
    output = root / ("panel_" + panel["label"])
    output.mkdir(exist_ok=True)
    table = measure(root, config, panel["cells"])
    table.to_csv(output / "fwhm_measurements.csv", index=False)
    valid = table.loc[table.valid_xyz].copy()
    valid.to_csv(output / "boxplot_values.csv", index=False)
    summaries = []
    for (cell, method), group in table.groupby(["cell", "method"], sort=False):
        values = group.loc[group.valid_xyz, "axial_lateral_ratio"]
        summaries.append(dict(cell=cell, method=method, n_common_droplets=len(group),
                              n_valid=len(values), n_invalid=len(group)-len(values),
                              median=values.median(), mean=values.mean(),
                              q25=values.quantile(.25), q75=values.quantile(.75)))
    pd.DataFrame(summaries).to_csv(output / "sample_sizes_and_summary.csv", index=False)
    return valid

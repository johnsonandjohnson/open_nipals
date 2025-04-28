"""
Setup file for open_nipals.
make a Hotelling's T^2 vs- DModX plot
with the simulated data

conda env for execution can be constructed using the requirements.txt
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from open_nipals.nipalsPCA import NipalsPCA

# Load Data
datapath = Path(__file__).parents[2].joinpath("data")
savepath = Path(__file__).parents[0]

pca_dat = pd.read_excel(
    datapath.joinpath("PCATestData.xlsx"), header=None, engine="openpyxl"
)
input_array = pca_dat.to_numpy()

# preprocess data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(input_array)

# fit NipalsPCA object, transform data
model = NipalsPCA().fit(scaled_data)
transformed_data = model.transform(scaled_data)

imd = model.calc_imd(input_scores=model.fit_scores, metric="HotellingT2")
oomd = model.calc_oomd(scaled_data, metric="DModX")

# calculate limits
imd_lmt = model.calc_limit(metric="HotellingT2", alpha=0.95)
oomd_lmt = model.calc_limit(metric="DModX", alpha=0.95)

# plot points
plt.scatter(imd, oomd, s=5, color="blue", label="simulated data")

plt.xlim([0, 8])
plt.ylim([0, 1.5])
plt.xlabel("in-model distance (" + r"$Hotelling's$ $T^2$" + ")")
plt.ylabel("out-of model distance (" + r"$DModX$" + ")")

# plot limits
plt.vlines(
    x=imd_lmt,
    ymin=0,
    ymax=oomd_lmt,
    colors="black",
    linestyles="dashed",
    label="0.95 confidence interval",
)
plt.hlines(
    y=oomd_lmt, xmin=0, xmax=imd_lmt, colors="black", linestyles="dashed"
)

plt.legend()

plt.savefig(savepath.joinpath("HT2_DModX_example_plot.png"), dpi=600)

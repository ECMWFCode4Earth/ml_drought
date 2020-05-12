from pathlib import Path
import sys

sys.path.append("..")

from src.analysis import EventDetector

data_dir = Path("data")
chirps_dir = data_dir / "interim" / "chirps_preprocessed" / "chirps_kenya.nc"

e = EventDetector(chirps_dir)

variable = "precip"
hilo = "low"
method = "std"

e.detect(variable=variable, time_period="dayofyear", hilo=hilo, method=method)

# plot the output
exceed = e.reapply_mask_to_boolean_xarray("precip", e.exceedences)

fig, ax = plt.subplots()

df = exceed.sum(dim=["lat", "lon"]).to_dataframe()
df.plot(ax=ax, label=f"{variable}_{hilo}_{method}", legend=True)
df.rolling(24).mean().plot(ax=ax, label="24mth mean")

ax.set_title(
    f'Number of Pixels with {variable} \
    {"below" if hilo == "low" else "above"} {method}'
)

fig.savefig(".png")

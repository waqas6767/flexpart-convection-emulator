"""Download the ERA5 columns used in Phase 1 (run in Google Colab or any Python environment).

- Training/validation/test: tropical Indian Ocean (10S-10N, 60-100E), 2019-2023 -> era5_phase1/
- Held-out region:          tropical Atlantic (10S-10N, 40W-0), 2023          -> era5_heldout/

Each file: 4 months x 4 days x 2 times, 32 pressure levels (1000-10 hPa), 1 degree grid (~27,500 columns).

Requirements: a Copernicus Climate Data Store account, the ERA5 pressure-level licence accepted on the
dataset's Download tab, and `pip install cdsapi`. The API key is asked for interactively and never stored
in this file.

Copernicus allows only ONE queued request per dataset at a time, so years are requested sequentially.
Re-running the script skips files that already exist.
"""
import os
import sys
from getpass import getpass

import cdsapi

BASE = sys.argv[1] if len(sys.argv) > 1 else (
    "/content/drive/MyDrive" if os.path.isdir("/content/drive/MyDrive") else os.getcwd())

LEVELS = ["1000", "975", "950", "925", "900", "875", "850", "825", "800", "775", "750",
          "700", "650", "600", "550", "500", "450", "400", "350", "300", "250",
          "225", "200", "175", "150", "125", "100", "70", "50", "30", "20", "10"]

JOBS = [(f"{BASE}/era5_phase1/era5_{y}.nc", y, [10, 60, -10, 100]) for y in ["2019", "2020", "2021", "2022", "2023"]]
JOBS.append((f"{BASE}/era5_heldout/era5_atlantic_2023.nc", "2023", [10, -40, -10, 0]))


def ensure_key():
    rc = os.path.expanduser("~/.cdsapirc")
    if not os.path.exists(rc):
        key = getpass("Paste your Copernicus API key and press Enter: ")
        with open(rc, "w") as f:
            f.write("url: https://cds.climate.copernicus.eu/api\n")
            f.write(f"key: {key.strip()}\n")


def main():
    ensure_key()
    client = cdsapi.Client()
    for out, year, area in JOBS:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out):
            print(f"{out}: already downloaded")
            continue
        print(f"{out}: requesting (may wait in the Copernicus queue) ...", flush=True)
        client.retrieve("reanalysis-era5-pressure-levels", {
            "product_type": ["reanalysis"],
            "variable": ["temperature", "specific_humidity"],
            "year": [year], "month": ["01", "04", "07", "10"], "day": ["01", "08", "15", "22"],
            "time": ["00:00", "12:00"], "pressure_level": LEVELS,
            "data_format": "netcdf", "download_format": "unarchived",
            "area": area, "grid": [1.0, 1.0],
        }, out)
        print(f"{out}: saved ({os.path.getsize(out) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()

"""Download and extract the official UCI Tetouan City power-consumption dataset."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests

URL = "https://archive.ics.uci.edu/static/public/849/power%2Bconsumption%2Bof%2Btetouan%2Bcity.zip"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(OUTPUT_DIR)
    print(f"Dataset extracted to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

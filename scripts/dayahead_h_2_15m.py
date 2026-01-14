#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert hourly Day-Ahead prices to 15-minute resolution (piecewise constant).

Input:
  HTW_V2H_WS25/data/dayahead_2025_hourly.csv
  (Format example)
    Datum (MEZ);"Day Ahead Auktion (DE-LU) ""Preis (EUR/MWh, EUR/tCO2)"
    01.01.2025 00:00;2.16
    01.01.2025 01:00;1.6
    ...

Output:
  HTW_V2H_WS25/data/dayahead_2025_15min.csv

Key points:
- Prices are *not* interpolated. Each hourly price is copied to 4 quarter-hours (ffill after resample).
- DST is handled for Europe/Berlin:
  - ambiguous (fall-back hour) handled via ambiguous="infer"
  - nonexistent (spring-forward hour) handled via nonexistent="shift_forward"
- Optional: write output either in Europe/Berlin or in UTC.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_hourly_prices(input_file: Path) -> pd.DataFrame:
    """Read the hourly price CSV exported from ENTSO-E/EPEX-like sources."""
    df = pd.read_csv(
        input_file,
        sep=";",
        header=0,
        names=["time", "price"],
        encoding="utf-8",
    )

    # Parse timestamps in German day-first format like "01.01.2025 00:00"
    df["time"] = pd.to_datetime(df["time"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()

    # Ensure numeric prices (comma decimals would be handled by replace if needed)
    if df["price"].dtype == object:
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Drop rows where price is missing after conversion
    df = df.dropna(subset=["price"])

    return df


def hourly_to_15min(
    df_hourly: pd.DataFrame,
    output_tz: str = "Europe/Berlin",
) -> pd.DataFrame:
    """
    Convert hourly prices to 15-minute prices (piecewise constant) and handle DST.

    Parameters
    ----------
    df_hourly:
        DataFrame indexed by naive local timestamps representing delivery hour starts.
    output_tz:
        "Europe/Berlin" or "UTC".
    """
    # Localize the *naive* timestamps to Europe/Berlin with DST handling
    # - ambiguous="infer": resolves 02:00 during fall-back if the series is ordered
    # - nonexistent="shift_forward": resolves the missing hour during spring-forward
    df_local = df_hourly.copy()
    df_local.index = df_local.index.tz_localize(
        "Europe/Berlin",
        ambiguous="infer",
        nonexistent="shift_forward",
    )

    # Resample to 15-min and forward-fill (constant within the hour)
    df_15 = df_local.resample("15min").ffill()

    # Convert timezone if desired
    if output_tz.upper() == "UTC":
        df_15 = df_15.tz_convert("UTC")
    elif output_tz != "Europe/Berlin":
        df_15 = df_15.tz_convert(output_tz)

    return df_15


def main() -> None:
    # Paths relative to this script:
    # HTW_V2H_WS25/
    #   scripts/dayahead_h_2_15m.py  (this file)
    #   data/dayahead_2025_hourly.csv
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data"

    input_file = data_dir / "dayahead_2025_hourly.csv"
    output_file = data_dir / "dayahead_2025_15min.csv"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            f"Expected location: {project_root / 'data' / 'dayahead_2025_hourly.csv'}"
        )

    print(f"Reading input file: {input_file}")

    df_hourly = read_hourly_prices(input_file)

    # Choose output timezone:
    # - "Europe/Berlin" keeps local time with DST
    # - "UTC" avoids DST entirely (often easiest for modelling)
    OUTPUT_TZ = "Europe/Berlin"  # change to "UTC" if preferred

    df_15 = hourly_to_15min(df_hourly, output_tz=OUTPUT_TZ)

    # Export
    df_15.to_csv(output_file, sep=";", float_format="%.2f", encoding="utf-8")
    print(f"Exported to: {output_file}")

    # Quick sanity checks
    print("\nSample (2025-01-01 14:00–15:00):")
    # When tz-aware, slicing should include timezone; pandas accepts naive string as local in most cases.
    print(df_15.loc["2025-01-01 14:00":"2025-01-01 15:00"])

    # Check step size distribution (DST days will show some irregularities, that is expected)
    diffs = df_15.index.to_series().diff().value_counts().head(5)
    print("\nIndex step size distribution (top 5):")
    print(diffs)


if __name__ == "__main__":
    main()

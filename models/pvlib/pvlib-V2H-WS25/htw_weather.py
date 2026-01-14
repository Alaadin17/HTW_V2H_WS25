#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare HTW weather data for pvlib.

Expected input (15-min, comma-separated):
time, ghi, dhi, t_luft, v_wind
"""

import pandas as pd
import pvlib

from config import PATH_HTW_WEATHER, HTW_LAT, HTW_LON


def load_htw_weather_15min():
    """
    Loads the HTW weather CSV (15-min) and returns a DataFrame with:
    columns: ghi, dhi, temp_air, wind_speed
    index: DatetimeIndex (naive timestamps)
    """
    df = pd.read_csv(PATH_HTW_WEATHER, sep=",")

    # Parse time and set index
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    # Keep / rename columns to pvlib standard
    df = df.rename(columns={
        "t_luft": "temp_air",
        "v_wind": "wind_speed",
    })

    # Ensure numeric
    for c in ["ghi", "dhi", "temp_air", "wind_speed"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Basic cleanup
    df["ghi"] = df["ghi"].clip(lower=0)
    df["dhi"] = df["dhi"].clip(lower=0)

    return df[["ghi", "dhi", "temp_air", "wind_speed"]]


def add_dni_from_ghi_dhi(df_weather):
    """
    Adds 'dni' derived from measured ghi & dhi.

    Uses solar zenith computed for HTW_LAT/HTW_LON.
    """
    times = df_weather.index

    # Solar position (zenith)
    solpos = pvlib.solarposition.get_solarposition(times, HTW_LAT, HTW_LON)

    # DNI from GHI + DHI + zenith
    dni = pvlib.irradiance.dni(
        ghi=df_weather["ghi"],
        dhi=df_weather["dhi"],
        zenith=solpos["zenith"],
    )

    df_out = df_weather.copy()
    df_out["dni"] = dni.fillna(0).clip(lower=0)

    return df_out

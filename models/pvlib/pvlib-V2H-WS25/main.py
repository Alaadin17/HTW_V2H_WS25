#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script: PV-Ertragssimulation für das V2H-System
mit Jinko Tiger Neo 440 W und Sunny Tripower 10.0.

Orientiert am PV3-HTW-Beispiel, aber reduziert auf ein Array.
"""

import calendar as cal

import pvlib
import pandas as pd
import matplotlib.pyplot as plt

from config import HTW_LON, HTW_LAT, PATH_HTW_WEATHER, PATH_FRED_WEATHER, PATH_RESULTS, PATH_PLOTS
import modules          #  Jinko-Modul
import v2h_inverter     #  Sunny Tripower
import htw_weather


def setup_model(name, system, location):
    """
    Wrapper für pvlib.modelchain.ModelChain.
    """
    return pvlib.modelchain.ModelChain(
        system=system,
        location=location,
        aoi_model="physical",
        spectral_model="no_loss",
        losses_model="pvwatts",
        name=name,
    )


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Standort
    # ------------------------------------------------------------------
    htw_location = pvlib.location.Location(
        name="HTW Berlin",
        latitude=HTW_LAT,
        longitude=HTW_LON,
        tz="Europe/Berlin",
        altitude=80,
    )

    # ------------------------------------------------------------------
    # Anlagengeometrie
    # ------------------------------------------------------------------
    # Tilt: Winkel von horizontal (0° = flach, 90° = senkrecht)
    surface_tilt = 30.0          # TODO: an dein Dach anpassen
    # Azimut: Nord=0, Ost=90, Süd=180, West=270
    surface_azimuth = 180.0      # TODO: an dein Dach anpassen

    albedo = 0.2

    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[
        "sapm"
    ]["open_rack_glass_polymer"]

    # PVWatts-Verlustannahmen (kannst du später feinjustieren)
    pvwatts_losses = {
        "soiling": 2,
        "shading": 3,
        "snow": 0,
        "mismatch": 2,
        "wiring": 2,
        "connections": 0.5,
        "lid": 1.5,
        "nameplate_rating": 1,
        "age": 0,
        "availability": 3,
    }

    # ------------------------------------------------------------------
    # Generator + Wechselrichter (ein Array, WR1)
    # ------------------------------------------------------------------
    module_name = "Jinko_TigerNeo_440"
    module_parameters = modules.modul1()          # Series mit CEC-Parametern

    inverter_name = "Sunny_Tripower_10"
    inverter_parameters =  v2h_inverter.inv1()    # Sandia-Parameter des STP

    # Stringauslegung: 12 Module in Serie, 2 Strings parallel
    modules_per_string = 12
    strings_per_inverter = 2

    # Optional: installierte Generatorleistung ausgeben
    p_dc_stc = (
        module_parameters["STC"] * modules_per_string * strings_per_inverter / 1000.0
    )
    print(f"Installierte PV-Leistung (STC): {p_dc_stc:.2f} kWp")

    wr1 = pvlib.pvsystem.PVSystem(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        albedo=albedo,
        module=module_name,
        module_type="glass_polymer",
        module_parameters=module_parameters,
        temperature_model_parameters=temperature_model_parameters,
        modules_per_string=modules_per_string,
        strings_per_inverter=strings_per_inverter,
        inverter=inverter_name,
        inverter_parameters=inverter_parameters,
        racking_model="close_mount",
        losses_parameters=pvwatts_losses,
        name="wr1",
    )

    # ModelChain anlegen
    mc1 = setup_model(wr1.name, wr1, htw_location)
    models = [mc1]

    # ------------------------------------------------------------------
    # Wetterdaten einlesen und aufbereiten
    # ------------------------------------------------------------------
    df_htw = pd.read_csv(PATH_HTW_WEATHER, sep=";")   # MView-Datei
    df_fred = pd.read_csv(PATH_FRED_WEATHER, sep=",")

    # Spalten für HTW-Daten anpassen
    df_htw = htw_weather.convert_column_names(
        df_htw, time="timestamp", ghi="g_hor_si", wind_speed="v_wind", temp_air="t_luft"
    )
    # FRED-Daten: Spaltennamen übernehmen
    df_fred = htw_weather.convert_column_names(
        df_fred, time="time", ghi="ghi", wind_speed="wind_speed", temp_air="temp_air"
    )

    # Diffuse und DNI für HTW berechnen
    df_htw = htw_weather.calculate_diffuse_irradiation(
        df_htw, parameter_name="ghi", lat=HTW_LAT, lon=HTW_LON
    )

    # Auf Stundendaten resamplen
    weather_htw = df_htw[["ghi", "dni", "dhi"]].resample("h").mean()
    weather_fred = df_fred.resample("h").mean()
    weather_fred = weather_fred[weather_fred.index.year > 2014]

    # ------------------------------------------------------------------
    # Simulation mit HTW-Wetter
    # ------------------------------------------------------------------
    for mc in models:
        mc.run_model(weather=weather_htw)

    result_monthly_htw = pd.DataFrame()
    for mc in models:
        result_monthly_htw[mc.name] = (
            mc.results.ac.resample("ME").sum() / 1000.0
        ).round(1)  # kWh

    result_monthly_htw.index = [cal.month_name[i] for i in range(1, 13)]

    result_annual_htw = pd.DataFrame({"annual_yield": result_monthly_htw.sum()})

    print("#" * 50)
    print(f"{' Execution successful! (HTW) ':^50}")
    print("#" * 50, "\n")
    print(f"{' Results Monthly HTW ':#^50}")
    print(result_monthly_htw, "\n")
    print(f"{' Results Annual HTW ':#^50}")
    print(result_annual_htw, "\n")
    print(f"{' Results Total HTW ':#^50}")
    print(result_annual_htw.sum())

    (result_monthly_htw).to_csv(
        PATH_RESULTS / "results_monthly_htw.csv", sep=";", encoding="utf-8"
    )


    ax = result_monthly_htw.plot.bar(
        rot=90, title="Monthly PV yield (HTW weather)", ylabel="Energy in kWh", grid=True
    )
    plt.tight_layout()
    plt.savefig(PATH_PLOTS / "results_monthly_htw.svg")

    # ------------------------------------------------------------------
    # Simulation mit FRED-Wetter
    # ------------------------------------------------------------------
    for mc in models:
        mc.run_model(weather=weather_fred)

    result_monthly_fred = pd.DataFrame()
    for mc in models:
        result_monthly_fred[mc.name] = (
            mc.results.ac.resample("ME").sum() / 1000.0
        ).round(1)

    result_monthly_fred.index = [cal.month_name[i] for i in range(1, 13)]

    result_annual_fred = pd.DataFrame({"annual_yield": result_monthly_fred.sum()})

    print("#" * 50)
    print(f"{' Execution successful! (FRED) ':^50}")
    print("#" * 50, "\n")
    print(f"{' Results Monthly FRED ':#^50}")
    print(result_monthly_fred, "\n")
    print(f"{' Results Annual FRED ':#^50}")
    print(result_annual_fred, "\n")
    print(f"{' Results Total FRED ':#^50}")
    print(result_annual_fred.sum())

    result_monthly_fred.to_csv(
        PATH_RESULTS / "results_monthly_htw.csv", sep=";", encoding="utf-8"
    )

    ax = result_monthly_fred.plot.bar(
        rot=90,
        title="Monthly PV yield (FRED weather)",
        ylabel="Energy in kWh",
        grid=True,
    )
    plt.tight_layout()
    plt.savefig(PATH_PLOTS / "results_monthly_fred.svg")


# ======================================================================
# 15-Minuten-Zeitreihe der PV-Leistung (HTW-Wetter)
# ======================================================================

# Wetterdaten auf 15-min-Raster bringen
# Falls df_htw bereits 15-minütig ist, ändert sich dadurch fast nichts.
weather_htw_15min = (
    df_htw[["ghi", "dni", "dhi"]]
    .resample("15min")
    .interpolate()
)

# Neue ModelChain, damit wir die bisherigen Ergebnisse nicht überschreiben
mc1_15 = setup_model("wr1_15min_htw", wr1, htw_location)
mc1_15.run_model(weather=weather_htw_15min)

# AC-Leistung in kW (pvlib liefert W)
pv_ac_15min_kw = (mc1_15.results.ac / 1000.0).rename("P_AC_kW")

# DataFrame bauen
pv_ts_15min = pd.DataFrame(pv_ac_15min_kw)
pv_ts_15min.index.name = "time"

# Als CSV speichern
ts15_path = PATH_RESULTS / "pv_timeseries_htw_15min.csv"
pv_ts_15min.to_csv(ts15_path, sep=";", encoding="utf-8")

print(f"15-min PV-Zeitreihe (HTW) gespeichert unter: {ts15_path}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script: PV-Ertragssimulation für das V2H-System
mit Jinko Tiger Neo 440 W und Sunny Tripower 10.0.

Orientiert am PV3-HTW-Beispiel, aber reduziert auf:
- eine Wetterquelle (HTW-Daten)
- eine zeitliche Auflösung (15 Minuten)
- eine ModelChain / ein Array.

Outputs:
- pv_timeseries_15min.csv
- results_monthly.csv
- results_monthly.svg
"""

import calendar as cal

import pvlib
import pandas as pd
import matplotlib.pyplot as plt

from config import HTW_LON, HTW_LAT, PATH_HTW_WEATHER, PATH_RESULTS, PATH_PLOTS
import modules          # Jinko-Modul (CEC-Parameter)
import v2h_inverter     # Sunny Tripower (Sandia-Parameter)
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
    surface_tilt = 30.0         # Dachneigung (von horizontal)
    surface_azimuth = 180.0     # Südausrichtung
    albedo = 0.2                # typischer Wert für städtische Umgebung

    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[
        "sapm"
    ]["open_rack_glass_polymer"]

    # PVWatts-Verlustannahmen (können im Bericht referenziert werden)
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
    # Generator + Wechselrichter (ein Array)
    # ------------------------------------------------------------------
    module_name = "Jinko_TigerNeo_440"
    module_parameters = modules.modul1()       # Series mit CEC-Parametern

    inverter_name = "Sunny_Tripower_10"
    inverter_parameters = v2h_inverter.inv1()  # Sandia-Parameter des STP 10

    # Stringauslegung: 12 Module in Serie, 2 Strings parallel → ca. 10,56 kWp
    modules_per_string = 12
    strings_per_inverter = 2

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

    # Eine ModelChain
    mc = setup_model(wr1.name, wr1, htw_location)

    # ------------------------------------------------------------------
    # Wetterdaten einlesen und aufbereiten (15 min)
    # ------------------------------------------------------------------
    df_htw = pd.read_csv(PATH_HTW_WEATHER, sep=";")

    # Spaltennamen vereinheitlichen und Zeitstempel setzen
    df_htw = htw_weather.convert_column_names(
        df_htw,
        time="timestamp",
        ghi="g_hor_si",
        wind_speed="v_wind",
        temp_air="t_luft",
    )

    # Diffuse und DNI für HTW berechnen
    df_htw = htw_weather.calculate_diffuse_irradiation(
        df_htw, parameter_name="ghi", lat=HTW_LAT, lon=HTW_LON
    )

    # Auf 15-Minuten-Raster bringen
    # (falls Daten bereits 15-minütig vorliegen, bleibt die Struktur weitgehend erhalten)
    weather_htw_15min = (
        df_htw[["ghi", "dni", "dhi"]]
        .resample("15min")
        .interpolate()
    )

    # ------------------------------------------------------------------
    # Simulation: 15-Minuten-AC-Leistungszeitreihe
    # ------------------------------------------------------------------
    mc.run_model(weather=weather_htw_15min)

    # pvlib liefert W → in kW umrechnen
    pv_ac_15min_kw = (mc.results.ac / 1000.0).rename("P_AC_kW")

    # DataFrame bauen
    pv_ts_15min = pd.DataFrame(pv_ac_15min_kw)
    pv_ts_15min.index.name = "time"

    # ------------------------------------------------------------------
    # 15-min-Zeitreihe speichern (CSV → PATH_RESULTS)
    # ------------------------------------------------------------------
    ts15_path = PATH_RESULTS / "pv_timeseries_15min.csv"
    pv_ts_15min.to_csv(ts15_path, sep=";", encoding="utf-8")
    print(f"15-min PV-Zeitreihe gespeichert unter: {ts15_path}")

    # ------------------------------------------------------------------
    # Monatserträge berechnen
    # ------------------------------------------------------------------
    result_monthly = pd.DataFrame()
    result_monthly[mc.name] = (
        pv_ts_15min["P_AC_kW"].resample("ME").sum()
    ).round(1)  # kWh

    result_monthly.index = [cal.month_name[i] for i in range(1, 13)]

    result_annual = pd.DataFrame({"annual_yield": result_monthly.sum()})

    print("#" * 50)
    print(f"{' Execution successful! (HTW, 15min) ':^50}")
    print("#" * 50, "\n")
    print(f"{' Results Monthly ':#^50}")
    print(result_monthly, "\n")
    print(f"{' Results Annual ':#^50}")
    print(result_annual, "\n")

    # ------------------------------------------------------------------
    # Monatserträge speichern (CSV → PATH_RESULTS)
    # ------------------------------------------------------------------
    monthly_path = PATH_RESULTS / "results_monthly.csv"
    result_monthly.to_csv(monthly_path, sep=";", encoding="utf-8")
    print(f"Monatserträge gespeichert unter: {monthly_path}")

    # ------------------------------------------------------------------
    # Monatsplot speichern (SVG → PATH_PLOTS)
    # ------------------------------------------------------------------
    ax = result_monthly.plot.bar(
        rot=90,
        title="Monatserträge PV-Anlage (HTW-Wetter, 2015)",
        ylabel="Energie in kWh",
        grid=True,
        legend=False,
    )
    plt.tight_layout()

    svg_path = PATH_PLOTS / "results_monthly.svg"
    plt.savefig(svg_path)
    print(f"Monatsplot gespeichert unter: {svg_path}")

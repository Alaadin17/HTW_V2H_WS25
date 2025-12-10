
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location


# ============================================================
# Konfiguration
# ============================================================

# Standort (ggf. anpassen)
LATITUDE = 52.5         # z.B. Berlin
LONGITUDE = 13.4
ALTITUDE = 34
TIMEZONE = "Europe/Berlin"

# Simulationsjahr
YEAR = 2015  # Jahr anpassen

# PV-System
PV_PEAK_POWER_KW = 7.0     # installierte DC-Leistung (kWp) – ANPASSEN
PV_TILT_DEG = 35.0         # Neigung
PV_AZIMUTH_DEG = 180.0     # 180 = Süd
PERFORMANCE_RATIO = 0.85   # pauschaler PR (Modul- + WR-Verluste)

# Dateinamen
RESULTS_DIR_NAME = "results/pvlib-V2H-WS25"
OUTPUT_CSV_NAME = "pv_profile_15min_simple.csv"


# ============================================================
# Hilfsfunktionen
# ============================================================

def get_project_root() -> Path:
    """
    Liefert das Projekt-Root-Verzeichnis (HTW_V2H_WS25).
    Annahme: Dieses Skript liegt unter <ROOT>/models/pvlib/pvlib-V2H-WS25/.
    """
    return Path(__file__).resolve().parents[3]


def build_time_index(year: int, tz: str) -> pd.DatetimeIndex:
    """
    Erzeugt eine 15-minütige Zeitachse für ein volles Jahr (lokale Zeit mit DST),
    speichert aber später ohne Zeitzonen-Info in der CSV.
    """
    start = f"{year}-01-01 00:00"
    end = f"{year}-12-31 23:45"
    times = pd.date_range(start=start, end=end, freq="15min", tz=tz)
    return times


def calculate_pv_power_kw(times: pd.DatetimeIndex) -> pd.Series:
    """
    Berechnet die PV-Leistung in kW mit einem einfachen PR-Modell
    auf Basis eines Clearsky-Modells (Ineichen).
    """

    site = Location(LATITUDE, LONGITUDE, TIMEZONE, ALTITUDE, "V2H_PV_Site")

    # Clearsky-Einstrahlung (W/m²)
    cs = site.get_clearsky(times, model="ineichen")  # gibt GHI, DNI, DHI zurück

    # Sonnenstand
    solar_pos = site.get_solarposition(times)

    # Einstrahlung auf geneigte Fläche
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=PV_TILT_DEG,
        surface_azimuth=PV_AZIMUTH_DEG,
        dni=cs["dni"],
        ghi=cs["ghi"],
        dhi=cs["dhi"],
        solar_zenith=solar_pos["zenith"],
        solar_azimuth=solar_pos["azimuth"],
    )

    poa_global = poa["poa_global"]  # W/m²

    # Einfaches PR-Modell: P = P_peak * (G / 1000 W/m²) * PR
    G_STC = 1000.0
    pv_dc_kw = PV_PEAK_POWER_KW * (poa_global / G_STC) * PERFORMANCE_RATIO

    # Numerischen Rauschen < 0 abschneiden
    pv_dc_kw = pv_dc_kw.clip(lower=0)

    return pv_dc_kw


def main():
    # Projektpfade
    project_root = get_project_root()
    results_dir = project_root / RESULTS_DIR_NAME
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Projektroot: {project_root}")
    print(f"Speichere Ergebnisse unter: {results_dir}")

    # Zeitachse
    times = build_time_index(YEAR, TIMEZONE)

    # PV-Leistung berechnen
    pv_kw = calculate_pv_power_kw(times)

    # DataFrame aufbauen
    df_pv = pd.DataFrame(index=times, data={"PV_kW": pv_kw.values})

    # Zeitzone für CSV entfernen, damit datetime zu deinem Load-CSV passt
    df_pv.index = df_pv.index.tz_localize(None)
    df_pv = df_pv.reset_index().rename(columns={"index": "datetime"})

    # Jahresenergie in kWh
    # 15-min Auflösung -> 0.25 h pro Zeitschritt
    annual_energy_kwh = (df_pv["PV_kW"] * 0.25).sum()
    print(f"Jahresenergie PV (PR-Modell): {annual_energy_kwh:.1f} kWh")

    # CSV speichern
    output_path = results_dir / OUTPUT_CSV_NAME
    df_pv.to_csv(output_path, index=False)
    print(f"PV-Zeitreihe gespeichert unter: {output_path}")

    # Ein einfacher Kontrollplot (SVG, vektorbasiert)
    try:
        import matplotlib.pyplot as plt

        # Beispiel: typischer Sommertag (21. Juni)
        mask_june21 = df_pv["datetime"].dt.date == pd.Timestamp(f"{YEAR}-06-21").date()
        df_day = df_pv.loc[mask_june21]

        if not df_day.empty:
            plt.figure()
            plt.plot(df_day["datetime"], df_day["PV_kW"])
            plt.xlabel("Zeit")
            plt.ylabel("PV-Leistung [kW]")
            plt.title(f"PV-Leistung am 21.06.{YEAR}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_path = plots_dir / f"pv_day_21-06-{YEAR}.svg"
            plt.savefig(plot_path)
            plt.close()
            print(f"Kontrollplot gespeichert unter: {plot_path}")
        else:
            print("Warnung: Keine Daten für 21.06 gefunden – Plot wird übersprungen.")

    except Exception as e:
        print(f"Plot-Erstellung fehlgeschlagen (optional): {e}")


if __name__ == "__main__":
    main()

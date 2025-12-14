

"""
Build_timeseries_load.py

Erzeugt aus den dreiphasigen Leistungsdaten (PL1, PL2, PL3)
ein 15-minütiges Haushaltslastprofil (kW) für das Oemof-Modell.

Input Profil 31 (im Ordner "Data/"):
    - PL1.csv, PL2.csv, PL3.csv   (Tjaden-Datensatz)
    - time_datevec_MEZ.csv        (Jahr, Monat, Tag, Stunde, Minute, Sekunde)

Output:
    - load_profile_15min.csv mit Spalten:
        datetime, P_load_kW
"""

from pathlib import Path
import pandas as pd


def main():
    # Basisordner: Projekt-Root = zwei Ebenen über diesem Skript
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent  # falls Scripts direkt unter Root liegen
    data_dir = project_root / "Data"

    # Dateien
    pl1_path = data_dir / "PL1.csv"
    pl2_path = data_dir / "PL2.csv"
    pl3_path = data_dir / "PL3.csv"
    time_path = data_dir / "time_datevec_MEZ.csv"

    print(f"Lese Daten aus: {data_dir}")

    # -----------------------------
    # 1. Zeitstempel einlesen
    # -----------------------------
    # Annahme: time_datevec_MEZ.csv hat 6 Spalten:
    # Jahr, Monat, Tag, Stunde, Minute, Sekunde (ohne Header)
    time_df = pd.read_csv(time_path, header=None)
    time_df.columns = ["year", "month", "day", "hour", "minute", "second"]

    # Zu pandas-Datetime konvertieren (MEZ)
    dt_index = pd.to_datetime(time_df)

    # -----------------------------
    # 2. Leistungsdaten einlesen
    # -----------------------------
    # Annahme: Keine Header, Profil 31 ist Spalte 31 (1-basiert) → iloc[:, 30]
    PROFILE_COL = 30  # 0-basiert

    pl1 = pd.read_csv(pl1_path, header=None)
    pl2 = pd.read_csv(pl2_path, header=None)
    pl3 = pd.read_csv(pl3_path, header=None)

    # Sicherstellen, dass alle gleich lang sind
    min_len = min(len(pl1), len(pl2), len(pl3), len(dt_index))
    if min_len < len(dt_index):
        print(f"Achtung! Kürze Zeitreihe auf {min_len} Einträge, um Längen anzupassen.")
    dt_index = dt_index[:min_len]
    pl1 = pl1.iloc[:min_len, :]
    pl2 = pl2.iloc[:min_len, :]
    pl3 = pl3.iloc[:min_len, :]

    # Profil 31 je Phase (in Watt)
    P_L1_W = pl1.iloc[:, PROFILE_COL]
    P_L2_W = pl2.iloc[:, PROFILE_COL]
    P_L3_W = pl3.iloc[:, PROFILE_COL]

    # -----------------------------
    # 3. Drei Phasen aufsummieren
    # -----------------------------
    P_total_W = P_L1_W + P_L2_W + P_L3_W

    # DataFrame mit Zeitindex bauen
    df = pd.DataFrame({"P_load_W": P_total_W.values}, index=dt_index)
    df.index.name = "datetime"

    # Optional: kurzer Check
    print("Original-Auflösung:")
    print("Anzahl Punkte:", len(df))
    print("Mittlere Leistung [W]:", df["P_load_W"].mean())
    energy_kwh = (df["P_load_W"] / 1000 * (1 / 60)).sum()    # 1 Minute Auflösung → Δt = 1/60 h
    print("Summe Energie [kWh]:", energy_kwh)

    # -----------------------------------
    # 4. Auf 15 Minuten mitteln (kW)
    # -----------------------------------
    #   – Eingangsdaten von 1 min auf 15 Minuten mitteln

    # 15-min-Resampling
    df_15min = df.resample("15min").mean()

    # In kW umrechnen
    df_15min["P_load_kW"] = df_15min["P_load_W"] / 1000.0
    df_15min = df_15min[["P_load_kW"]]  # nur die kW-Spalte behalten

    print("15-min-Auflösung:")
    print("Anzahl Punkte:", len(df_15min))
    print("Mittlere Leistung [kW]:", df_15min["P_load_kW"].mean())
    print(
        "  Summe Energie [kWh]:",
        (df_15min["P_load_kW"] * 0.25).sum(),  # 0.25 h pro Intervall
    )

    # -----------------------------
    # 5. CSV speichern
    # -----------------------------
    out_path = data_dir / "load_profile_15min.csv"
    df_15min.to_csv(out_path, index=True)  # index = datetime

    print(f"load_profile_15min.csv gespeichert unter: {out_path}")


if __name__ == "__main__":
    main()

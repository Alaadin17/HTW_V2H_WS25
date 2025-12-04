"""
Braucht scipy.io zum lesen der .mat Datei
Skript liest die .mat  Datei und erstellt aus den inhalten  jeweils eine  .csv datei
für:
P_da= elektrische Dauerlast/ Haushaltsprofil
P_PVS=PV-Systemleistung

Umwandlung von 1 Minute Zeitschritten in 15 Minute
erzeugen von timestamp/Zeitachse
Start: 1.1.2023 00:00
Länge: 35 040 Zeitschritte

project/
│
├── data/
│   ├── A06_Daten.mat
│   ├── *.csv   ← wird automatisch erzeugt
│
└── scripts/
    └── mat_to_csv.py
"""
# scripts/a06_pda_ppvs_to_15min_csv.py

import os
import numpy as np
import pandas as pd
import scipy.io

# ----------------------------------------------------------------------
# Pfade
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

MAT_FILENAME = "A06_Daten.mat"   # ggf. anpassen
mat_path = os.path.join(DATA_DIR, MAT_FILENAME)

print(f"➜ Lade MAT-Datei: {mat_path}")
mat = scipy.io.loadmat(mat_path, squeeze_me=True)

# ----------------------------------------------------------------------
# Hilfsfunktion: 1-min-Leistung zu 15-min-Leistung (Mittelwert)
# ----------------------------------------------------------------------
def to_15min_power(arr_1min: np.ndarray) -> np.ndarray:
    """
    Aggregiert eine 1-min-Auflösung auf 15-min-Auflösung,
    indem jeweils über 15 Werte gemittelt wird.
    Erwartet ein 1D-Array.
    """
    arr_1min = np.asarray(arr_1min, dtype=float).squeeze()

    if arr_1min.ndim != 1:
        raise ValueError(f"Erwarte 1D-Array, bekommen: shape={arr_1min.shape}")

    n = len(arr_1min)
    if n % 15 != 0:
        raise ValueError(f"Länge {n} ist nicht durch 15 teilbar – "
                         f"kann nicht sauber auf 15 min aggregieren.")

    reshaped = arr_1min.reshape(-1, 15)  # (35040, 15)
    return reshaped.mean(axis=1)


# ----------------------------------------------------------------------
# Pda und ppvs auslesen
# ----------------------------------------------------------------------
try:
    Pda_1min = mat["Pda"]
    ppvs_1min = mat["ppvs"]
except KeyError as e:
    raise KeyError(f"Variable {e} nicht in {MAT_FILENAME} gefunden. "
                   f"Verfügbare Keys: {list(mat.keys())}")

Pda_1min = np.squeeze(Pda_1min)
ppvs_1min = np.squeeze(ppvs_1min)

print(f"Pda:  shape={Pda_1min.shape} (sollte 525600 sein)")
print(f"ppvs: shape={ppvs_1min.shape} (sollte 525600 sein)")

# ----------------------------------------------------------------------
# Auf 15-min-Auflösung bringen
# ----------------------------------------------------------------------
Pda_15 = to_15min_power(Pda_1min)
ppvs_15 = to_15min_power(ppvs_1min)

print(f"Pda_15:  shape={Pda_15.shape} (erwartet 35040)")
print(f"ppvs_15: shape={ppvs_15.shape} (erwartet 35040)")

# ----------------------------------------------------------------------
# Synthetische Zeitachse für oemof
# ----------------------------------------------------------------------
# 35040 Schritte à 15 min ≙ 1 Jahr
start_time = "2023-01-01 00:00"
index_15min = pd.date_range(start=start_time, periods=len(Pda_15), freq="15min")

# ----------------------------------------------------------------------
# DataFrames bauen und als CSV speichern
# ----------------------------------------------------------------------
df_pda = pd.DataFrame({"Pda": Pda_15}, index=index_15min)
df_pda.index.name = "timestamp"

df_ppvs = pd.DataFrame({"ppvs": ppvs_15}, index=index_15min)
df_ppvs.index.name = "timestamp"

csv_pda = os.path.join(DATA_DIR, "A06_Pda_15min.csv")
csv_ppvs = os.path.join(DATA_DIR, "A06_ppvs_15min.csv")

df_pda.to_csv(csv_pda)
df_ppvs.to_csv(csv_ppvs)

print(f"✔ CSV geschrieben: {csv_pda}")
print(f"✔ CSV geschrieben: {csv_ppvs}")
print("Fertig.")


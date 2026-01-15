#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

"""
In this config file you have to assign the variables which are used in the calculations.
"""

# Define the paths of the weather-data files.
PATH_HTW_WEATHER = r"wetter-htw-2025.csv"
PATH_FRED_WEATHER = r"openfred_weatherdata_2015_htw.csv"

# Define the location of the PV-system (lat, lon).
HTW_LAT = 52.45544
HTW_LON = 13.52481

# Define the path where the results are to be stored
# If the string is empty, the files are saved where the script is executed.
# It is important to type path seperator at the end: e.g. /home/user/Documents/


# ---------------------------------------------------------------------------
# 1) Projekt-Root automatisch bestimmen
#    (config.py liegt unter: models/pvlib/pvlib-V2H-WS25/)
#    → Root ist zwei Ebenen darüber: HTW_V2H_WS25/
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# 2) Resultat-Ordner definieren (relativ zum Projekt)
# ---------------------------------------------------------------------------

PATH_RESULTS = PROJECT_ROOT / "results" / "pvlib-V2H-WS25"

PATH_PLOTS = PROJECT_ROOT / "results" / "pvlib-V2H-WS25" / "plots"


# Ordner automatisch erstellen
PATH_RESULTS.mkdir(parents=True, exist_ok=True)
PATH_PLOTS.mkdir(exist_ok=True, parents=True)


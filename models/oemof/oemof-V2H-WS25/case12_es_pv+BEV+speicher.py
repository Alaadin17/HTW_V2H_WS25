"""
oemof Energy System Model für PV + BEV System + Batteriespeicher

Verbesserungen gegenüber der Original-Version:
- Konfigurationsparameter als Dataclass für bessere Übersicht
- Trennung von Datenladung und Systemerstellung
- Verwendung von pathlib statt hardcoded Pfaden
- Bessere Fehlerbehandlung
- Type Hints für bessere Code-Qualität
- Logging-Konfiguration als separate Methode
- Validierung der Eingangsdaten

System-Topologie:
        Bus_electricity              Bus_mobility
            |                            |
PV--------->|<---->home_battery          |<---->BEV_battery
            |    (Batteriespeicher)      |
Grid------->|<--------Wallbox----------->|------>BEV_consumption (Sink)
            |        (Converter)         |
excess------|                            |
            |                            |
demand<-----|                            |

Komponenten:
- Bus_electricity:  Haushalts-Elektrizitätsbus (PV, Netz, Hausbedarf, Überschuss)
- Bus_mobility:     Mobilitäts-/BEV-Bus (Batterie und Fahrbedarf)
- PV:               Photovoltaik-Anlage (Source)
- Grid:             Netzanschluss (Source für Bezug, Sink für Einspeisung via excess)
- Hybridwechselrichter
- home_battery:     Stationärer Batteriespeicher (10 kWh, SOC 10-100%, balanced=False)
- Wallbox:          Ladestation als Converter zwischen electricity und mobility Bus
                    - Input: Bus_electricity (variable_costs=0.0 für maximales Laden)
                    - Output: Bus_mobility (max=BEV_at_home, nominal=11 kW)
- BEV_battery:      Batteriespeicher (77 kWh, SOC 20-95%, balanced=False)
- BEV_consumption:  Fahrverbrauch als separater Sink (fix=consumption/0.25 [kW])
- excess:           Überschuss-Senke für Netzeinspeisung
- demand:           Haushaltslast

Wichtige Hinweise:
1. Wallbox ist CONVERTER (nicht Source) - verbindet beide Busse
2. BEV_consumption als SEPARATER SINK
3. variable_costs=0.0 an Wallbox → maximiert Ladevorgänge
4. Consumption in kW umgerechnet (kWh/0.25)
5. Batteriespeicher optimiert Eigenverbrauch
6. Überschüssige PV-Energie wird gespeichert
7. Batterie kann Haushaltslast bei geringer PV-Erzeugung decken
"""

###########################################################################
# Imports
###########################################################################
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import warnings

import pandas as pd
from oemof.solph import (
    EnergySystem,
    Model,
    buses,
    components as cmp,
    flows,
    helpers,
    processing,
    NonConvex
)
from oemof.tools import logger
from pyomo.opt import SolverStatus, TerminationCondition


###########################################################################
# Konfiguration
###########################################################################
@dataclass
class SystemConfig:
    """Konfigurationsparameter für das Energiesystem"""
    
    # Zeitliche Parameter
    start_date: str = "2022-01-01"
    periods: int = 96  # 15-Minuten-Schritte (96 = 1 Tag für Debug)
    freq: str = "15min"
    
    # Solver-Einstellungen
    solver: str = "cbc"
    solver_verbose: bool = False
    debug: bool = True  # DEBUG-MODUS für LP-Datei
    solver_threads: int = 6 # Anzahl Threads für den Solver
    solver_ratio_gap: float = 0.05  # 1% Lücken-Toleranz
    
    # System-Parameter
    grid_supply_power_kW: float = 30.0
    wallbox_power_kW: float = 11.0
    wallbox_efficiency_charge: float = 0.95  # Wirkungsgrad beim Laden
    wallbox_efficiency_discharge: float = 0.90  # Wirkungsgrad beim Entladen (V2H)
    enable_v2g: bool = True  # V2G aktivieren/deaktivieren
    v2g_variable_costs: float = 5.0  # €/MWh - verhindert unnötiges Ent-/Wiederaufladen
    
    # BEV-Parameter
    bev_capacity_kWh: float = 77.0
    bev_min_soc: float = 0.2
    bev_max_soc: float = 0.95
    bev_initial_soc: float = 0.95

    # Stationärer Batteriespeicher-Parameter
    battery_capacity_kWh: float = 10.2  # Kapazität des Hausbatteriespeichers von BYD B-Box Premium HVS 10.2 Battery Storage 10.24 kWh
    battery_min_soc: float = 0.1  # Minimaler Ladezustand (10%)
    battery_max_soc: float = 1.0  # Maximaler Ladezustand (100%)
    battery_initial_soc: float = 0.5  # Anfangs-Ladezustand (50%)
    battery_efficiency: float = 0.95  # Wirkungsgrad beim Laden/Entladen
    battery_max_power_kW: float = 10.0  # Maximale Lade-/Entladeleistung (AC Coupling) von Fronius Symo GEN24 10.0 Plus Hybrid-Inverter
    
    
    # Kosten (€/MWh oder €/kWh)
    pv_variable_costs: float = 0.0
    grid_variable_costs: float = 30.0
    grid_feedin_tariff: float = -7.9  # Negativ, weil es Einnahmen sind
    
    # Ergebnis-Speicherung
    should_dump_results: bool = True
    dump_filename: str = "case12_pv+BEV+speicher"
    
    # Logging
    log_filename: str = "oemof_case12.log"
    log_screen_level: int = logging.INFO
    log_file_level: int = logging.INFO  # DEBUG würde oemof verlangsamen


###########################################################################
# Hilfsfunktionen
###########################################################################
def setup_logging(config: SystemConfig) -> None:
    """
    Konfiguriert das Logging-System
    
    Parameters:
    -----------
    config : SystemConfig
        Konfigurationsobjekt mit Logging-Parametern
    """
    logger.define_logging(
        logfile=config.log_filename,
        screen_level=config.log_screen_level,
        file_level=config.log_file_level,
    )


def load_timeseries(
    input_file: Optional[Path] = None
) -> Tuple[pd.DataFrame, Path]:
    """
    Lädt die Zeitreihendaten aus CSV
    
    Parameters:
    -----------
    input_file : Path, optional
        Pfad zur Input-Datei. Falls None, wird der Standard-Pfad verwendet.
        
    Returns:
    --------
    df_timeseries : pd.DataFrame
        DataFrame mit allen Zeitreihen
    timeseries_path : Path
        Pfad zur geladenen Datei
    """
    if input_file is None:
        script_dir = Path(__file__).resolve().parent
        input_file = script_dir / "Input_timeseries" / "input_timeseries.csv"
    
    if not input_file.exists():
        raise FileNotFoundError(f"Zeitreihen-Datei nicht gefunden: {input_file}")
    
    logging.info(f"Lade Zeitreihen aus: {input_file}")
    df_timeseries = pd.read_csv(input_file, delimiter=",")
    
    # Validiere erforderliche Spalten
    required_columns = [
        "PV_kW", "Load_kW", "BEV_at_home", 
        "consumption", "charging_power_kW"
    ]
    missing_cols = set(required_columns) - set(df_timeseries.columns)
    if missing_cols:
        raise ValueError(f"Fehlende Spalten in der Zeitreihen-Datei: {missing_cols}")
    
    return df_timeseries, input_file


def validate_and_clean_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validiert und bereinigt die Zeitreihendaten
    
    Parameters:
    -----------
    df : pd.DataFrame
        Rohdaten
        
    Returns:
    --------
    df_clean : pd.DataFrame
        Bereinigte Daten
    """
    df_clean = df.copy()
    
    # PV-Werte müssen >= 0 sein (negative Werte → 0)
    df_clean["PV_kW"] = df_clean["PV_kW"].clip(lower=0)
    
    # Prüfe auf NaN-Werte
    nan_counts = df_clean[["PV_kW", "Load_kW", "BEV_at_home", "consumption"]].isna().sum()
    if nan_counts.any():
        logging.warning(f"NaN-Werte gefunden:\n{nan_counts[nan_counts > 0]}")
        df_clean = df_clean.fillna(0)
    
    # Validiere BEV_at_home (sollte binär sein)
    if not df_clean["BEV_at_home"].isin([0, 1]).all():
        logging.warning("BEV_at_home enthält nicht-binäre Werte. Runde auf 0/1.")
        df_clean["BEV_at_home"] = df_clean["BEV_at_home"].round().astype(int)
    
    logging.info(f"Zeitreihen validiert: {len(df_clean)} Zeitschritte")
    logging.info(f"  PV: {df_clean['PV_kW'].min():.2f} - {df_clean['PV_kW'].max():.2f} kW")
    logging.info(f"  Last: {df_clean['Load_kW'].min():.2f} - {df_clean['Load_kW'].max():.2f} kW")
    logging.info(f"  BEV Verbrauch: {df_clean['consumption'].sum():.2f} kWh gesamt")
    
    return df_clean


###########################################################################
# Hauptklasse
###########################################################################
class EnergySystemModel:
    """
    Modelliert und optimiert ein Energiesystem mit PV und BEV
    """
    
    def __init__(
        self, 
        config: Optional[SystemConfig] = None,
        timeseries_file: Optional[Path] = None
    ):
        """
        Initialisiert das Energiesystem-Modell
        
        Parameters:
        -----------
        config : SystemConfig, optional
            Konfiguration. Falls None, wird Standard verwendet.
        timeseries_file : Path, optional
            Pfad zur Zeitreihen-Datei. Falls None, Standard-Pfad.
        """
        self.config = config or SystemConfig()
        self.timeseries_file = timeseries_file
        
        # System-Variablen
        self.time_index: Optional[pd.DatetimeIndex] = None
        self.es: Optional[EnergySystem] = None
        self.model: Optional[Model] = None
        self.df_timeseries: Optional[pd.DataFrame] = None
        
        # Setup
        setup_logging(self.config)
        logging.info("=" * 80)
        logging.info("Initialisiere Energy System Model")
        logging.info("=" * 80)
        
    def run(self) -> None:
        """Führt die komplette Modellierung und Optimierung aus"""
        try:
            self._load_data()
            self._create_time_index()
            self._create_energy_system()
            self._create_components()
            self._optimize()
            self._solve()
            self._extract_results()
            self._save_results()
            
            logging.info("=" * 80)
            logging.info("[SUCCESS] Simulation erfolgreich abgeschlossen")
            logging.info("=" * 80)
            
        except Exception as e:
            logging.error(f"❌ Fehler während der Simulation: {e}")
            raise
    
    def _load_data(self) -> None:
        """Lädt und validiert die Eingangsdaten"""
        logging.info("Schritt 1: Lade Zeitreihendaten")
        df_raw, _ = load_timeseries(self.timeseries_file)
        self.df_timeseries = validate_and_clean_timeseries(df_raw)
        
        # Prüfe Länge der Zeitreihen
        if len(self.df_timeseries) < self.config.periods:
            logging.warning(
                f"Zeitreihen zu kurz! Verfügbar: {len(self.df_timeseries)}, "
                f"Benötigt: {self.config.periods}. Kürze Simulation."
            )
            self.config.periods = len(self.df_timeseries)
    
    def _create_time_index(self) -> None:
        """Erstellt den Zeit-Index für die Simulation"""
        logging.info("Schritt 2: Erstelle Zeit-Index")
        self.time_index = pd.date_range(
            start=self.config.start_date,
            periods=self.config.periods,
            freq=self.config.freq
        )
        logging.info(
            f"  Zeitraum: {self.time_index[0]} bis {self.time_index[-1]}"
        )
        logging.info(f"  Anzahl Zeitschritte: {self.config.periods}")
    
    def _create_energy_system(self) -> None:
        """Erstellt das oemof EnergySystem-Objekt"""
        logging.info("Schritt 3: Erstelle Energy System")
        self.es = EnergySystem(
            timeindex=self.time_index,
            infer_last_interval=True
        )
        logging.info(f"  EnergySystem erstellt mit {len(self.time_index)} Zeitschritten")
    
    def _create_components(self) -> None:
        """Erstellt alle Komponenten (Busse, Quellen, Senken, Speicher)"""
        logging.info("Schritt 4: Erstelle Komponenten")
        
        # Extrahiere Zeitreihen (nur bis periods)
        pv_timeseries = self.df_timeseries["PV_kW"].iloc[:self.config.periods]
        load_timeseries = self.df_timeseries["Load_kW"].iloc[:self.config.periods]
        bev_at_home = self.df_timeseries["BEV_at_home"].iloc[:self.config.periods]
        # Consumption in kWh pro Zeitschritt → muss in kW umgewandelt werden
        bev_consumption_kWh = self.df_timeseries["consumption"].iloc[:self.config.periods]
        bev_consumption = bev_consumption_kWh / 0.25  # kWh → kW für fixed_losses_absolute
        # ===== BUSSE =====
        logging.info("  - Erstelle Busse")
        b_el = buses.Bus(label="electricity")
        b_bev = buses.Bus(label="mobility", balanced=True)
        self.es.add(b_el, b_bev)
        
        # ===== ELEKTRIZITÄTSBUS KOMPONENTEN =====
        logging.info("  - Erstelle PV-Anlage")
        self.es.add(
            cmp.Source(
                label="pv",
                outputs={
                    b_el: flows.Flow(
                        fix=pv_timeseries,
                        nominal_value=1,
                        variable_costs=self.config.pv_variable_costs
                    )
                },
            )
        )
        
        logging.info("  - Erstelle Netz-Anbindung")
        self.es.add(
            cmp.Source(
                label="grid_supply",
                outputs={
                    b_el: flows.Flow(
                        variable_costs=self.config.grid_variable_costs,
                        nominal_value=self.config.grid_supply_power_kW
                    )
                },
            )
        )
        
        logging.info("  - Erstelle Netz-Einspeisung (Überschuss-Senke)")
        self.es.add(
            cmp.Sink(
                label="excess_electricity",
                inputs={
                    b_el: flows.Flow(
                        variable_costs=self.config.grid_feedin_tariff
                    )
                }
            )
        )
        
        logging.info("  - Erstelle Haushaltslast")
        self.es.add(
            cmp.Sink(
                label="household_demand",
                inputs={
                    b_el: flows.Flow(
                        fix=load_timeseries,
                        nominal_value=1
                    )
                },
            )
        )
        
        # ===== STATIONÄRER BATTERIESPEICHER =====
        logging.info("  - Erstelle stationären Batteriespeicher")
        self.es.add(
            cmp.GenericStorage(
                label="home_battery",
                inputs={
                    b_el: flows.Flow(
                        nominal_value=self.config.battery_max_power_kW
                    )
                },
                outputs={
                    b_el: flows.Flow(
                        nominal_value=self.config.battery_max_power_kW
                    )
                },
                nominal_storage_capacity=self.config.battery_capacity_kWh,
                min_storage_level=self.config.battery_min_soc,
                max_storage_level=self.config.battery_max_soc,
                initial_storage_level=self.config.battery_initial_soc,
                inflow_conversion_factor=self.config.battery_efficiency,  # Verluste beim Laden
                outflow_conversion_factor=self.config.battery_efficiency,  # Verluste beim Entladen
                loss_rate=0.001,  # 0.1% Selbstentladung pro Zeitschritt
                balanced=False,  # Speicher muss am Ende nicht gleich Anfang sein
            )
        )


        # ===== MOBILITÄTSBUS KOMPONENTEN =====
        logging.info("  - Erstelle Wallbox (Ladestation als Transformer)")
        self.es.add(
            cmp.Converter(
                label="wallbox",
                inputs={
                    b_el: flows.Flow(
                        variable_costs=0.0  # Kostenlos laden → Speicher lädt so oft wie möglich
                    )
                },
                outputs={
                    b_bev: flows.Flow(
                        max=bev_at_home,  # Kann nur laden, wenn BEV zu Hause
                        nominal_value=self.config.wallbox_power_kW
                    )
                },
                conversion_factors={b_bev: 1.0}  # 100% Effizienz
            )
        )

         # ===== V2G mit BINÄRER OPERATION =====
        if self.config.enable_v2g:
            logging.info(f"  - Erstelle V2G-Funktion (BINÄR + EXAKT {self.config.v2g_full_load_time_min}h erzwungen)")
            logging.info(f"    • nominal_value: {self.config.wallbox_power_kW} kW")
            logging.info(f"    • min=0, max=1.0 -> Nur 100% oder AUS")
            
            self.es.add(
                cmp.Converter(
                    label="wallbox_discharge",
                    inputs={
                        b_bev: flows.Flow(
                            variable_costs=self.config.v2g_variable_costs
                        )
                    },
                    outputs={
                        b_el: flows.Flow(
                            min=0.0,
                            max=bev_at_home,  # Kann nur entladen wenn zu Hause
                            nominal_value=self.config.wallbox_power_kW,
                        )
                    },
                    conversion_factors={b_el: self.config.wallbox_efficiency_discharge}
                )
            )
        
        logging.info("  - Erstelle BEV-Verbrauch (Fahrverbrauch als Sink)")
        self.es.add(
            cmp.Sink(
                label="bev_consumption",
                inputs={
                    b_bev: flows.Flow(
                        fix=bev_consumption,  # Consumption in kW (bereits umgerechnet)
                        nominal_value=1
                    )
                },
            )
        )
        
        logging.info("  - Erstelle BEV-Batterie")
        self.es.add(
            cmp.GenericStorage(
                label="bev_battery",
                inputs={b_bev: flows.Flow()},
                outputs={b_bev: flows.Flow()},
                nominal_storage_capacity=self.config.bev_capacity_kWh,
                min_storage_level=self.config.bev_min_soc,
                max_storage_level=self.config.bev_max_soc,
                initial_storage_level=self.config.bev_initial_soc,
                loss_rate=0.0,
                balanced=False,  # Speicher muss am Ende nicht gleich Anfang sein
            )
        )
        
        logging.info(f"  [OK] {len(self.es.nodes)} Komponenten erstellt")
    
    def _optimize(self) -> None:
        """Erstellt das Optimierungsmodell"""
        logging.info("Schritt 5: Erstelle Optimierungsmodell")
        self.model = Model(self.es)
        
        # Optional: LP-Datei schreiben für Debugging
        if self.config.debug:
            lp_path = Path(helpers.extend_basic_path("lp_files")) / "debug_model.lp"
            logging.info(f"  Debug-Modus: Schreibe LP-Datei nach {lp_path}")
            lp_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.write(str(lp_path), io_options={"symbolic_solver_labels": True})
    
    def _solve(self) -> None:
        """Löst das Optimierungsmodlem"""
        logging.info("Schritt 6: Löse Optimierungsproblem")
        logging.info(f"  Solver: {self.config.solver}")
        
                # Solver-Optionen
        solver_options = {}
        if self.config.solver == "cbc":
            solver_options = {
                "threads": self.config.solver_threads,
                "ratioGap": self.config.solver_ratio_gap,
            }

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            results = self.model.solve(
                solver=self.config.solver,
                solve_kwargs={"tee": self.config.solver_verbose},
                cmdline_options=solver_options
                
            )
            
            # Prüfe Solver-Status
            status = results.solver.status
            termination = results.solver.termination_condition
            
            if status != SolverStatus.ok or termination != TerminationCondition.optimal:
                error_msg = (
                    f"\n❌ Optimierung fehlgeschlagen!\n"
                    f"  Status: {status}\n"
                    f"  Terminierung: {termination}\n"
                    f"  Nachricht: {results.solver.message}\n"
                )
                logging.error(error_msg)
                raise RuntimeError(error_msg)
            
            logging.info("  [OK] Optimierung erfolgreich")
            logging.info(f"  Objective Value: {self.model.objective():.2f} €")
    
    def _extract_results(self) -> None:
        """Extrahiert die Ergebnisse aus dem gelösten Modell"""
        logging.info("Schritt 7: Extrahiere Ergebnisse")
        self.es.results["main"] = processing.results(self.model)
        self.es.results["meta"] = processing.meta_results(self.model)
        logging.info("  [OK] Ergebnisse extrahiert")
    
    def _save_results(self) -> None:
        """Speichert die Ergebnisse als Dump"""
        if not self.config.should_dump_results:
            logging.info("Schritt 8: Ergebnis-Speicherung übersprungen (deaktiviert)")
            return
        
        logging.info("Schritt 8: Speichere Ergebnisse")
        
        # Bestimme Speicherpfad
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parents[2]
        dump_path = project_root / "results" / "oemof-V2H-WS25" / "dumps"
        dump_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self.es.dump(
                dpath=str(dump_path),
                filename=self.config.dump_filename
            )
            logging.info(f"  [OK] Dump gespeichert: {dump_path / self.config.dump_filename}")
        except Exception as e:
            logging.error(f"  ❌ Fehler beim Speichern: {e}")
            raise


###########################################################################
# Hauptprogramm
###########################################################################
def main():
    """Hauptfunktion"""
    # Erstelle Konfiguration (kann angepasst werden)
    config = SystemConfig(
        # Zeitliche Parameter
        start_date="2022-01-01",
        periods=24*31*4,  # 1 Jahr mit 15-Minuten-Auflösung
        
        # System-Parameter
        grid_supply_power_kW=30.0,
        wallbox_power_kW=11.0,
        wallbox_efficiency_charge=0.95,
        wallbox_efficiency_discharge=0.90,
        enable_v2g=True,  # V2G aktivieren
        v2g_variable_costs=5.0,  # €/MWh - verhindert unnötiges Ent-/Wiederaufladen
        
        # BEV-Parameter
        bev_capacity_kWh=77.0,
        bev_min_soc=0.2,
        bev_max_soc=0.95,
        bev_initial_soc=0.5,



        # Stationärer Batteriespeicher-Parameter
        battery_capacity_kWh=10.2,  # Kapazität des Hausbatteriespeichers von BYD B-Box Premium HVS 10.2 Battery Storage 10.24 kWh
        battery_min_soc=0.1,  # Minimaler Ladezustand (10%)
        battery_max_soc=1.0,  # Maximaler Ladezustand (100%)
        battery_initial_soc=0.5,  # Anfangs-Ladezustand (50%)
        battery_efficiency=0.95,  # Wirkungsgrad beim Laden/Entladen
        battery_max_power_kW=10.0,  # Maximale Lade-/Entladeleistung (AC Coupling) von Fronius Symo GEN24 10.0 Plus Hybrid-Inverter


        
        # Kosten
        pv_variable_costs=0.0,  # cent/kWh
        grid_variable_costs=30.0,  # cent/kWh
        grid_feedin_tariff=-7.9,  # cent/kWh
        
        # Ergebnis-Speicherung
        should_dump_results=True,
        dump_filename="case12_es_pv+BEV+speicher",
        
        # Debugging
        debug=False,
        solver_verbose=False,
    )
    
    # Erstelle und führe Modell aus
    model = EnergySystemModel(config=config)
    model.run()


if __name__ == "__main__":
    main()

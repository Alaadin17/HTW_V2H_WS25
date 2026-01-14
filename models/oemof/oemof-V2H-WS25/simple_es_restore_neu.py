"""
ENERGIESYSTEM ERGEBNISSE RESTOREN UND VISUALISIEREN

Dieses Script lädt gespeicherte oemof-Ergebnisse und erstellt:
1. Ordentlich strukturierte CSV-Dateien
2. Interaktive Plots mit Ein-/Ausblenden von Linien

Features:
- Sinnvolle deutsche Spaltennamen
- Separate Extraktion von Electricity-Bus, BEV-Batterie, Heimspeicher
- Kombinierte CSV mit allen Daten
- Interaktiver Plot: Klick auf Legend zum Ein-/Ausblenden
"""

import matplotlib.pyplot as plt
import pandas as pd
from oemof.solph import EnergySystem, views
import logging
from oemof.tools import logger
import os
import matplotlib
from pathlib import Path

matplotlib.use("TkAgg")

logger.define_logging(
    logfile="oemof_restore.log",
    screen_level=logging.INFO,
    file_level=logging.INFO,
)


def get_dump_path():
    """Gibt den Pfad zum Dump-Ordner zurück"""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]
    dump_path = project_root / "results" / "oemof-V2H-WS25" / "dumps"
    return str(dump_path)


def get_save_path():
    """Gibt den Pfad zum Timeseries-Ordner zurück"""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]
    csv_path = project_root / "results" / "oemof-V2H-WS25" / "timeseries"
    csv_path.mkdir(parents=True, exist_ok=True)
    return str(csv_path)


def restore_and_process_results(case_name: str) -> dict:
    """
    Restored Energiesystem-Ergebnisse und bereitet alle Daten auf
    
    Returns:
        dict mit 'electricity', 'bev_battery', 'home_battery', 'combined'
    """
    print("\n" + "="*80)
    print(f"RESTORE UND VERARBEITE: {case_name}")
    print("="*80)
    
    # 1. Restore Energiesystem
    dump_path = get_dump_path()
    logging.info(f"Restore aus: {dump_path}")
    
    energysystem = EnergySystem()
    energysystem.restore(dpath=dump_path, filename=case_name)
    results = energysystem.results["main"]
    
    # 2. Extrahiere Elektrizitäts-Flüsse
    elec_flows = extract_electricity_flows(results)
    print(f"\n✓ Elektrizitäts-Flüsse: {len(elec_flows.columns)} Spalten")
    
    # 3. Extrahiere Batterie-Daten
    result_dict = {'electricity': elec_flows}
    
    if has_component(results, "bev_battery"):
        bev_battery = extract_battery_flows(results, "bev_battery", prefix="BEV")
        result_dict['bev_battery'] = bev_battery
        print(f"✓ BEV-Batterie: {len(bev_battery.columns)} Spalten")
    
    if has_component(results, "home_battery"):
        home_battery = extract_battery_flows(results, "home_battery", prefix="Heimspeicher")
        result_dict['home_battery'] = home_battery
        print(f"✓ Heimspeicher: {len(home_battery.columns)} Spalten")
    
    # 4. Kombiniere alle Daten
    dfs_to_combine = [elec_flows]
    if 'home_battery' in result_dict:
        dfs_to_combine.append(result_dict['home_battery'])
    if 'bev_battery' in result_dict:
        dfs_to_combine.append(result_dict['bev_battery'])
    
    combined = pd.concat(dfs_to_combine, axis=1)
    
    # 5. Sortiere Spalten in gewünschter Reihenfolge
    desired_order = [
        "Haushaltslast [kW]",
        "PV-Erzeugung [kW]",
        "Netzbezug [kW]",
        "Netzeinspeisung [kW]",
        "Wallbox Ladung [kW]",
        "Heimspeicher Laden [kW]",
        "Heimspeicher Entladen [kW]",
        "Heimspeicher Ladezustand [kWh]",
        "BEV Laden [kW]",
        "BEV Entladen [kW]",
        "BEV Ladezustand [kWh]"
    ]
    
    # Nur vorhandene Spalten verwenden
    ordered_cols = [col for col in desired_order if col in combined.columns]
    remaining_cols = [col for col in combined.columns if col not in ordered_cols]
    combined = combined[ordered_cols + remaining_cols]
    
    result_dict['combined'] = combined
    print(f"✓ Kombiniert: {len(combined.columns)} Spalten, {len(combined)} Zeitschritte")
    
    return result_dict


def has_component(results: dict, component_name: str) -> bool:
    """Prüft ob eine Komponente in den Ergebnissen existiert"""
    for key in results.keys():
        if component_name in str(key[0]) or component_name in str(key[1]):
            return True
    return False


def extract_electricity_flows(results: dict) -> pd.DataFrame:
    """Extrahiert Elektrizitäts-Bus Flüsse mit deutschen Namen"""
    flows = {}
    
    for k, v in results.items():
        node_from = str(k[0])
        node_to = str(k[1])
        
        if node_from == "pv" and node_to == "electricity":
            flows["PV-Erzeugung [kW]"] = v["sequences"]["flow"]
        elif node_from == "electricity" and node_to in ["excess_electricity", "excess_bel"]:
            flows["Netzeinspeisung [kW]"] = v["sequences"]["flow"]
        elif node_from in ["grid_supply", "grid-supply"] and node_to == "electricity":
            flows["Netzbezug [kW]"] = v["sequences"]["flow"]
        elif node_from == "electricity" and node_to in ["household_demand", "demand"]:
            flows["Haushaltslast [kW]"] = v["sequences"]["flow"]
        elif node_from == "electricity" and node_to == "mobility":
            flows["Wallbox Ladung [kW]"] = v["sequences"]["flow"]
        # Heimspeicher-Flüsse werden in extract_battery_flows() behandelt
    
    return pd.DataFrame(flows)


def extract_battery_flows(results: dict, battery_label: str, prefix: str = "") -> pd.DataFrame:
    """Extrahiert Batterie-Flüsse und SOC"""
    node_data = views.node(results, battery_label)
    sequences = node_data["sequences"]
    
    flows = {}
    
    for col in sequences.columns:
        col_name = str(col[1]).lower() if len(col) > 1 else str(col).lower()
        
        # SOC
        if any(x in col_name for x in ['storage_content', 'soc', 'state_of_charge', 'content']):
            flows[f"{prefix} Ladezustand [kWh]"] = sequences[col]
        
        # Richtung prüfen
        node_from = str(col[0][0]) if isinstance(col[0], tuple) else ""
        node_to = str(col[0][1]) if isinstance(col[0], tuple) else ""
        
        if node_to == battery_label and 'flow' in col_name:
            flows[f"{prefix} Laden [kW]"] = sequences[col]
        elif node_from == battery_label and 'flow' in col_name:
            flows[f"{prefix} Entladen [kW]"] = sequences[col]
    
    return pd.DataFrame(flows)


def save_all_results(result_dict: dict, case_name: str):
    """Speichert alle Ergebnisse als CSV"""
    save_path = get_save_path()
    print(f"\n{'='*80}")
    print(f"SPEICHERE CSV-DATEIEN")
    print(f"{'='*80}")
    
    # Kombinierte Datei
    combined_file = Path(save_path) / f"{case_name}_complete.csv"
    result_dict['combined'].to_csv(combined_file, sep=',', encoding='utf-8')
    print(f"✓ {case_name}_complete.csv ({len(result_dict['combined'].columns)} Spalten)")
    
    # Elektrizität
    elec_file = Path(save_path) / f"{case_name}_electricity.csv"
    result_dict['electricity'].to_csv(elec_file, sep=',', encoding='utf-8')
    print(f"✓ {case_name}_electricity.csv")
    
    # BEV-Batterie
    if 'bev_battery' in result_dict:
        bev_file = Path(save_path) / f"{case_name}_bev_battery.csv"
        result_dict['bev_battery'].to_csv(bev_file, sep=',', encoding='utf-8')
        print(f"✓ {case_name}_bev_battery.csv")
    
    # Heimspeicher
    if 'home_battery' in result_dict:
        home_file = Path(save_path) / f"{case_name}_home_battery.csv"
        result_dict['home_battery'].to_csv(home_file, sep=',', encoding='utf-8')
        print(f"✓ {case_name}_home_battery.csv")
    
    print(f"\n✓ Gespeichert in: {save_path}")


def plot_interactive(result_dict: dict, case_name: str):
    """Erstellt interaktiven Plot (Klick auf Legend zum Ein-/Ausblenden)"""
    df = result_dict['combined']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f'{case_name} - Energieflüsse und Speicher', fontsize=16, fontweight='bold')
    
    colors = {
        'PV-Erzeugung [kW]': '#FDB462',
        'Netzbezug [kW]': '#FB8072',
        'Netzeinspeisung [kW]': '#8DD3C7',
        'Haushaltslast [kW]': '#BEBADA',
        'Wallbox Ladung [kW]': '#80B1D3',
        'Heimspeicher Laden [kW]': '#B3DE69',
        'Heimspeicher Entladen [kW]': '#FCCDE5',
        'BEV Ladezustand [kWh]': '#1f77b4',
        'Heimspeicher Ladezustand [kWh]': '#2ca02c',
    }
    
    # Plot 1: Energieflüsse
    flow_columns = [col for col in df.columns if '[kW]' in col and 'Ladezustand' not in col]
    lines1 = []
    
    for col in flow_columns:
        color = colors.get(col, None)
        line = ax1.plot(df.index, df[col], label=col, linewidth=1.5, color=color, alpha=0.8)[0]
        lines1.append(line)
    
    ax1.set_xlabel('Zeit', fontsize=12)
    ax1.set_ylabel('Leistung [kW]', fontsize=12)
    ax1.set_title('Energieflüsse', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=9, ncol=2)
    
    # Plot 2: Ladezustände
    soc_columns = [col for col in df.columns if 'Ladezustand' in col]
    lines2 = []
    
    for col in soc_columns:
        color = colors.get(col, None)
        line = ax2.plot(df.index, df[col], label=col, linewidth=2, color=color, alpha=0.8)[0]
        lines2.append(line)
    
    ax2.set_xlabel('Zeit', fontsize=12)
    ax2.set_ylabel('Energie [kWh]', fontsize=12)
    ax2.set_title('Speicher-Ladezustände', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)
    
    # Interaktivität
    def on_legend_click(event):
        for legline, origline in zip(ax1.get_legend().get_lines(), lines1):
            if legline == event.artist:
                visible = not origline.get_visible()
                origline.set_visible(visible)
                legline.set_alpha(1.0 if visible else 0.2)
                fig.canvas.draw()
        
        for legline, origline in zip(ax2.get_legend().get_lines(), lines2):
            if legline == event.artist:
                visible = not origline.get_visible()
                origline.set_visible(visible)
                legline.set_alpha(1.0 if visible else 0.2)
                fig.canvas.draw()
    
    for legend in [ax1.get_legend(), ax2.get_legend()]:
        for legline in legend.get_lines():
            legline.set_picker(True)
            legline.set_pickradius(5)
    
    fig.canvas.mpl_connect('pick_event', on_legend_click)
    plt.tight_layout()
    
    # Speichere Plot
    save_path = get_save_path()
    png_file = Path(save_path).parent / "PNG" / f"{case_name}_plot.png"
    png_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot gespeichert: {png_file}")
    
    plt.show()


if __name__ == "__main__":
    """
    Hauptprogramm
    
    Verwendung:
    1. Case auswählen
    2. Script ausführen
    3. Klick auf Legend-Einträge zum Ein-/Ausblenden
    """
    
    # ==================== KONFIGURATION ====================
    case_0 = "case00_pv_only"
    case_10 = "case10_pv+battery"
    case_12 = "case12_pv+BEV+speicher"
    
    case_to_study = case_12  # <-- HIER CASE WÄHLEN
    # ======================================================
    
    print("\n" + "="*80)
    print("ENERGIESYSTEM ANALYSE")
    print("="*80)
    print(f"Case: {case_to_study}")
    print("="*80)
    
    # 1. Restore und verarbeite
    results = restore_and_process_results(case_to_study)
    
    # 2. Speichere CSVs
    save_all_results(results, case_to_study)
    
    # 3. Erstelle Plot
    print(f"\n{'='*80}")
    print("ERSTELLE INTERAKTIVEN PLOT")
    print(f"{'='*80}")
    print("Hinweis: Klicke auf Legend-Einträge um Linien ein-/auszublenden!")
    plot_interactive(results, case_to_study)
    
    print(f"\n{'='*80}")
    print("✓ ANALYSE ABGESCHLOSSEN")
    print(f"{'='*80}\n")

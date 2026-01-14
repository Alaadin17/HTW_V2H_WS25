#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script contains the pv-module definition for the V2H system:
Jinko Tiger Neo N-Type 54HL4R-B 440 W (JKM440N-54HL4R-B).
"""

import pandas as pd
from pvlib import pvsystem, ivtools


def create_modules_df():
    """
    Creates a pandas dataframe with the correct parameter designation due to CEC method.

    Returns
    -------
    DataFrame
        Empty pandas DataFrame with index rows due to the CEC (California Energy Commission) convention.
    """
    module_param_keys = [
        # 'Name',
        "Technology",   # example: Mono-c-Si or Multi-c-Si
        "Bifacial",     # 0: not Bifacial, 1: Bifacial
        "STC",          # Power at STC in W
        "PTC",          # PVUSA Test Conditions power in W (if known)
        "A_c",          # Module area in m²
        "Length",       # Module length in m
        "Width",        # Module width in m
        "N_s",          # Number of cells in series
        "I_sc_ref",     # Short-circuit current at STC in A
        "V_oc_ref",     # Open-circuit voltage at STC in V
        "I_mp_ref",     # Current at MPP at STC in A
        "V_mp_ref",     # Voltage at MPP at STC in V
        "alpha_sc",     # Temp. coeff. of Isc in A/°C
        "beta_oc",      # Temp. coeff. of Voc in V/°C
        "T_NOCT",       # Nominal cell temperature in °C
        "a_ref",        # Diode factor * Ns * Vt at ref. conditions in V
        "I_L_ref",      # Light-generated current at reference conditions in A
        "I_o_ref",      # Diode reverse saturation current at reference conditions in A
        "R_s",          # Series resistance at reference conditions in Ohm
        "R_sh_ref",     # Shunt resistance at reference conditions in Ohm
        "Adjust",       # Adjustment to temp. coeff. of Isc in %
        "gamma_r",      # Temp. coeff. of Pmp in %/°C
        "BIPV",         # N: no, Y: yes
        "Version",      # Version
        "Date"          # Date
    ]
    modules = pd.DataFrame(index=module_param_keys)
    return modules


def modul1():
    """
    Creates a pandas Series of module 1:
    Jinko Tiger Neo N-Type 54HL4R-B 440 W (JKM440N-54HL4R-B).

    Returns
    -------
    pandas.Series
        Contains the module parameters due to CEC convention.
    """
    # --- 1) Manufacturer data (STC) ---
    # celltype for pvlib (CEC fitting)
    celltype = "monoSi"   # 'monoSi', 'multiSi', 'polySi', 'cis', 'cigs', 'cdte', 'amorphous'

    # Geometry (m)
    length = 1.762
    width = 1.134
    area = length * width   # ~1.998 m²

    # Electrical STC values (from datasheet JKM440N-54HL4R-B)
    stc = 440.0            # Pmp at STC in W
    v_mp = 32.99           # V at MPP (STC) in V
    i_mp = 13.34           # I at MPP (STC) in A
    v_oc = 39.57           # Voc (STC) in V
    i_sc = 13.80           # Isc (STC) in A

    # Temperature coefficients (datasheet gives in %/°C)
    # 0.045 %/°C * Isc -> A/°C
    alpha_sc = 0.00045 * i_sc          # A/°C  ≈ 0.00621
    # -0.25 %/°C * Voc -> V/°C
    beta_voc = -0.0025 * v_oc          # V/°C ≈ -0.0989
    gamma_pmp = -0.29                  # %/°C

    # Other physical data
    cells_in_series = 108              # 108 cells (6x18) – assumption as series cell count
    t_noct = 45.0                      # °C (NOCT)
    ptc = None                         # not given in datasheet
    bifacial = 0                       # all-black, not bifacial
    technology = "mono-si"             # free text
    temp_ref = 25.0                    # reference cell temperature for fit_cec_sam

    # --- 2) Fit CEC single-diode model parameters from STC data ---
    cec_params = ivtools.sdm.fit_cec_sam(
        celltype=celltype,
        v_mp=v_mp,
        i_mp=i_mp,
        v_oc=v_oc,
        i_sc=i_sc,
        alpha_sc=alpha_sc,
        beta_voc=beta_voc,
        gamma_pmp=gamma_pmp,
        cells_in_series=cells_in_series,
        temp_ref=temp_ref
    )
    # cec_params returns: I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust

    # --- 3) Assemble into CEC-style module dict ---
    modules = create_modules_df()

    module_1 = {
        # "Name": "Jinko_TigerNeo_440",  # Name is implicit via column label
        "Technology": technology,
        "Bifacial": bifacial,
        "STC": stc,
        "PTC": ptc,
        "A_c": area,
        "Length": length,
        "Width": width,
        "N_s": cells_in_series,
        "I_sc_ref": i_sc,
        "V_oc_ref": v_oc,
        "I_mp_ref": i_mp,
        "V_mp_ref": v_mp,
        "alpha_sc": alpha_sc,
        "beta_oc": beta_voc,
        "T_NOCT": t_noct,
        "a_ref": cec_params[4],
        "I_L_ref": cec_params[0],
        "I_o_ref": cec_params[1],
        "R_s": cec_params[2],
        "R_sh_ref": cec_params[3],
        "Adjust": cec_params[5],
        "gamma_r": gamma_pmp,
        "BIPV": "N",
        "Version": "",
        "Date": ""
    }

    modules["Jinko_TigerNeo_440"] = list(module_1.values())

    # Return the Series for this module

    return modules["Jinko_TigerNeo_440"]


if __name__ == "__main__":
    print("\n", modul1())

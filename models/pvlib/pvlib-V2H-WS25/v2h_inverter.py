#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script contains the pv-inverter model for the V2H system.
(Currently based on an example efficiency curve.)
"""

import numpy as np
from pvlib import inverter as pv_inverter   # <- WICHTIG: Alias verwenden


def inv1():
    """
    Return inverter parameters (Sandia model).
    """
    # inverter efficiency at different power points (source: PV*SOL)
    eta_min = [0, 0.953, 0.959, 0.963, 0.9612, 0.959]  # P/P_max = 0, 0.2, 0.3, 0.5, 0.75, 1; U = 210V
    eta_nom = [0, 0.961, 0.967, 0.971, 0.969, 0.967]  # P/P_max = 0, 0.2, 0.3, 0.5, 0.75, 1; U = 530V
    eta_max = [0, 0.952, 0.958, 0.962, 0.96, 0.958]   # P/P_max = 0, 0.2, 0.3, 0.5, 0.75, 1; U = 560V

    dc_voltage = [[230.], [350.], [480.]]
    p_dc_nom = 3750

    p_dc = [0,
            0.2 * p_dc_nom,
            0.3 * p_dc_nom,
            0.5 * p_dc_nom,
            0.75 * p_dc_nom,
            1.0 * p_dc_nom]

    p_ac_min = []
    p_ac_nom = []
    p_ac_max = []
    val_count = len(eta_min)

    for i in range(val_count):
        p_ac_min.append(p_dc[i] * eta_min[i])
        p_ac_nom.append(p_dc[i] * eta_nom[i])
        p_ac_max.append(p_dc[i] * eta_max[i])

    ac_power = np.array(p_ac_min + p_ac_nom + p_ac_max)

    sma_sb_data = {
        "ac_power": ac_power,
        "dc_power": np.array(p_dc + p_dc + p_dc),
        "dc_voltage": np.array(dc_voltage[0] * val_count +
                               dc_voltage[1] * val_count +
                               dc_voltage[2] * val_count),
        "dc_voltage_level": np.array(["Vmin"] * val_count +
                                     ["Vnom"] * val_count +
                                     ["Vmax"] * val_count),
        "p_ac_0": 2900.,
        "p_nt": 1.0,
    }

    inverter_1 = pv_inverter.fit_sandia(
        sma_sb_data["ac_power"],
        sma_sb_data["dc_power"],
        sma_sb_data["dc_voltage"],
        sma_sb_data["dc_voltage_level"],
        sma_sb_data["p_ac_0"],
        sma_sb_data["p_nt"],
    )
    return inverter_1


if __name__ == "__main__":
    print(inv1())

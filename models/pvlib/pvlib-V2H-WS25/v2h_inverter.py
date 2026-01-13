#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inverter model for the V2H system: SMA Sunny Tripower 10.0 (STP10.0-3AV-40)

We use pvlib's PVWatts inverter model because the datasheet provides
nominal AC power and efficiencies, but not a full Sandia/CEC efficiency map.

Datasheet (STP10.0):
- Nominal AC power: 10,000 W
- Max efficiency: 98.3% (0.983)
- European efficiency: 98.0% (0.980)
"""

def inv1(
    pac0: float = 10000.0,
    eta_euro: float = 0.980,
    eta_max: float = 0.983,
):
    """
    Return inverter parameters compatible with pvlib.inverter.pvwatts().

    Parameters
    ----------
    pac0 : float
        Nominal AC power in W (datasheet: 10,000 W).
    eta_euro : float
        European efficiency (datasheet: 98.0%).
        Used as pvwatts 'eta_inv_nom'.
    eta_max : float
        Maximum efficiency (datasheet: 98.3%).
        Used as pvwatts 'eta_inv_ref'.

    Returns
    -------
    dict
        Inverter parameters for PVWatts model:
        - pdc0: DC input limit (W)
        - eta_inv_nom: nominal efficiency (unitless)
        - eta_inv_ref: reference efficiency (unitless)
    """
    # PVWatts expects pdc0 as DC input limit. Choose it so that
    # at pdc = pdc0, pac clips close to pac0.
    # Using eta_max as reference efficiency is a pragmatic datasheet-based choice.
    pdc0 = pac0 / eta_max  # e.g. 10000/0.983 ≈ 10173 W

    inverter_parameters = {
        "pdc0": pdc0,
        "eta_inv_nom": eta_euro,
        "eta_inv_ref": eta_max,
    }
    return inverter_parameters


if __name__ == "__main__":
    print(inv1())

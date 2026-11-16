"""Benchmark parameters and original numerical settings."""

import json
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from common.paths import DATA_ROOT


@dataclass
class SchedulerParams:
    """SchedulerParams: benchmark parameters and original numerical settings."""

    name: str
    num_steps: int
    dt: float
    num_generators: int
    num_buses: int
    num_lines: int
    ref_bus: int

    P_min: List[float]
    P_max: List[float]
    ramp_up: List[float]
    ramp_down: List[float]
    startup_ramp: List[float]
    shutdown_ramp: List[float]
    min_up_time: List[int]
    min_down_time: List[int]
    u0: List[int]
    P0: List[float]
    initial_status_steps: List[int]
    gen_bus: List[int]

    cost_a: List[float]
    cost_b: List[float]
    cost_c: List[float]
    cost_startup: List[float]
    cost_shutdown: List[float]

    use_piecewise: bool
    num_segments: int
    pw_slopes: List[List[float]]
    pw_widths: List[List[float]]

    line_from: List[int]
    line_to: List[int]
    line_reactance: List[float]
    line_capacity: List[float]

    curtailment_cost: float = 50.0

    has_storage: bool = False
    storage_bus: int = 0
    E_max: float = 0.0
    P_ch_max: float = 0.0
    P_dis_max: float = 0.0
    SoC_min: float = 0.1
    SoC_max: float = 0.9
    SoC0: float = 0.5
    eta_ch: float = 0.95
    eta_dis: float = 0.95

    contingencies: List[int] = field(default_factory=list)
    emergency_capacity_factor: float = 1.2

    enable_loss: bool = False
    rx_ratio: List[float] = field(default_factory=list)


def _build_from_json(
    json_path,
    name,
    line_cap_default=500.0,
    line_cap_scale=1.0,
    enable_contingencies=False,
    emergency_capacity_factor=1.2,
    enable_loss=False,
    rx_ratio_range=(0.05, 0.25),
    rx_seed=99,
):
    """Load network data and derive the original generator and line parameters."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bus_names = sorted(data["Buses"].keys(), key=lambda x: int(x[1:]))
    bus_map = {bname: i for i, bname in enumerate(bus_names)}
    num_buses = len(bus_names)
    gen_names = sorted(data["Generators"].keys(), key=lambda x: int(x[1:]))
    num_generators = len(gen_names)
    gen_bus, P_min, P_max = [], [], []
    ramp_up, ramp_down, startup_ramp, shutdown_ramp = [], [], [], []
    min_up_time, min_down_time = [], []
    cost_a, cost_b, cost_c, cost_startup, cost_shutdown = [], [], [], [], []
    pw_slopes_all, pw_widths_all = [], []
    dt = 0.25
    for gname in gen_names:
        g = data["Generators"][gname]
        gen_bus.append(bus_map[g["Bus"]])
        mw_pts = g["Production cost curve (MW)"]
        cost_pts = g["Production cost curve ($)"]
        P_min.append(mw_pts[0])
        P_max.append(mw_pts[-1])
        slopes, widths = [], []
        for i in range(1, len(mw_pts)):
            dw = mw_pts[i] - mw_pts[i - 1]
            dc = cost_pts[i] - cost_pts[i - 1]
            slopes.append(round(dc / max(dw, 1e-6), 4))
            widths.append(round(dw, 4))
        pw_slopes_all.append(slopes)
        pw_widths_all.append(widths)
        avg_slope = (cost_pts[-1] - cost_pts[0]) / max(mw_pts[-1] - mw_pts[0], 1e-6)
        cost_a.append(0.0)
        cost_b.append(round(avg_slope, 4))
        cost_c.append(round(cost_pts[0], 4))
        ramp_up.append(round(g.get("Ramp up limit (MW)", mw_pts[-1]) * dt, 4))
        ramp_down.append(round(g.get("Ramp down limit (MW)", mw_pts[-1]) * dt, 4))
        startup_ramp.append(g.get("Startup limit (MW)", mw_pts[-1]))
        shutdown_ramp.append(g.get("Shutdown limit (MW)", mw_pts[-1]))
        min_up_h = g.get("Minimum uptime (h)", 1)
        min_down_h = g.get("Minimum downtime (h)", 1)
        min_up_time.append(min_up_h * 4)
        min_down_time.append(min_down_h * 4)
        su_costs = g.get("Startup costs ($)", [0.0])
        cost_startup.append(su_costs[0])
        cost_shutdown.append(0.0)
    valid_segs = [len(s) for s in pw_slopes_all if len(s) > 0]
    if valid_segs:
        num_segments = min(valid_segs)
    else:
        num_segments = 0
    for g in range(num_generators):
        if len(pw_slopes_all[g]) == 0:
            pw_slopes_all[g] = [0.0] * max(num_segments, 1)
            pw_widths_all[g] = [0.0] * max(num_segments, 1)
    if num_segments > 0:
        pw_slopes = [s[:num_segments] for s in pw_slopes_all]
        pw_widths = [w[:num_segments] for w in pw_widths_all]
        use_piecewise = True
    else:
        pw_slopes = [[0.0]] * num_generators
        pw_widths = [[0.0]] * num_generators
        num_segments = 1
        use_piecewise = False

    data_dir = os.path.dirname(json_path)
    bus_load_path = os.path.join(data_dir, "bus_load_uc.csv")
    if os.path.exists(bus_load_path):
        bl = pd.read_csv(bus_load_path)
        bus_cols = [c for c in bl.columns if c.startswith("bus_")]
        bl["total"] = bl[bus_cols].sum(axis=1)
        total_load_peak = float(bl["total"].max())
        total_load_valley = float(bl["total"].min())
    else:
        total_load_peak = 0
        total_load_valley = 0
        for bname in bus_names:
            load_val = data["Buses"][bname]["Load (MW)"]
            if isinstance(load_val, list):
                total_load_peak += max(load_val)
                total_load_valley += min(load_val)
            else:
                total_load_peak += load_val
                total_load_valley += load_val
    total_cap = sum(P_max)
    est_valley = total_load_valley
    est_peak_need = total_load_peak * 1.10
    gen_order = sorted(range(num_generators), key=lambda g: cost_b[g])
    cumsum = 0
    n_online = 0
    for g in gen_order:
        cumsum += P_max[g]
        n_online += 1
        if cumsum >= est_peak_need:
            break
    online_set = set(gen_order[:n_online])
    while True:
        online_Pmin = sum(P_min[g] for g in online_set)
        if online_Pmin <= est_valley * 0.9:
            break
        worst = max(online_set, key=lambda g: P_min[g])
        online_set.remove(worst)
        if len(online_set) == 0:
            break
    online_Pmax_check = sum(P_max[g] for g in online_set)
    if online_Pmax_check < est_peak_need:
        remaining = sorted(set(range(num_generators)) - online_set, key=lambda g: P_min[g])
        for g in remaining:
            online_set.add(g)
            online_Pmax_check += P_max[g]
            if online_Pmax_check >= est_peak_need:
                break
    u0, P0, initial_status_steps = [], [], []
    for g in range(num_generators):
        if g in online_set:
            u0.append(1)
            safe_p0 = min(shutdown_ramp[g] * 0.8, P_max[g] * 0.4)
            P0.append(round(max(safe_p0, P_min[g]), 4))
            initial_status_steps.append(int(24 / dt))
        else:
            u0.append(0)
            P0.append(0.0)
            initial_status_steps.append(-int(24 / dt))

    line_names = sorted(data["Transmission lines"].keys(), key=lambda x: int(x[1:]))
    line_name_to_idx = {lname: i for i, lname in enumerate(line_names)}
    num_lines = len(line_names)
    line_from, line_to, line_x, line_cap = [], [], [], []

    for lname in line_names:
        ln = data["Transmission lines"][lname]
        line_from.append(bus_map[ln["Source bus"]])
        line_to.append(bus_map[ln["Target bus"]])
        x = ln.get("Reactance (ohms)", 0.01)
        line_x.append(max(abs(x), 1e-4))
        raw_cap = ln.get("Normal flow limit (MW)", line_cap_default)
        line_cap.append(raw_cap * line_cap_scale)

    rx_ratio = []
    if enable_loss:
        rng = np.random.RandomState(rx_seed)
        rx_ratio = [
            round(rng.uniform(rx_ratio_range[0], rx_ratio_range[1]), 4) for _ in range(num_lines)
        ]

    contingencies = []
    if enable_contingencies and "Contingencies" in data:
        for cname in sorted(data["Contingencies"].keys(), key=lambda x: int(x[1:])):
            affected = data["Contingencies"][cname].get("Affected lines", [])
            if len(affected) == 1 and affected[0] in line_name_to_idx:
                contingencies.append(line_name_to_idx[affected[0]])

    return SchedulerParams(
        name=name,
        num_steps=96,
        dt=dt,
        num_generators=num_generators,
        num_buses=num_buses,
        num_lines=num_lines,
        ref_bus=0,
        gen_bus=gen_bus,
        P_min=P_min,
        P_max=P_max,
        ramp_up=ramp_up,
        ramp_down=ramp_down,
        startup_ramp=startup_ramp,
        shutdown_ramp=shutdown_ramp,
        min_up_time=min_up_time,
        min_down_time=min_down_time,
        u0=u0,
        P0=P0,
        initial_status_steps=initial_status_steps,
        cost_a=cost_a,
        cost_b=cost_b,
        cost_c=cost_c,
        cost_startup=cost_startup,
        cost_shutdown=cost_shutdown,
        use_piecewise=use_piecewise,
        num_segments=num_segments,
        pw_slopes=pw_slopes,
        pw_widths=pw_widths,
        line_from=line_from,
        line_to=line_to,
        line_reactance=line_x,
        line_capacity=line_cap,
        curtailment_cost=50.0,
        contingencies=contingencies,
        emergency_capacity_factor=emergency_capacity_factor,
        enable_loss=enable_loss,
        rx_ratio=rx_ratio,
    )


bus118 = _build_from_json(
    json_path=DATA_ROOT / "uc" / "118-bus" / "network_uc.json",
    name="118-bus",
    line_cap_default=200.0,
    enable_loss=True,
    rx_ratio_range=(0.03, 0.08),
)


bus1354 = _build_from_json(
    json_path=DATA_ROOT / "uc" / "1354-bus" / "network_uc.json",
    name="1354-bus",
    line_cap_default=2000.0,
    line_cap_scale=1.2,
    enable_loss=True,
    rx_ratio_range=(0.03, 0.08),
)

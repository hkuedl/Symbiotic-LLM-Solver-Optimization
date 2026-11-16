"""Benchmark parameters and original numerical settings."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SchedulerParams:
    """SchedulerParams: benchmark parameters and original numerical settings."""

    name: str
    num_steps: int
    dt: float
    num_lines: int
    num_processes: int

    # [line][process]
    eta: List[List[float]]
    power_per_ton: List[List[float]]
    production_rate_min: List[List[float]]
    production_rate_max: List[List[float]]
    ramp_rate: List[List[float]]
    u0: List[List[int]]
    cost_startup: List[List[float]]
    cost_shutdown: List[List[float]]
    on_time: List[List[int]]

    # Material inventory arrays have shape [line][process - 1].
    SoS0: List[List[float]]
    SoS_min: List[List[float]]
    SoS_max: List[List[float]]

    # Transfers move material between production lines with a time-step delay.
    transfer_time: int
    transfer_cost_per_ton: float
    transfer_rate_max: float

    # Tuples use (line_id, first_step, last_step), with inclusive endpoints.
    maintenance_windows: List[Tuple[int, int, int]]

    # Grid and electrical storage limits.
    P_grid_max: float
    E_max: float
    P_ch_max: float
    P_dis_max: float
    SoC_min: float
    SoC_max: float
    SoC0: float
    eta_ch: float
    eta_dis: float

    q_req_line_min: List[float]

    @property
    def num_buffers(self) -> int:
        return self.num_processes - 1


cement_plant = SchedulerParams(
    name="cement plant",
    num_steps=96,
    dt=0.25,
    num_lines=4,
    num_processes=4,
    eta=[
        [0.95, 0.98, 0.65, 1.05],
        [0.92, 0.96, 0.70, 1.08],
        [0.94, 0.97, 0.68, 1.02],
        [0.96, 0.99, 0.72, 1.04],
    ],
    power_per_ton=[
        [0.015, 0.025, 0.060, 0.040],
        [0.018, 0.028, 0.065, 0.038],
        [0.014, 0.024, 0.062, 0.042],
        [0.016, 0.026, 0.063, 0.041],
    ],
    production_rate_min=[
        [300.0, 150.0, 100.0, 120.0],
        [280.0, 140.0, 95.0, 115.0],
        [320.0, 160.0, 105.0, 125.0],
        [310.0, 155.0, 100.0, 122.0],
    ],
    production_rate_max=[
        [500.0, 250.0, 150.0, 220.0],
        [480.0, 240.0, 145.0, 210.0],
        [520.0, 260.0, 155.0, 230.0],
        [510.0, 255.0, 150.0, 225.0],
    ],
    ramp_rate=[
        [80.0, 40.0, 25.0, 40.0],
        [75.0, 38.0, 24.0, 38.0],
        [85.0, 42.0, 26.0, 42.0],
        [82.0, 40.0, 25.0, 41.0],
    ],
    u0=[
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ],
    cost_startup=[
        [30.0, 45.0, 60.0, 40.0],
        [35.0, 50.0, 65.0, 42.0],
        [28.0, 42.0, 62.0, 38.0],
        [32.0, 47.0, 63.0, 41.0],
    ],
    cost_shutdown=[
        [20.0, 25.0, 40.0, 20.0],
        [22.0, 28.0, 42.0, 22.0],
        [18.0, 24.0, 38.0, 18.0],
        [21.0, 26.0, 41.0, 21.0],
    ],
    on_time=[
        [4, 4, 6, 4],
        [4, 6, 6, 4],
        [4, 4, 8, 5],
        [4, 4, 6, 4],
    ],
    SoS0=[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
    SoS_min=[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
    SoS_max=[
        [800.0, 1000.0, 800.0],
        [750.0, 950.0, 780.0],
        [850.0, 1050.0, 820.0],
        [780.0, 980.0, 810.0],
    ],
    transfer_time=2,
    transfer_cost_per_ton=2.0,
    transfer_rate_max=120.0,
    maintenance_windows=[
        (1, 30, 35),
        (2, 60, 68),
    ],
    P_grid_max=80.0,
    E_max=40.0,
    P_ch_max=20.0,
    P_dis_max=20.0,
    SoC_min=0.1,
    SoC_max=0.9,
    SoC0=0.5,
    eta_ch=0.95,
    eta_dis=0.95,
    q_req_line_min=[500.0, 500.0, 500.0, 500.0],
)


steel_powder_plant = SchedulerParams(
    name="steel powder plant",
    num_steps=96,
    dt=0.25,
    num_lines=4,
    num_processes=8,
    eta=[
        [0.95, 1.08, 0.65, 1.05, 0.93, 0.97, 0.90, 0.92],
        [0.92, 1.06, 0.70, 1.08, 0.90, 0.95, 0.86, 0.90],
        [0.94, 1.07, 0.68, 1.02, 0.93, 0.96, 0.89, 0.93],
        [0.96, 1.09, 0.72, 1.04, 0.91, 0.98, 0.87, 0.91],
    ],
    power_per_ton=[
        [0.015, 0.025, 0.060, 0.040, 0.035, 0.020, 0.028, 0.022],
        [0.018, 0.028, 0.065, 0.038, 0.032, 0.022, 0.030, 0.024],
        [0.014, 0.024, 0.062, 0.042, 0.036, 0.019, 0.026, 0.020],
        [0.016, 0.026, 0.063, 0.041, 0.034, 0.021, 0.029, 0.023],
    ],
    production_rate_min=[
        [300, 150, 100, 120, 200, 180, 160, 130],
        [280, 140, 95, 115, 190, 170, 150, 125],
        [320, 160, 105, 125, 210, 190, 170, 145],
        [310, 155, 100, 122, 205, 185, 165, 135],
    ],
    production_rate_max=[
        [500, 250, 150, 220, 350, 300, 270, 230],
        [480, 240, 145, 210, 340, 290, 260, 220],
        [520, 260, 155, 230, 360, 310, 280, 240],
        [510, 255, 150, 225, 355, 305, 275, 235],
    ],
    ramp_rate=[
        [120, 80, 50, 70, 100, 90, 75, 55],
        [110, 75, 48, 65, 95, 85, 70, 50],
        [130, 85, 52, 75, 105, 95, 80, 60],
        [125, 80, 50, 72, 100, 90, 76, 58],
    ],
    u0=[
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    cost_startup=[
        [30, 45, 60, 40, 50, 35, 42, 48],
        [35, 50, 65, 42, 52, 38, 46, 52],
        [28, 42, 62, 38, 48, 33, 40, 45],
        [32, 47, 63, 41, 51, 36, 44, 50],
    ],
    cost_shutdown=[
        [20, 25, 40, 20, 30, 22, 26, 18],
        [22, 28, 42, 22, 32, 24, 28, 20],
        [18, 24, 38, 18, 28, 20, 24, 16],
        [21, 26, 41, 21, 31, 23, 27, 19],
    ],
    on_time=[
        [4, 4, 6, 4, 5, 4, 4, 6],
        [4, 6, 6, 4, 5, 4, 5, 6],
        [4, 4, 8, 5, 6, 4, 4, 7],
        [4, 4, 6, 4, 5, 4, 5, 6],
    ],
    SoS0=[
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    SoS_min=[
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    SoS_max=[
        [800, 1000, 800, 900, 850, 780, 700],
        [750, 950, 780, 880, 830, 750, 680],
        [850, 1050, 820, 920, 870, 800, 720],
        [780, 980, 810, 900, 850, 770, 690],
    ],
    transfer_time=2,
    transfer_cost_per_ton=2.0,
    transfer_rate_max=120.0,
    maintenance_windows=[
        (1, 30, 35),
        (2, 60, 68),
    ],
    P_grid_max=150.0,
    E_max=40.0,
    P_ch_max=20.0,
    P_dis_max=20.0,
    SoC_min=0.1,
    SoC_max=0.9,
    SoC0=0.5,
    eta_ch=0.95,
    eta_dis=0.95,
    q_req_line_min=[500.0, 500.0, 500.0, 500.0],
)

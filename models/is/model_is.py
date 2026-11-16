"""Pyomo variables, constraints, objective, and solution extraction."""

import pyomo.environ as pyo

from params import IndustrialParams as SchedulerParams


class IndustrialModel:
    """IndustrialModel: pyomo variables, constraints, objective, and solution extraction."""

    def __init__(self, params: SchedulerParams):
        self.p = params
        self.model = None

    def build(self, price, pv_output, q_req):
        """Construct the original optimization model for one operational day."""
        p = self.p
        T_set = range(p.num_steps)
        L_set = range(p.num_lines)
        K_set = range(p.num_processes)
        S_set = range(p.num_buffers)

        m = pyo.ConcreteModel()
        m.T = pyo.RangeSet(0, p.num_steps - 1)
        m.L = pyo.RangeSet(0, p.num_lines - 1)
        m.K = pyo.RangeSet(0, p.num_processes - 1)
        m.S = pyo.RangeSet(0, p.num_buffers - 1)

        m.dt = pyo.Param(initialize=p.dt)
        m.price = pyo.Param(m.T, initialize={t: price[t] for t in T_set})
        m.pv_output = pyo.Param(m.T, initialize={t: pv_output[t] for t in T_set})
        m.q_req_total = pyo.Param(initialize=q_req)

        m.eta = pyo.Param(m.L, m.K, initialize={(l, k): p.eta[l][k] for l in L_set for k in K_set})
        m.power_per_ton = pyo.Param(
            m.L, m.K, initialize={(l, k): p.power_per_ton[l][k] for l in L_set for k in K_set}
        )
        m.prod_min = pyo.Param(
            m.L, m.K, initialize={(l, k): p.production_rate_min[l][k] for l in L_set for k in K_set}
        )
        m.prod_max = pyo.Param(
            m.L, m.K, initialize={(l, k): p.production_rate_max[l][k] for l in L_set for k in K_set}
        )
        m.u0 = pyo.Param(
            m.L,
            m.K,
            within=pyo.Binary,
            initialize={(l, k): p.u0[l][k] for l in L_set for k in K_set},
        )
        m.cost_startup = pyo.Param(
            m.L, m.K, initialize={(l, k): p.cost_startup[l][k] for l in L_set for k in K_set}
        )
        m.cost_shutdown = pyo.Param(
            m.L, m.K, initialize={(l, k): p.cost_shutdown[l][k] for l in L_set for k in K_set}
        )
        m.on_time = pyo.Param(
            m.L, m.K, initialize={(l, k): p.on_time[l][k] for l in L_set for k in K_set}
        )
        m.ramp_rate = pyo.Param(
            m.L, m.K, initialize={(l, k): p.ramp_rate[l][k] for l in L_set for k in K_set}
        )

        m.SoS0 = pyo.Param(
            m.L, m.S, initialize={(l, s): p.SoS0[l][s] for l in L_set for s in S_set}
        )
        m.SoS_min = pyo.Param(
            m.L, m.S, initialize={(l, s): p.SoS_min[l][s] for l in L_set for s in S_set}
        )
        m.SoS_max = pyo.Param(
            m.L, m.S, initialize={(l, s): p.SoS_max[l][s] for l in L_set for s in S_set}
        )

        m.transfer_time = pyo.Param(initialize=p.transfer_time)
        m.transfer_cost = pyo.Param(initialize=p.transfer_cost_per_ton)
        m.transfer_rate_max = pyo.Param(initialize=p.transfer_rate_max)

        m.q_req_line_min = pyo.Param(m.L, initialize={l: p.q_req_line_min[l] for l in L_set})

        m.P_grid_max = pyo.Param(initialize=p.P_grid_max)
        m.E_bat = pyo.Param(initialize=p.E_max)
        m.P_ch_max = pyo.Param(initialize=p.P_ch_max)
        m.P_dis_max = pyo.Param(initialize=p.P_dis_max)
        m.SoC_min = pyo.Param(initialize=p.SoC_min)
        m.SoC_max = pyo.Param(initialize=p.SoC_max)
        m.SoC0 = pyo.Param(initialize=p.SoC0)
        m.eta_ch = pyo.Param(initialize=p.eta_ch)
        m.eta_dis = pyo.Param(initialize=p.eta_dis)

        m.u = pyo.Var(m.L, m.K, m.T, within=pyo.Binary)
        m.mu = pyo.Var(m.L, m.K, m.T, within=pyo.Binary)
        m.nu = pyo.Var(m.L, m.K, m.T, within=pyo.Binary)
        m.prod_in = pyo.Var(m.L, m.K, m.T, within=pyo.NonNegativeReals)
        m.prod_out = pyo.Var(m.L, m.K, m.T, within=pyo.NonNegativeReals)
        m.P_sub = pyo.Var(m.L, m.K, m.T, within=pyo.NonNegativeReals)

        m.SoS = pyo.Var(m.L, m.S, m.T, within=pyo.NonNegativeReals)
        m.transfer = pyo.Var(m.L, m.L, m.S, m.T, within=pyo.NonNegativeReals)
        m.Y = pyo.Var(m.L, within=pyo.NonNegativeReals)

        m.P_pv = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.P_ch = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.P_dis = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.SoC = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.u_ch = pyo.Var(m.T, within=pyo.Binary)
        m.u_dis = pyo.Var(m.T, within=pyo.Binary)
        m.P_grid = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.P_prod_total = pyo.Var(m.T, within=pyo.NonNegativeReals)

        # Couple each process's input and output through its conversion yield.
        def material_conversion_rule(m, l, k, t):
            return m.prod_out[l, k, t] == m.eta[l, k] * m.prod_in[l, k, t]

        m.MaterialConversion = pyo.Constraint(m.L, m.K, m.T, rule=material_conversion_rule)

        def prod_lower_rule(m, l, k, t):
            return m.prod_out[l, k, t] >= m.prod_min[l, k] * m.u[l, k, t]

        m.ProdLower = pyo.Constraint(m.L, m.K, m.T, rule=prod_lower_rule)

        def prod_upper_rule(m, l, k, t):
            return m.prod_out[l, k, t] <= m.prod_max[l, k] * m.u[l, k, t]

        m.ProdUpper = pyo.Constraint(m.L, m.K, m.T, rule=prod_upper_rule)

        def power_rule(m, l, k, t):
            return m.P_sub[l, k, t] == m.power_per_ton[l, k] * m.prod_out[l, k, t]

        m.PowerConsumption = pyo.Constraint(m.L, m.K, m.T, rule=power_rule)

        def startup_shutdown_rule(m, l, k, t):
            prev = m.u0[l, k] if t == 0 else m.u[l, k, t - 1]
            return m.u[l, k, t] - prev == m.mu[l, k, t] - m.nu[l, k, t]

        m.StartupShutdown = pyo.Constraint(m.L, m.K, m.T, rule=startup_shutdown_rule)

        def exclusive_rule(m, l, k, t):
            return m.mu[l, k, t] + m.nu[l, k, t] <= 1

        m.Exclusive = pyo.Constraint(m.L, m.K, m.T, rule=exclusive_rule)

        def min_on_time_rule(m, l, k, t):
            L_on = int(pyo.value(m.on_time[l, k]))
            if L_on <= 1 or t + L_on - 1 >= p.num_steps:
                return pyo.Constraint.Skip
            return sum(m.u[l, k, tau] for tau in range(t, t + L_on)) >= L_on * m.mu[l, k, t]

        m.MinOnTime = pyo.Constraint(m.L, m.K, m.T, rule=min_on_time_rule)

        # Material buffers include production flows and delayed inter-line transfers.
        def storage_dynamics_rule(m, l, s, t):
            prev = m.SoS0[l, s] if t == 0 else m.SoS[l, s, t - 1]
            inflow = m.prod_out[l, s, t] * m.dt
            outflow = m.prod_in[l, s + 1, t] * m.dt
            transfer_out = sum(m.transfer[l, j, s, t] * m.dt for j in L_set if j != l)
            t_arrival = t - int(pyo.value(m.transfer_time))
            if t_arrival >= 0:
                transfer_in = sum(m.transfer[i, l, s, t_arrival] * m.dt for i in L_set if i != l)
            else:
                transfer_in = 0
            return m.SoS[l, s, t] == prev + inflow - outflow - transfer_out + transfer_in

        m.StorageDynamics = pyo.Constraint(m.L, m.S, m.T, rule=storage_dynamics_rule)

        m.StorageLower = pyo.Constraint(
            m.L, m.S, m.T, rule=lambda m, l, s, t: m.SoS[l, s, t] >= m.SoS_min[l, s]
        )
        m.StorageUpper = pyo.Constraint(
            m.L, m.S, m.T, rule=lambda m, l, s, t: m.SoS[l, s, t] <= m.SoS_max[l, s]
        )

        def transfer_limit_rule(m, i, j, s, t):
            if i == j:
                return m.transfer[i, j, s, t] == 0
            return m.transfer[i, j, s, t] <= m.transfer_rate_max

        m.TransferLimit = pyo.Constraint(m.L, m.L, m.S, m.T, rule=transfer_limit_rule)

        def transfer_availability_rule(m, i, j, s, t):
            if i == j:
                return pyo.Constraint.Skip
            prev = m.SoS0[i, s] if t == 0 else m.SoS[i, s, t - 1]
            total_out = sum(m.transfer[i, j_prime, s, t] for j_prime in L_set if j_prime != i)
            return total_out * m.dt <= prev - m.SoS_min[i, s]

        m.TransferAvailability = pyo.Constraint(m.L, m.L, m.S, m.T, rule=transfer_availability_rule)

        for line_id, t_start, t_end in p.maintenance_windows:
            for k in K_set:
                for t in range(t_start, min(t_end + 1, p.num_steps)):
                    m.add_component(
                        f"Maintenance_L{line_id}_K{k}_T{t}",
                        pyo.Constraint(expr=m.u[line_id, k, t] == 0),
                    )

        def ramp_up_rule(m, l, k, t):
            if t == 0:
                return pyo.Constraint.Skip
            return (
                m.prod_out[l, k, t] - m.prod_out[l, k, t - 1]
                <= m.ramp_rate[l, k] + m.prod_max[l, k] * m.mu[l, k, t]
            )

        m.RampUp = pyo.Constraint(m.L, m.K, m.T, rule=ramp_up_rule)

        def ramp_down_rule(m, l, k, t):
            if t == 0:
                return pyo.Constraint.Skip
            return (
                m.prod_out[l, k, t - 1] - m.prod_out[l, k, t]
                <= m.ramp_rate[l, k] + m.prod_max[l, k] * m.nu[l, k, t]
            )

        m.RampDown = pyo.Constraint(m.L, m.K, m.T, rule=ramp_down_rule)

        def line_output_rule(m, l):
            last_proc = p.num_processes - 1
            return m.Y[l] == sum(m.prod_out[l, last_proc, t] * m.dt for t in m.T)

        m.LineOutput = pyo.Constraint(m.L, rule=line_output_rule)

        def line_min_output_rule(m, l):
            return m.Y[l] >= m.q_req_line_min[l]

        m.LineMinOutput = pyo.Constraint(m.L, rule=line_min_output_rule)

        def total_production_rule(m):
            return sum(m.Y[l] for l in m.L) >= m.q_req_total

        m.TotalProduction = pyo.Constraint(rule=total_production_rule)

        m.PVLimit = pyo.Constraint(m.T, rule=lambda m, t: m.P_pv[t] <= m.pv_output[t])

        # Battery energy accounting uses the original time-step and efficiencies.
        def SoC_dynamics_rule(m, t):
            prev = m.SoC0 if t == 0 else m.SoC[t - 1]
            return m.SoC[t] == (
                prev
                + m.eta_ch * m.P_ch[t] * m.dt / m.E_bat
                - m.P_dis[t] * m.dt / (m.eta_dis * m.E_bat)
            )

        m.SoCDynamics = pyo.Constraint(m.T, rule=SoC_dynamics_rule)

        m.SoCLower = pyo.Constraint(m.T, rule=lambda m, t: m.SoC[t] >= m.SoC_min)
        m.SoCUpper = pyo.Constraint(m.T, rule=lambda m, t: m.SoC[t] <= m.SoC_max)
        m.SoCTerminal = pyo.Constraint(rule=lambda m: m.SoC[p.num_steps - 1] == m.SoC0)

        m.ChargeLimit = pyo.Constraint(m.T, rule=lambda m, t: m.P_ch[t] <= m.u_ch[t] * m.P_ch_max)
        m.DischargeLimit = pyo.Constraint(
            m.T, rule=lambda m, t: m.P_dis[t] <= m.u_dis[t] * m.P_dis_max
        )
        m.ChargeDischargeExclusive = pyo.Constraint(
            m.T, rule=lambda m, t: m.u_ch[t] + m.u_dis[t] <= 1
        )

        def total_power_rule(m, t):
            return m.P_prod_total[t] == sum(m.P_sub[l, k, t] for l in m.L for k in m.K)

        m.TotalPower = pyo.Constraint(m.T, rule=total_power_rule)

        def power_balance_rule(m, t):
            return m.P_grid[t] + m.P_pv[t] + m.P_dis[t] == m.P_prod_total[t] + m.P_ch[t]

        m.PowerBalance = pyo.Constraint(m.T, rule=power_balance_rule)

        m.GridLimit = pyo.Constraint(m.T, rule=lambda m, t: m.P_grid[t] <= m.P_grid_max)

        # Account for purchased electricity, process transitions, and material transfer.
        def objective_rule(m):
            power_cost = sum(m.price[t] * m.P_grid[t] * m.dt for t in m.T)
            startup_cost = sum(
                m.cost_startup[l, k] * m.mu[l, k, t] for l in m.L for k in m.K for t in m.T
            )
            shutdown_cost = sum(
                m.cost_shutdown[l, k] * m.nu[l, k, t] for l in m.L for k in m.K for t in m.T
            )
            transfer_cost = sum(
                m.transfer_cost * m.transfer[i, j, s, t] * m.dt
                for i in m.L
                for j in m.L
                for s in m.S
                for t in m.T
                if i != j
            )
            return power_cost + startup_cost + shutdown_cost + transfer_cost

        m.Obj = pyo.Objective(sense=pyo.minimize, rule=objective_rule)

        self.model = m
        return m

    def extract_results(self, m=None):
        """Extract numerical arrays from the solved Pyomo model."""
        if m is None:
            m = self.model
        if m is None:
            raise RuntimeError("No model available.")

        p = self.p
        L_set = range(p.num_lines)
        K_set = range(p.num_processes)
        S_set = range(p.num_buffers)
        T_set = range(p.num_steps)

        return {
            "objective": pyo.value(m.Obj),
            "Y": [pyo.value(m.Y[l]) for l in L_set],
            "total_production": sum(pyo.value(m.Y[l]) for l in L_set),
            "P_grid": [pyo.value(m.P_grid[t]) for t in T_set],
            "P_pv": [pyo.value(m.P_pv[t]) for t in T_set],
            "P_ch": [pyo.value(m.P_ch[t]) for t in T_set],
            "P_dis": [pyo.value(m.P_dis[t]) for t in T_set],
            "SoC": [pyo.value(m.SoC[t]) for t in T_set],
            "P_prod_total": [pyo.value(m.P_prod_total[t]) for t in T_set],
            "u_ch": [pyo.value(m.u_ch[t]) for t in T_set],
            "u_dis": [pyo.value(m.u_dis[t]) for t in T_set],
            "u": [[[pyo.value(m.u[l, k, t]) for t in T_set] for k in K_set] for l in L_set],
            "mu": [[[pyo.value(m.mu[l, k, t]) for t in T_set] for k in K_set] for l in L_set],
            "nu": [[[pyo.value(m.nu[l, k, t]) for t in T_set] for k in K_set] for l in L_set],
            "prod_in": [
                [[pyo.value(m.prod_in[l, k, t]) for t in T_set] for k in K_set] for l in L_set
            ],
            "prod_out": [
                [[pyo.value(m.prod_out[l, k, t]) for t in T_set] for k in K_set] for l in L_set
            ],
            "P_sub": [[[pyo.value(m.P_sub[l, k, t]) for t in T_set] for k in K_set] for l in L_set],
            "SoS": [[[pyo.value(m.SoS[l, s, t]) for t in T_set] for s in S_set] for l in L_set],
            "transfer": [
                [[[pyo.value(m.transfer[i, j, s, t]) for t in T_set] for s in S_set] for j in L_set]
                for i in L_set
            ],
        }

    def cost_breakdown(self, m=None):
        if m is None:
            m = self.model
        p = self.p
        L_set = range(p.num_lines)
        K_set = range(p.num_processes)
        S_set = range(p.num_buffers)
        T_set = range(p.num_steps)

        power_cost = sum(pyo.value(m.price[t]) * pyo.value(m.P_grid[t]) * p.dt for t in T_set)
        startup_cost = sum(
            pyo.value(m.cost_startup[l, k]) * pyo.value(m.mu[l, k, t])
            for l in L_set
            for k in K_set
            for t in T_set
        )
        shutdown_cost = sum(
            pyo.value(m.cost_shutdown[l, k]) * pyo.value(m.nu[l, k, t])
            for l in L_set
            for k in K_set
            for t in T_set
        )
        transfer_cost = sum(
            p.transfer_cost_per_ton * pyo.value(m.transfer[i, j, s, t]) * p.dt
            for i in L_set
            for j in L_set
            for s in S_set
            for t in T_set
            if i != j
        )

        return {
            "power_cost": power_cost,
            "startup_cost": startup_cost,
            "shutdown_cost": shutdown_cost,
            "transfer_cost": transfer_cost,
            "total_cost": power_cost + startup_cost + shutdown_cost + transfer_cost,
        }

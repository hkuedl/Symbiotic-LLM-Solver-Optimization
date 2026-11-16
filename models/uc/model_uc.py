"""Pyomo variables, constraints, objective, and solution extraction."""

import numpy as np
import pyomo.environ as pyo

from params.uc.params_uc import SchedulerParams


class UnitCommitmentModel:
    """UnitCommitmentModel: pyomo variables, constraints, objective, and solution extraction."""

    def __init__(self, params: SchedulerParams):
        self.p = params
        self.model = None
        self._wind_buses = []
        self._solar_buses = []

    def _compute_lodf(self):
        p = self.p
        N = p.num_buses
        K = p.num_lines
        ref = p.ref_bus

        B = np.zeros((N, N))
        for k in range(K):
            i, j = p.line_from[k], p.line_to[k]
            b = 1.0 / p.line_reactance[k]
            B[i, i] += b
            B[j, j] += b
            B[i, j] -= b
            B[j, i] -= b

        non_ref = [n for n in range(N) if n != ref]
        B_red = B[np.ix_(non_ref, non_ref)]
        B_red_inv = np.linalg.inv(B_red)
        idx_map = {n: i for i, n in enumerate(non_ref)}

        def get_inv(i, j):
            if i == ref or j == ref:
                return 0.0
            return B_red_inv[idx_map[i], idx_map[j]]

        PTDF = np.zeros((K, N))
        for k in range(K):
            fi, fj = p.line_from[k], p.line_to[k]
            bk = 1.0 / p.line_reactance[k]
            for n in range(N):
                PTDF[k, n] = bk * (get_inv(fi, n) - get_inv(fj, n))

        LODF = np.zeros((K, K))
        for c in range(K):
            fc, tc = p.line_from[c], p.line_to[c]
            denom = 1.0 - (PTDF[c, fc] - PTDF[c, tc])
            if abs(denom) < 1e-10:
                LODF[:, c] = 0.0
                continue
            for l in range(K):
                if l == c:
                    LODF[l, c] = -1.0
                else:
                    LODF[l, c] = (PTDF[l, fc] - PTDF[l, tc]) / denom
        return LODF

    def build(self, demand_bus, reserve, wind=None, solar=None):
        """Construct the original optimization model for one operational day."""
        p = self.p
        T = p.num_steps
        G_set = range(p.num_generators)
        T_set = range(T)
        N_set = range(p.num_buses)
        K_set = range(p.num_lines)

        demand_total = [sum(demand_bus[n][t] for n in N_set) for t in T_set]

        has_wind = wind is not None and len(wind) > 0
        has_solar = solar is not None and len(solar) > 0

        wind_buses = sorted(wind.keys()) if has_wind else []
        solar_buses = sorted(solar.keys()) if has_solar else []
        self._wind_buses = wind_buses
        self._solar_buses = solar_buses

        m = pyo.ConcreteModel(name=p.name)

        m.G = pyo.RangeSet(0, p.num_generators - 1)
        m.T = pyo.RangeSet(0, T - 1)
        m.N = pyo.RangeSet(0, p.num_buses - 1)
        m.K = pyo.RangeSet(0, p.num_lines - 1)

        m.dt = pyo.Param(initialize=p.dt)
        m.P_min = pyo.Param(m.G, initialize={g: p.P_min[g] for g in G_set})
        m.P_max = pyo.Param(m.G, initialize={g: p.P_max[g] for g in G_set})
        m.ramp_up = pyo.Param(m.G, initialize={g: p.ramp_up[g] for g in G_set})
        m.ramp_down = pyo.Param(m.G, initialize={g: p.ramp_down[g] for g in G_set})
        m.startup_ramp = pyo.Param(m.G, initialize={g: p.startup_ramp[g] for g in G_set})
        m.shutdown_ramp = pyo.Param(m.G, initialize={g: p.shutdown_ramp[g] for g in G_set})
        m.min_up = pyo.Param(m.G, initialize={g: p.min_up_time[g] for g in G_set})
        m.min_down = pyo.Param(m.G, initialize={g: p.min_down_time[g] for g in G_set})
        m.u0 = pyo.Param(m.G, within=pyo.Binary, initialize={g: p.u0[g] for g in G_set})
        m.P0 = pyo.Param(m.G, initialize={g: p.P0[g] for g in G_set})
        m.cost_a = pyo.Param(m.G, initialize={g: p.cost_a[g] for g in G_set})
        m.cost_b = pyo.Param(m.G, initialize={g: p.cost_b[g] for g in G_set})
        m.cost_c = pyo.Param(m.G, initialize={g: p.cost_c[g] for g in G_set})
        m.cost_startup = pyo.Param(m.G, initialize={g: p.cost_startup[g] for g in G_set})
        m.cost_shutdown = pyo.Param(m.G, initialize={g: p.cost_shutdown[g] for g in G_set})

        m.demand_bus = pyo.Param(
            m.N, m.T, initialize={(n, t): demand_bus[n][t] for n in N_set for t in T_set}
        )
        m.reserve = pyo.Param(m.T, initialize={t: reserve[t] for t in T_set})
        m.demand_total = pyo.Param(m.T, initialize={t: demand_total[t] for t in T_set})

        m.line_x = pyo.Param(m.K, initialize={k: p.line_reactance[k] for k in K_set})
        m.line_cap = pyo.Param(m.K, initialize={k: p.line_capacity[k] for k in K_set})

        m.u = pyo.Var(m.G, m.T, within=pyo.Binary)
        m.v = pyo.Var(m.G, m.T, within=pyo.Binary)
        m.w = pyo.Var(m.G, m.T, within=pyo.Binary)
        m.P = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals)
        m.theta = pyo.Var(m.N, m.T, within=pyo.Reals)
        m.f = pyo.Var(m.K, m.T, within=pyo.Reals)

        if has_wind:
            n_wf = len(wind_buses)
            m.WF = pyo.RangeSet(0, n_wf - 1)
            m.wind_avail = pyo.Param(
                m.WF,
                m.T,
                initialize={(wf, t): wind[wind_buses[wf]][t] for wf in range(n_wf) for t in T_set},
            )
            m.P_wind = pyo.Var(m.WF, m.T, within=pyo.NonNegativeReals)
            m.wind_curt = pyo.Var(m.WF, m.T, within=pyo.NonNegativeReals)

        if has_solar:
            n_sf = len(solar_buses)
            m.SF = pyo.RangeSet(0, n_sf - 1)
            m.solar_avail = pyo.Param(
                m.SF,
                m.T,
                initialize={
                    (sf, t): solar[solar_buses[sf]][t] for sf in range(n_sf) for t in T_set
                },
            )
            m.P_solar = pyo.Var(m.SF, m.T, within=pyo.NonNegativeReals)
            m.solar_curt = pyo.Var(m.SF, m.T, within=pyo.NonNegativeReals)

        if p.has_storage:
            m.P_ch = pyo.Var(m.T, within=pyo.NonNegativeReals)
            m.P_dis = pyo.Var(m.T, within=pyo.NonNegativeReals)
            m.SoC = pyo.Var(m.T, within=pyo.NonNegativeReals)
            m.u_ch = pyo.Var(m.T, within=pyo.Binary)
            m.u_dis = pyo.Var(m.T, within=pyo.Binary)

        if p.use_piecewise and p.pw_slopes is not None:
            m.SEG = pyo.RangeSet(0, p.num_segments - 1)
            m.delta = pyo.Var(m.G, m.SEG, m.T, within=pyo.NonNegativeReals)

        gen_at_bus = {n: [] for n in N_set}
        for g in G_set:
            gen_at_bus[p.gen_bus[g]].append(g)

        wind_at_bus = {}
        for wf, bus in enumerate(wind_buses):
            wind_at_bus.setdefault(bus, []).append(wf)

        solar_at_bus = {}
        for sf, bus in enumerate(solar_buses):
            solar_at_bus.setdefault(bus, []).append(sf)

        # Link commitment transitions to startup and shutdown decisions.
        def startup_shutdown_rule(m, g, t):
            prev = m.u0[g] if t == 0 else m.u[g, t - 1]
            return m.u[g, t] - prev == m.v[g, t] - m.w[g, t]

        m.StartupShutdown = pyo.Constraint(m.G, m.T, rule=startup_shutdown_rule)

        def exclusive_rule(m, g, t):
            return m.v[g, t] + m.w[g, t] <= 1

        m.Exclusive = pyo.Constraint(m.G, m.T, rule=exclusive_rule)

        def min_up_time_rule(m, g, t):
            UT = int(pyo.value(m.min_up[g]))
            if UT <= 1 or t + UT > T:
                return pyo.Constraint.Skip
            return sum(m.u[g, tau] for tau in range(t, t + UT)) >= UT * m.v[g, t]

        m.MinUpTime = pyo.Constraint(m.G, m.T, rule=min_up_time_rule)

        def min_down_time_rule(m, g, t):
            DT = int(pyo.value(m.min_down[g]))
            if DT <= 1 or t + DT > T:
                return pyo.Constraint.Skip
            return sum(1 - m.u[g, tau] for tau in range(t, t + DT)) >= DT * m.w[g, t]

        m.MinDownTime = pyo.Constraint(m.G, m.T, rule=min_down_time_rule)

        for g in G_set:
            status_h = p.initial_status_steps[g]
            if status_h > 0:
                remaining = max(0, p.min_up_time[g] - status_h)
                for t in range(min(remaining, T)):
                    m.add_component(f"InitUp_G{g}_T{t}", pyo.Constraint(expr=m.u[g, t] == 1))
            elif status_h < 0:
                remaining = max(0, p.min_down_time[g] - abs(status_h))
                for t in range(min(remaining, T)):
                    m.add_component(f"InitDown_G{g}_T{t}", pyo.Constraint(expr=m.u[g, t] == 0))

        def power_lower_rule(m, g, t):
            return m.P[g, t] >= m.P_min[g] * m.u[g, t]

        m.PowerLower = pyo.Constraint(m.G, m.T, rule=power_lower_rule)

        def power_upper_rule(m, g, t):
            return m.P[g, t] <= m.P_max[g] * m.u[g, t]

        m.PowerUpper = pyo.Constraint(m.G, m.T, rule=power_upper_rule)

        def ramp_up_rule(m, g, t):
            if t == 0:
                prev_P = m.P0[g]
                prev_u = m.u0[g]
            else:
                prev_P = m.P[g, t - 1]
                prev_u = m.u[g, t - 1]
            return m.P[g, t] - prev_P <= (m.ramp_up[g] * prev_u + m.startup_ramp[g] * m.v[g, t])

        m.RampUp = pyo.Constraint(m.G, m.T, rule=ramp_up_rule)

        def ramp_down_rule(m, g, t):
            if t == 0:
                prev_P = m.P0[g]
            else:
                prev_P = m.P[g, t - 1]
            return prev_P - m.P[g, t] <= (
                m.ramp_down[g] * m.u[g, t] + m.shutdown_ramp[g] * m.w[g, t]
            )

        m.RampDown = pyo.Constraint(m.G, m.T, rule=ramp_down_rule)

        def ref_angle_rule(m, t):
            return m.theta[p.ref_bus, t] == 0

        m.RefAngle = pyo.Constraint(m.T, rule=ref_angle_rule)

        # Preserve the original DC-flow convention and line-capacity scaling.
        def dc_flow_rule(m, k, t):
            b_k = 1.0 / pyo.value(m.line_x[k])
            return m.f[k, t] == b_k * (m.theta[p.line_from[k], t] - m.theta[p.line_to[k], t])

        m.DCFlow = pyo.Constraint(m.K, m.T, rule=dc_flow_rule)

        def line_cap_pos_rule(m, k, t):
            return m.f[k, t] <= m.line_cap[k]

        m.LineCapPos = pyo.Constraint(m.K, m.T, rule=line_cap_pos_rule)

        def line_cap_neg_rule(m, k, t):
            return -m.line_cap[k] <= m.f[k, t]

        m.LineCapNeg = pyo.Constraint(m.K, m.T, rule=line_cap_neg_rule)

        if p.enable_loss and len(p.rx_ratio) == p.num_lines:
            m.f_pos = pyo.Var(m.K, m.T, within=pyo.NonNegativeReals)
            m.f_neg = pyo.Var(m.K, m.T, within=pyo.NonNegativeReals)
            m.loss = pyo.Var(m.K, m.T, within=pyo.NonNegativeReals)

            m.rx_ratio = pyo.Param(m.K, initialize={k: p.rx_ratio[k] for k in K_set})

            def f_split_rule(m, k, t):
                return m.f[k, t] == m.f_pos[k, t] - m.f_neg[k, t]

            m.FlowSplit = pyo.Constraint(m.K, m.T, rule=f_split_rule)

            def f_pos_limit(m, k, t):
                return m.f_pos[k, t] <= m.line_cap[k]

            m.FlowPosLimit = pyo.Constraint(m.K, m.T, rule=f_pos_limit)

            def f_neg_limit(m, k, t):
                return m.f_neg[k, t] <= m.line_cap[k]

            m.FlowNegLimit = pyo.Constraint(m.K, m.T, rule=f_neg_limit)

            def loss_rule(m, k, t):
                return m.loss[k, t] == m.rx_ratio[k] * (m.f_pos[k, t] + m.f_neg[k, t])

            m.LossCalc = pyo.Constraint(m.K, m.T, rule=loss_rule)

            lines_at_bus = {n: [] for n in N_set}
            for k in K_set:
                lines_at_bus[p.line_from[k]].append(k)
                lines_at_bus[p.line_to[k]].append(k)

            def nodal_balance_rule(m, n, t):
                gen_inj = sum(m.P[g, t] for g in gen_at_bus[n])
                re_inj = 0
                if has_wind and n in wind_at_bus:
                    for wf in wind_at_bus[n]:
                        re_inj += m.P_wind[wf, t]
                if has_solar and n in solar_at_bus:
                    for sf in solar_at_bus[n]:
                        re_inj += m.P_solar[sf, t]
                stor_inj = 0
                if p.has_storage and n == p.storage_bus:
                    stor_inj = m.P_dis[t] - m.P_ch[t]
                flow_out = sum(m.f[k, t] for k in K_set if p.line_from[k] == n)
                flow_in = sum(m.f[k, t] for k in K_set if p.line_to[k] == n)
                bus_loss = sum(0.5 * m.loss[k, t] for k in lines_at_bus[n])
                return (
                    gen_inj + re_inj + stor_inj - m.demand_bus[n, t]
                    == flow_out - flow_in + bus_loss
                )

            m.NodalBalance = pyo.Constraint(m.N, m.T, rule=nodal_balance_rule)

        else:

            def nodal_balance_rule(m, n, t):
                gen_inj = sum(m.P[g, t] for g in gen_at_bus[n])
                re_inj = 0
                if has_wind and n in wind_at_bus:
                    for wf in wind_at_bus[n]:
                        re_inj += m.P_wind[wf, t]
                if has_solar and n in solar_at_bus:
                    for sf in solar_at_bus[n]:
                        re_inj += m.P_solar[sf, t]
                stor_inj = 0
                if p.has_storage and n == p.storage_bus:
                    stor_inj = m.P_dis[t] - m.P_ch[t]
                flow_out = sum(m.f[k, t] for k in K_set if p.line_from[k] == n)
                flow_in = sum(m.f[k, t] for k in K_set if p.line_to[k] == n)
                return gen_inj + re_inj + stor_inj - m.demand_bus[n, t] == flow_out - flow_in

            m.NodalBalance = pyo.Constraint(m.N, m.T, rule=nodal_balance_rule)

        # Committed capacity must cover demand plus the required reserve.
        def reserve_rule(m, t):
            return sum(m.P_max[g] * m.u[g, t] for g in m.G) >= (m.demand_total[t] + m.reserve[t])

        m.ReserveReq = pyo.Constraint(m.T, rule=reserve_rule)

        if has_wind:
            m.WindLimit = pyo.Constraint(
                m.WF,
                m.T,
                rule=lambda m, wf, t: m.P_wind[wf, t] + m.wind_curt[wf, t] == m.wind_avail[wf, t],
            )

        if has_solar:
            m.SolarLimit = pyo.Constraint(
                m.SF,
                m.T,
                rule=lambda m, sf, t: (
                    m.P_solar[sf, t] + m.solar_curt[sf, t] == m.solar_avail[sf, t]
                ),
            )

        if p.has_storage:

            def soc_rule(m, t):
                prev = p.SoC0 if t == 0 else m.SoC[t - 1]
                return m.SoC[t] == (
                    prev
                    + p.eta_ch * m.P_ch[t] * m.dt / p.E_max
                    - m.P_dis[t] * m.dt / (p.eta_dis * p.E_max)
                )

            m.SoCDynamics = pyo.Constraint(m.T, rule=soc_rule)
            m.SoCLower = pyo.Constraint(m.T, rule=lambda m, t: m.SoC[t] >= p.SoC_min)
            m.SoCUpper = pyo.Constraint(m.T, rule=lambda m, t: m.SoC[t] <= p.SoC_max)
            m.SoCTerminal = pyo.Constraint(rule=lambda m: m.SoC[T - 1] == p.SoC0)
            m.ChLimit = pyo.Constraint(m.T, rule=lambda m, t: m.P_ch[t] <= m.u_ch[t] * p.P_ch_max)
            m.DisLimit = pyo.Constraint(
                m.T, rule=lambda m, t: m.P_dis[t] <= m.u_dis[t] * p.P_dis_max
            )
            m.ChDisExcl = pyo.Constraint(m.T, rule=lambda m, t: m.u_ch[t] + m.u_dis[t] <= 1)

        if p.use_piecewise and p.pw_slopes is not None:
            S_set = range(p.num_segments)

            def seg_width_rule(m, g, s, t):
                return m.delta[g, s, t] <= p.pw_widths[g][s] * m.u[g, t]

            m.SegWidth = pyo.Constraint(m.G, m.SEG, m.T, rule=seg_width_rule)

            def seg_sum_rule(m, g, t):
                return m.P[g, t] == m.P_min[g] * m.u[g, t] + sum(m.delta[g, s, t] for s in S_set)

            m.SegSum = pyo.Constraint(m.G, m.T, rule=seg_sum_rule)

        if len(p.contingencies) > 0:
            print(f"[N-1] Computing LODF for {len(p.contingencies)} contingencies...")
            LODF = self._compute_lodf()
            em_factor = p.emergency_capacity_factor
            LODF_THRESHOLD = 1e-4

            for c_idx, c_line in enumerate(p.contingencies):
                for l in K_set:
                    if l == c_line:
                        continue
                    lodf_val = LODF[l, c_line]
                    if abs(lodf_val) < LODF_THRESHOLD:
                        continue
                    em_cap = p.line_capacity[l] * em_factor
                    for t in T_set:
                        m.add_component(
                            f"SC_pos_c{c_idx}_l{l}_t{t}",
                            pyo.Constraint(expr=m.f[l, t] + lodf_val * m.f[c_line, t] <= em_cap),
                        )
                        m.add_component(
                            f"SC_neg_c{c_idx}_l{l}_t{t}",
                            pyo.Constraint(expr=-(m.f[l, t] + lodf_val * m.f[c_line, t]) <= em_cap),
                        )

            n_sc = sum(1 for c in m.component_objects(pyo.Constraint) if c.name.startswith("SC_"))
            print(f"[N-1] Added {n_sc} security constraints")

        # Combine operating, switching, and renewable-curtailment costs.
        def objective_rule(m):
            if p.use_piecewise and p.pw_slopes is not None:
                fuel_cost = (
                    sum(
                        m.cost_c[g] * m.u[g, t]
                        + sum(p.pw_slopes[g][s] * m.delta[g, s, t] for s in range(p.num_segments))
                        for g in m.G
                        for t in m.T
                    )
                    * m.dt
                )
            else:
                fuel_cost = sum(
                    (
                        m.cost_a[g] * m.P_max[g] * m.P[g, t]
                        + m.cost_b[g] * m.P[g, t]
                        + m.cost_c[g] * m.u[g, t]
                    )
                    * m.dt
                    for g in m.G
                    for t in m.T
                )
            startup_cost = sum(m.cost_startup[g] * m.v[g, t] for g in m.G for t in m.T)
            shutdown_cost = sum(m.cost_shutdown[g] * m.w[g, t] for g in m.G for t in m.T)
            curt_cost = 0
            if has_wind:
                curt_cost += sum(
                    p.curtailment_cost * m.wind_curt[wf, t] * m.dt for wf in m.WF for t in m.T
                )
            if has_solar:
                curt_cost += sum(
                    p.curtailment_cost * m.solar_curt[sf, t] * m.dt for sf in m.SF for t in m.T
                )
            return fuel_cost + startup_cost + shutdown_cost + curt_cost

        m.Obj = pyo.Objective(sense=pyo.minimize, rule=objective_rule)

        self.model = m
        return m

    def extract_results(self, m=None):
        """Extract numerical arrays from the solved Pyomo model."""
        if m is None:
            m = self.model
        p = self.p
        G_set = range(p.num_generators)
        T_set = range(p.num_steps)
        K_set = range(p.num_lines)
        N_set = range(p.num_buses)

        results = {
            "objective": pyo.value(m.Obj),
            "u": [[pyo.value(m.u[g, t]) for t in T_set] for g in G_set],
            "v": [[pyo.value(m.v[g, t]) for t in T_set] for g in G_set],
            "w": [[pyo.value(m.w[g, t]) for t in T_set] for g in G_set],
            "P": [[pyo.value(m.P[g, t]) for t in T_set] for g in G_set],
            "total_gen": [sum(pyo.value(m.P[g, t]) for g in G_set) for t in T_set],
            "demand": [pyo.value(m.demand_total[t]) for t in T_set],
            "theta": [[pyo.value(m.theta[n, t]) for t in T_set] for n in N_set],
            "f": [[pyo.value(m.f[k, t]) for t in T_set] for k in K_set],
        }

        if p.enable_loss and hasattr(m, "loss"):
            results["total_loss"] = [sum(pyo.value(m.loss[k, t]) for k in K_set) for t in T_set]

        if hasattr(m, "P_wind"):
            results["wind_buses"] = self._wind_buses
            results["P_wind"] = {
                self._wind_buses[wf]: [pyo.value(m.P_wind[wf, t]) for t in T_set]
                for wf in range(len(self._wind_buses))
            }
            results["wind_curt"] = {
                self._wind_buses[wf]: [pyo.value(m.wind_curt[wf, t]) for t in T_set]
                for wf in range(len(self._wind_buses))
            }
            results["P_wind_total"] = [
                sum(pyo.value(m.P_wind[wf, t]) for wf in range(len(self._wind_buses)))
                for t in T_set
            ]
            results["wind_curt_total"] = [
                sum(pyo.value(m.wind_curt[wf, t]) for wf in range(len(self._wind_buses)))
                for t in T_set
            ]

        if hasattr(m, "P_solar"):
            results["solar_buses"] = self._solar_buses
            results["P_solar"] = {
                self._solar_buses[sf]: [pyo.value(m.P_solar[sf, t]) for t in T_set]
                for sf in range(len(self._solar_buses))
            }
            results["solar_curt"] = {
                self._solar_buses[sf]: [pyo.value(m.solar_curt[sf, t]) for t in T_set]
                for sf in range(len(self._solar_buses))
            }
            results["P_solar_total"] = [
                sum(pyo.value(m.P_solar[sf, t]) for sf in range(len(self._solar_buses)))
                for t in T_set
            ]
            results["solar_curt_total"] = [
                sum(pyo.value(m.solar_curt[sf, t]) for sf in range(len(self._solar_buses)))
                for t in T_set
            ]

        if p.has_storage:
            results["P_ch"] = [pyo.value(m.P_ch[t]) for t in T_set]
            results["P_dis"] = [pyo.value(m.P_dis[t]) for t in T_set]
            results["SoC"] = [pyo.value(m.SoC[t]) for t in T_set]

        return results

    def cost_breakdown(self, m=None):
        if m is None:
            m = self.model
        p = self.p
        G_set = range(p.num_generators)
        T_set = range(p.num_steps)

        if p.use_piecewise and p.pw_slopes is not None:
            fuel_cost = sum(
                (
                    pyo.value(m.cost_c[g]) * pyo.value(m.u[g, t])
                    + sum(
                        p.pw_slopes[g][s] * pyo.value(m.delta[g, s, t])
                        for s in range(p.num_segments)
                    )
                )
                * p.dt
                for g in G_set
                for t in T_set
            )
        else:
            fuel_cost = sum(
                (
                    pyo.value(m.cost_a[g]) * pyo.value(m.P_max[g]) * pyo.value(m.P[g, t])
                    + pyo.value(m.cost_b[g]) * pyo.value(m.P[g, t])
                    + pyo.value(m.cost_c[g]) * pyo.value(m.u[g, t])
                )
                * p.dt
                for g in G_set
                for t in T_set
            )

        startup_cost = sum(
            pyo.value(m.cost_startup[g]) * pyo.value(m.v[g, t]) for g in G_set for t in T_set
        )
        shutdown_cost = sum(
            pyo.value(m.cost_shutdown[g]) * pyo.value(m.w[g, t]) for g in G_set for t in T_set
        )

        curt_cost = 0
        if hasattr(m, "wind_curt"):
            curt_cost += sum(
                p.curtailment_cost * pyo.value(m.wind_curt[wf, t]) * p.dt
                for wf in range(len(self._wind_buses))
                for t in T_set
            )
        if hasattr(m, "solar_curt"):
            curt_cost += sum(
                p.curtailment_cost * pyo.value(m.solar_curt[sf, t]) * p.dt
                for sf in range(len(self._solar_buses))
                for t in T_set
            )

        total = fuel_cost + startup_cost + shutdown_cost + curt_cost
        result = {
            "fuel_cost": fuel_cost,
            "startup_cost": startup_cost,
            "shutdown_cost": shutdown_cost,
            "curtailment_cost": curt_cost,
            "total_cost": total,
        }

        if p.enable_loss and hasattr(m, "loss"):
            total_loss = sum(pyo.value(m.loss[k, t]) for k in range(p.num_lines) for t in T_set)
            result["total_loss_mwh"] = total_loss * p.dt

        return result

    def line_utilization(self, m=None):
        if m is None:
            m = self.model
        p = self.p
        T_set = range(p.num_steps)
        K_set = range(p.num_lines)
        result = {}
        for k in K_set:
            flows = [abs(pyo.value(m.f[k, t])) for t in T_set]
            cap = p.line_capacity[k]
            result[k] = {
                "from": p.line_from[k],
                "to": p.line_to[k],
                "capacity": cap,
                "max_flow": max(flows),
                "avg_flow": sum(flows) / len(flows),
                "utilization": max(flows) / cap if cap > 0 else 0,
            }
        return result

    def write_lp(self, filepath="_uc.lp"):
        if self.model is None:
            raise RuntimeError("No model to write.")
        self.model.write(filepath, io_options={"symbolic_solver_labels": True})

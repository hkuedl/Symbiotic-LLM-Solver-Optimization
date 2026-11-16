"""Reference optimization method."""

import logging
import time

import numpy as np
import pyomo.environ as pyo

from common.solver_config import DEFAULT_TIME_LIMIT, create_solver, solve_model, validate_time_limit


class RollingHorizonSolver:
    """RollingHorizonSolver: reference optimization method."""

    def __init__(self, horizon_length=24, seed=42, time_limit=DEFAULT_TIME_LIMIT):
        self.time_limit = validate_time_limit(time_limit)
        self.horizon_length = horizon_length
        self.seed = seed
        self.solve_time = 0
        self.history = []

    def solve(self, model_obj, demand_bus, demand_total, reserve, wind=None, solar=None):
        """Run the method and return the extracted solution, or None on failure."""
        p = model_obj.p
        if model_obj.model is None:
            raise RuntimeError("Model not built yet.")

        total_start = time.time()
        self.history = []

        G, T = p.num_generators, p.num_steps
        n_windows = (T + self.horizon_length - 1) // self.horizon_length

        print(f"[RH] Rolling Horizon (H={self.horizon_length}, windows={n_windows}, T={T})")

        solver = create_solver(self.time_limit)

        committed_u = np.zeros((G, T), dtype=int)
        committed_P = np.zeros((G, T))

        prev_u = np.array(p.u0, dtype=int)
        prev_P = np.array(p.P0)
        prev_status_steps = np.array(p.initial_status_steps, dtype=int)
        prev_soc = p.SoC0 if p.has_storage else None

        t_current = 0
        window_idx = 0

        while t_current < T:
            t_end = min(t_current + self.horizon_length, T)
            H = t_end - t_current
            is_last = t_end == T

            demand_w = {}
            for n in range(p.num_buses):
                demand_w[n] = demand_bus[n][t_current:t_end]
            reserve_w = reserve[t_current:t_end]

            wind_w = None
            if wind is not None:
                wind_w = {b: v[t_current:t_end] for b, v in wind.items()}
            solar_w = None
            if solar is not None:
                solar_w = {b: v[t_current:t_end] for b, v in solar.items()}

            sub_model = self._build_subproblem(
                p,
                H,
                prev_u,
                prev_P,
                prev_status_steps,
                prev_soc,
                demand_w,
                reserve_w,
                wind_w,
                solar_w,
                is_last,
            )

            t0 = time.time()
            logging.getLogger("pyomo.core").setLevel(logging.ERROR)
            results = solve_model(solver, sub_model, tee=False)
            logging.getLogger("pyomo.core").setLevel(logging.WARNING)
            sub_time = time.time() - t0

            status = str(results.solver.termination_condition).lower()

            if status in ("optimal", "feasible", "locallyoptimal"):
                for g in range(G):
                    for t in range(H):
                        committed_u[g, t_current + t] = int(round(pyo.value(sub_model.u[g, t])))
                        committed_P[g, t_current + t] = pyo.value(sub_model.P[g, t])

                prev_u = np.array([committed_u[g, t_end - 1] for g in range(G)])
                prev_P = np.array([committed_P[g, t_end - 1] for g in range(G)])
                prev_status_steps = self._compute_status_steps(committed_u, G, t_end)

                if p.has_storage:
                    prev_soc = pyo.value(sub_model.SoC[H - 1])

                window_obj = pyo.value(sub_model.Obj)
                print(
                    f"  Window {window_idx}: t=[{t_current},{t_end}), "
                    f"${window_obj:,.0f}, {status}, {sub_time:.1f}s"
                )
            else:
                print(f"  Window {window_idx}: t=[{t_current},{t_end}), {status}, {sub_time:.1f}s")
                self.solve_time = time.time() - total_start
                return None

            self.history.append(
                {
                    "window": window_idx,
                    "t_start": t_current,
                    "t_end": t_end,
                    "status": status,
                    "solve_time": sub_time,
                }
            )

            t_current = t_end
            window_idx += 1

        print("[RH] Final evaluation on full model")
        final_obj, final_sol = self._evaluate_u(committed_u, model_obj)

        if final_sol is None:
            print("[RH] Final evaluation infeasible.")
            self.solve_time = time.time() - total_start
            return None

        self.solve_time = time.time() - total_start
        print(
            f"[RH] Done. obj=${final_obj:,.0f}, windows={window_idx}, time={self.solve_time:.1f}s"
        )
        return final_sol

    def _compute_status_steps(self, committed_u, G, t_end):
        status_steps = np.zeros(G, dtype=int)
        for g in range(G):
            count = 0
            current_val = committed_u[g, t_end - 1]
            for t in range(t_end - 1, -1, -1):
                if committed_u[g, t] == current_val:
                    count += 1
                else:
                    break
            status_steps[g] = count if current_val == 1 else -count
        return status_steps

    def _build_subproblem(
        self,
        p,
        H,
        prev_u,
        prev_P,
        prev_status_steps,
        prev_soc,
        demand_w,
        reserve_w,
        wind_w,
        solar_w,
        is_last,
    ):
        G = p.num_generators
        N = p.num_buses
        K = p.num_lines
        dt = p.dt

        G_set = range(G)
        T_set = range(H)
        N_set = range(N)
        K_set = range(K)

        demand_total = [sum(demand_w[n][t] for n in N_set) for t in T_set]

        has_wind = wind_w is not None and len(wind_w) > 0
        has_solar = solar_w is not None and len(solar_w) > 0
        wind_buses = sorted(wind_w.keys()) if has_wind else []
        solar_buses = sorted(solar_w.keys()) if has_solar else []

        m = pyo.ConcreteModel()

        m.G = pyo.RangeSet(0, G - 1)
        m.T = pyo.RangeSet(0, H - 1)
        m.N = pyo.RangeSet(0, N - 1)
        m.K = pyo.RangeSet(0, K - 1)

        m.u = pyo.Var(m.G, m.T, within=pyo.Binary)
        m.v = pyo.Var(m.G, m.T, within=pyo.Binary)
        m.w = pyo.Var(m.G, m.T, within=pyo.Binary)
        m.P = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals)
        m.theta = pyo.Var(m.N, m.T, within=pyo.Reals)
        m.f = pyo.Var(m.K, m.T, within=pyo.Reals)

        if has_wind:
            n_wf = len(wind_buses)
            m.WF = pyo.RangeSet(0, n_wf - 1)
            m.P_wind = pyo.Var(m.WF, m.T, within=pyo.NonNegativeReals)
            m.wind_curt = pyo.Var(m.WF, m.T, within=pyo.NonNegativeReals)

        if has_solar:
            n_sf = len(solar_buses)
            m.SF = pyo.RangeSet(0, n_sf - 1)
            m.P_solar = pyo.Var(m.SF, m.T, within=pyo.NonNegativeReals)
            m.solar_curt = pyo.Var(m.SF, m.T, within=pyo.NonNegativeReals)

        if p.has_storage:
            m.P_ch = pyo.Var(m.T, within=pyo.NonNegativeReals)
            m.P_dis = pyo.Var(m.T, within=pyo.NonNegativeReals)
            m.SoC = pyo.Var(m.T, within=pyo.NonNegativeReals)
            m.u_ch = pyo.Var(m.T, within=pyo.Binary)
            m.u_dis = pyo.Var(m.T, within=pyo.Binary)

        use_pw = p.use_piecewise and p.pw_slopes is not None
        if use_pw:
            m.SEG = pyo.RangeSet(0, p.num_segments - 1)
            m.delta = pyo.Var(m.G, m.SEG, m.T, within=pyo.NonNegativeReals)

        m.StartupShutdown = pyo.Constraint(
            m.G,
            m.T,
            rule=lambda m, g, t: (
                m.u[g, t] - (prev_u[g] if t == 0 else m.u[g, t - 1]) == m.v[g, t] - m.w[g, t]
            ),
        )

        m.Exclusive = pyo.Constraint(m.G, m.T, rule=lambda m, g, t: m.v[g, t] + m.w[g, t] <= 1)

        def min_up_rule(m, g, t):
            UT = p.min_up_time[g]
            if UT <= 1 or t + UT > H:
                return pyo.Constraint.Skip
            return sum(m.u[g, tau] for tau in range(t, t + UT)) >= UT * m.v[g, t]

        m.MinUp = pyo.Constraint(m.G, m.T, rule=min_up_rule)

        def min_down_rule(m, g, t):
            DT = p.min_down_time[g]
            if DT <= 1 or t + DT > H:
                return pyo.Constraint.Skip
            return sum(1 - m.u[g, tau] for tau in range(t, t + DT)) >= DT * m.w[g, t]

        m.MinDown = pyo.Constraint(m.G, m.T, rule=min_down_rule)

        for g in G_set:
            UT = p.min_up_time[g]
            DT = p.min_down_time[g]
            for t in T_set:
                if t + UT > H and t > 0:
                    m.add_component(f"TailUp_G{g}_T{t}", pyo.Constraint(expr=m.v[g, t] <= 0))
                if t + DT > H and t > 0:
                    m.add_component(f"TailDown_G{g}_T{t}", pyo.Constraint(expr=m.w[g, t] <= 0))

        m.PLower = pyo.Constraint(
            m.G, m.T, rule=lambda m, g, t: m.P[g, t] >= p.P_min[g] * m.u[g, t]
        )
        m.PUpper = pyo.Constraint(
            m.G, m.T, rule=lambda m, g, t: m.P[g, t] <= p.P_max[g] * m.u[g, t]
        )

        def ramp_up_rule(m, g, t):
            pP = prev_P[g] if t == 0 else m.P[g, t - 1]
            pu = prev_u[g] if t == 0 else m.u[g, t - 1]
            return m.P[g, t] - pP <= p.ramp_up[g] * pu + p.startup_ramp[g] * m.v[g, t]

        m.RampUp = pyo.Constraint(m.G, m.T, rule=ramp_up_rule)

        def ramp_down_rule(m, g, t):
            pP = prev_P[g] if t == 0 else m.P[g, t - 1]
            return pP - m.P[g, t] <= p.ramp_down[g] * m.u[g, t] + p.shutdown_ramp[g] * m.w[g, t]

        m.RampDown = pyo.Constraint(m.G, m.T, rule=ramp_down_rule)

        m.RefAngle = pyo.Constraint(m.T, rule=lambda m, t: m.theta[p.ref_bus, t] == 0)

        m.DCFlow = pyo.Constraint(
            m.K,
            m.T,
            rule=lambda m, k, t: (
                m.f[k, t]
                == (1.0 / p.line_reactance[k])
                * (m.theta[p.line_from[k], t] - m.theta[p.line_to[k], t])
            ),
        )

        m.LineCapPos = pyo.Constraint(
            m.K, m.T, rule=lambda m, k, t: m.f[k, t] <= p.line_capacity[k]
        )
        m.LineCapNeg = pyo.Constraint(
            m.K, m.T, rule=lambda m, k, t: -p.line_capacity[k] <= m.f[k, t]
        )

        gen_at_bus = {n: [] for n in N_set}
        for g in G_set:
            gen_at_bus[p.gen_bus[g]].append(g)
        wind_at_bus = {}
        for wf, bus in enumerate(wind_buses):
            wind_at_bus.setdefault(bus, []).append(wf)
        solar_at_bus = {}
        for sf, bus in enumerate(solar_buses):
            solar_at_bus.setdefault(bus, []).append(sf)

        if p.enable_loss and len(p.rx_ratio) == p.num_lines:
            m.f_pos = pyo.Var(m.K, m.T, within=pyo.NonNegativeReals)
            m.f_neg = pyo.Var(m.K, m.T, within=pyo.NonNegativeReals)
            m.loss = pyo.Var(m.K, m.T, within=pyo.NonNegativeReals)

            m.FlowSplit = pyo.Constraint(
                m.K, m.T, rule=lambda m, k, t: m.f[k, t] == m.f_pos[k, t] - m.f_neg[k, t]
            )
            m.FlowPosLimit = pyo.Constraint(
                m.K, m.T, rule=lambda m, k, t: m.f_pos[k, t] <= p.line_capacity[k]
            )
            m.FlowNegLimit = pyo.Constraint(
                m.K, m.T, rule=lambda m, k, t: m.f_neg[k, t] <= p.line_capacity[k]
            )
            m.LossCalc = pyo.Constraint(
                m.K,
                m.T,
                rule=lambda m, k, t: (
                    m.loss[k, t] == p.rx_ratio[k] * (m.f_pos[k, t] + m.f_neg[k, t])
                ),
            )

            lines_at_bus = {n: [] for n in N_set}
            for k in K_set:
                lines_at_bus[p.line_from[k]].append(k)
                lines_at_bus[p.line_to[k]].append(k)

            def nodal_balance_rule(m, n, t):
                gen_inj = sum(m.P[g, t] for g in gen_at_bus[n])
                re_inj = 0
                if has_wind and n in wind_at_bus:
                    re_inj += sum(m.P_wind[wf, t] for wf in wind_at_bus[n])
                if has_solar and n in solar_at_bus:
                    re_inj += sum(m.P_solar[sf, t] for sf in solar_at_bus[n])
                stor_inj = 0
                if p.has_storage and n == p.storage_bus:
                    stor_inj = m.P_dis[t] - m.P_ch[t]
                flow_out = sum(m.f[k, t] for k in K_set if p.line_from[k] == n)
                flow_in = sum(m.f[k, t] for k in K_set if p.line_to[k] == n)
                bus_loss = sum(0.5 * m.loss[k, t] for k in lines_at_bus[n])
                return gen_inj + re_inj + stor_inj - demand_w[n][t] == flow_out - flow_in + bus_loss

            m.NodalBalance = pyo.Constraint(m.N, m.T, rule=nodal_balance_rule)

        else:

            def nodal_balance_rule(m, n, t):
                gen_inj = sum(m.P[g, t] for g in gen_at_bus[n])
                re_inj = 0
                if has_wind and n in wind_at_bus:
                    re_inj += sum(m.P_wind[wf, t] for wf in wind_at_bus[n])
                if has_solar and n in solar_at_bus:
                    re_inj += sum(m.P_solar[sf, t] for sf in solar_at_bus[n])
                stor_inj = 0
                if p.has_storage and n == p.storage_bus:
                    stor_inj = m.P_dis[t] - m.P_ch[t]
                flow_out = sum(m.f[k, t] for k in K_set if p.line_from[k] == n)
                flow_in = sum(m.f[k, t] for k in K_set if p.line_to[k] == n)
                return gen_inj + re_inj + stor_inj - demand_w[n][t] == flow_out - flow_in

            m.NodalBalance = pyo.Constraint(m.N, m.T, rule=nodal_balance_rule)

        m.ReserveReq = pyo.Constraint(
            m.T,
            rule=lambda m, t: (
                sum(p.P_max[g] * m.u[g, t] for g in m.G) >= demand_total[t] + reserve_w[t]
            ),
        )

        if has_wind:
            m.WindLimit = pyo.Constraint(
                m.WF,
                m.T,
                rule=lambda m, wf, t: (
                    m.P_wind[wf, t] + m.wind_curt[wf, t] == wind_w[wind_buses[wf]][t]
                ),
            )
        if has_solar:
            m.SolarLimit = pyo.Constraint(
                m.SF,
                m.T,
                rule=lambda m, sf, t: (
                    m.P_solar[sf, t] + m.solar_curt[sf, t] == solar_w[solar_buses[sf]][t]
                ),
            )

        if p.has_storage:
            init_soc = prev_soc if prev_soc is not None else p.SoC0
            m.SoCDyn = pyo.Constraint(
                m.T,
                rule=lambda m, t: (
                    m.SoC[t]
                    == (init_soc if t == 0 else m.SoC[t - 1])
                    + p.eta_ch * m.P_ch[t] * dt / p.E_max
                    - m.P_dis[t] * dt / (p.eta_dis * p.E_max)
                ),
            )
            m.SoCLo = pyo.Constraint(m.T, rule=lambda m, t: m.SoC[t] >= p.SoC_min)
            m.SoCHi = pyo.Constraint(m.T, rule=lambda m, t: m.SoC[t] <= p.SoC_max)
            m.ChLim = pyo.Constraint(m.T, rule=lambda m, t: m.P_ch[t] <= m.u_ch[t] * p.P_ch_max)
            m.DisLim = pyo.Constraint(m.T, rule=lambda m, t: m.P_dis[t] <= m.u_dis[t] * p.P_dis_max)
            m.ChDisExcl = pyo.Constraint(m.T, rule=lambda m, t: m.u_ch[t] + m.u_dis[t] <= 1)
            if is_last:
                m.SoCTerminal = pyo.Constraint(rule=lambda m: m.SoC[H - 1] == p.SoC0)

        if use_pw:
            S_set = range(p.num_segments)
            m.SegWidth = pyo.Constraint(
                m.G,
                m.SEG,
                m.T,
                rule=lambda m, g, s, t: m.delta[g, s, t] <= p.pw_widths[g][s] * m.u[g, t],
            )
            m.SegSum = pyo.Constraint(
                m.G,
                m.T,
                rule=lambda m, g, t: (
                    m.P[g, t] == p.P_min[g] * m.u[g, t] + sum(m.delta[g, s, t] for s in S_set)
                ),
            )

        def obj_rule(m):
            if use_pw:
                fuel = sum(
                    (
                        p.cost_c[g] * m.u[g, t]
                        + sum(p.pw_slopes[g][s] * m.delta[g, s, t] for s in range(p.num_segments))
                    )
                    * dt
                    for g in m.G
                    for t in m.T
                )
            else:
                fuel = sum(
                    (
                        p.cost_a[g] * p.P_max[g] * m.P[g, t]
                        + p.cost_b[g] * m.P[g, t]
                        + p.cost_c[g] * m.u[g, t]
                    )
                    * dt
                    for g in m.G
                    for t in m.T
                )
            startup = sum(p.cost_startup[g] * m.v[g, t] for g in m.G for t in m.T)
            shutdown = sum(p.cost_shutdown[g] * m.w[g, t] for g in m.G for t in m.T)
            curt = 0
            if has_wind:
                curt += sum(
                    p.curtailment_cost * m.wind_curt[wf, t] * dt for wf in m.WF for t in m.T
                )
            if has_solar:
                curt += sum(
                    p.curtailment_cost * m.solar_curt[sf, t] * dt for sf in m.SF for t in m.T
                )
            return fuel + startup + shutdown + curt

        m.Obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        return m

    def _evaluate_u(self, u, model_obj):
        m = model_obj.model
        p = model_obj.p
        G, T = p.num_generators, p.num_steps

        for g in range(G):
            for t in range(T):
                val = int(u[g, t])
                m.u[g, t].fix(val)
                prev = p.u0[g] if t == 0 else int(u[g, t - 1])
                m.v[g, t].fix(max(0, val - prev))
                m.w[g, t].fix(max(0, prev - val))

        solver = create_solver(self.time_limit)

        logging.getLogger("pyomo.core").setLevel(logging.ERROR)
        results = solve_model(solver, m, tee=False, load_solutions=False)
        logging.getLogger("pyomo.core").setLevel(logging.WARNING)

        status = str(results.solver.termination_condition).lower()
        has_solution = len(results.solution) > 0
        if not has_solution and status in ("optimal", "feasible", "locallyoptimal"):
            try:
                m.solutions.load_from(results)
                has_solution = True
            except Exception:
                has_solution = False
        elif has_solution:
            m.solutions.load_from(results)

        for g in range(G):
            for t in range(T):
                m.u[g, t].unfix()
                m.v[g, t].unfix()
                m.w[g, t].unfix()

        if not has_solution:
            return float("inf"), None
        sol = model_obj.extract_results()
        return sol["objective"], sol

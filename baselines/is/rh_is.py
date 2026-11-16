"""Reference optimization method."""

import logging
import time

import numpy as np
import pyomo.environ as pyo

from common.solver_config import DEFAULT_TIME_LIMIT, create_solver, solve_model, validate_time_limit
from models import IndustrialModel


class RollingHorizonSolver:
    """RollingHorizonSolver: reference optimization method."""

    def __init__(self, horizon_length=24, seed=42, time_limit=DEFAULT_TIME_LIMIT):
        self.time_limit = validate_time_limit(time_limit)
        self.horizon_length = horizon_length
        self.seed = seed
        self.solve_time = 0.0
        self.history = []

    def solve(self, model_obj, price, pv_output, q_req):
        """Run the method and return the extracted solution, or None on failure."""
        p = model_obj.p
        if model_obj.model is None:
            raise RuntimeError("Model not built yet.")

        L = p.num_lines
        K = p.num_processes
        S = p.num_buffers
        T = p.num_steps
        dt = float(p.dt)

        price = np.asarray(price, dtype=float)
        pv_output = np.asarray(pv_output, dtype=float)

        if len(price) != T or len(pv_output) != T:
            raise ValueError("price and pv_output must have num_steps values.")

        total_start = time.time()
        self.history = []

        n_windows = (T + self.horizon_length - 1) // self.horizon_length
        print(
            f"[RH] Rolling Horizon (H={self.horizon_length}, windows={n_windows}, T={T})",
            flush=True,
        )

        committed_u = np.zeros((L, K, T), dtype=int)
        committed_prod_out = np.zeros((L, K, T), dtype=float)

        prev_u = np.asarray(p.u0, dtype=int).copy()
        prev_prod_out = np.zeros((L, K), dtype=float)
        prev_sos = np.asarray(p.SoS0, dtype=float).copy()
        prev_soc = float(p.SoC0)
        prev_on_age = np.zeros((L, K), dtype=int)

        for l in range(L):
            for k in range(K):
                if prev_u[l, k] == 1:
                    prev_on_age[l, k] = int(p.on_time[l][k])

        cumulative_production = np.zeros(L, dtype=float)

        t_current = 0
        window_idx = 0

        while t_current < T:
            t_end = min(t_current + self.horizon_length, T)
            H = t_end - t_current
            is_first = t_current == 0
            is_last = t_end == T
            remaining_steps = T - t_current

            price_w = price[t_current:t_end]
            pv_w = pv_output[t_current:t_end]

            remaining_total = max(
                0.0,
                float(q_req) - float(np.sum(cumulative_production)),
            )
            remaining_line = np.maximum(
                0.0,
                np.asarray(p.q_req_line_min, dtype=float) - cumulative_production,
            )

            window_total_req, window_line_req = self._window_requirements(
                p=p,
                t_current=t_current,
                t_end=t_end,
                remaining_total=remaining_total,
                remaining_line=remaining_line,
                remaining_steps=remaining_steps,
                is_last=is_last,
            )

            maint_w = []
            for maint_l, maint_start, maint_end in p.maintenance_windows:
                local_start = max(0, int(maint_start) - t_current)
                local_end = min(H - 1, int(maint_end) - t_current)
                if local_start <= local_end:
                    maint_w.append((maint_l, local_start, local_end))

            sub_model = self._build_subproblem(
                p=p,
                H=H,
                prev_u=prev_u,
                prev_prod_out=prev_prod_out,
                prev_on_age=prev_on_age,
                prev_sos=prev_sos,
                prev_soc=prev_soc,
                price_w=price_w,
                pv_w=pv_w,
                window_total_req=window_total_req,
                window_line_req=window_line_req,
                future_steps=T - t_end,
                maint_w=maint_w,
                is_first=is_first,
                is_last=is_last,
            )

            solver = self._make_solver()

            print(
                f"  Window {window_idx}: t=[{t_current},{t_end}), "
                f"required={window_total_req:.1f}, solving...",
                flush=True,
            )

            solve_start = time.time()
            result = self._solve_model(sub_model, solver, warmstart=False)
            sub_time = time.time() - solve_start

            if not result["has_solution"]:
                print(
                    f"  Window {window_idx}: t=[{t_current},{t_end}), "
                    f"{result['status']}, {sub_time:.1f}s",
                    flush=True,
                )
                self.history.append(
                    {
                        "window": window_idx,
                        "t_start": t_current,
                        "t_end": t_end,
                        "status": result["status"],
                        "solve_time": sub_time,
                        "objective": None,
                        "production": None,
                        "required_production": window_total_req,
                    }
                )
                self.solve_time = time.time() - total_start
                return None

            last_process = K - 1
            window_line_production = np.zeros(L, dtype=float)

            for l in range(L):
                for k in range(K):
                    for local_t in range(H):
                        global_t = t_current + local_t
                        committed_u[l, k, global_t] = int(
                            round(pyo.value(sub_model.u[l, k, local_t]))
                        )
                        committed_prod_out[l, k, global_t] = float(
                            pyo.value(sub_model.prod_out[l, k, local_t])
                        )

                window_line_production[l] = sum(
                    pyo.value(sub_model.prod_out[l, last_process, local_t]) * dt
                    for local_t in range(H)
                )

            cumulative_production += window_line_production

            prev_u = committed_u[:, :, t_end - 1].copy()
            prev_prod_out = committed_prod_out[:, :, t_end - 1].copy()
            prev_sos = np.array(
                [[pyo.value(sub_model.SoS[l, s, H - 1]) for s in range(S)] for l in range(L)],
                dtype=float,
            )
            prev_soc = float(pyo.value(sub_model.SoC[H - 1]))
            prev_on_age = self._compute_on_age(
                committed_u,
                prev_u,
                t_end,
            )

            window_obj = float(pyo.value(sub_model.Obj))
            window_prod = float(np.sum(window_line_production))

            print(
                f"  Window {window_idx}: t=[{t_current},{t_end}), "
                f"obj=${window_obj:,.0f}, prod={window_prod:.1f}, "
                f"cumulative={np.sum(cumulative_production):.1f}, "
                f"{result['status']}, {sub_time:.1f}s",
                flush=True,
            )

            self.history.append(
                {
                    "window": window_idx,
                    "t_start": t_current,
                    "t_end": t_end,
                    "status": result["status"],
                    "solve_time": sub_time,
                    "objective": window_obj,
                    "production": window_prod,
                    "required_production": window_total_req,
                }
            )

            t_current = t_end
            window_idx += 1

        print("[RH] Final evaluation on full model", flush=True)
        final_obj, final_sol = self._evaluate_fixed(
            committed_u,
            p,
            price,
            pv_output,
            q_req,
        )

        if final_sol is None:
            print("[RH] Final evaluation infeasible.", flush=True)
            self.solve_time = time.time() - total_start
            return None

        self.solve_time = time.time() - total_start
        print(
            f"[RH] Done. obj=${final_obj:,.0f}, "
            f"prod={final_sol['total_production']:.1f}, "
            f"windows={window_idx}, time={self.solve_time:.1f}s",
            flush=True,
        )
        return final_sol

    def _window_requirements(
        self, p, t_current, t_end, remaining_total, remaining_line, remaining_steps, is_last
    ):
        L = p.num_lines
        K = p.num_processes
        T = p.num_steps
        dt = float(p.dt)
        H = t_end - t_current
        last_process = K - 1

        if is_last:
            return float(remaining_total), np.asarray(
                remaining_line,
                dtype=float,
            )

        current_line_cap = np.zeros(L, dtype=float)
        future_line_cap = np.zeros(L, dtype=float)

        maintenance = {
            (int(line), int(t))
            for line, start, end in p.maintenance_windows
            for t in range(int(start), min(int(end) + 1, T))
        }

        for l in range(L):
            rate = float(p.production_rate_max[l][last_process]) * dt
            current_line_cap[l] = sum(
                rate for t in range(t_current, t_end) if (l, t) not in maintenance
            )
            future_line_cap[l] = sum(rate for t in range(t_end, T) if (l, t) not in maintenance)

        proportional_line = remaining_line * H / max(remaining_steps, 1)
        mandatory_line = np.maximum(0.0, remaining_line - future_line_cap)
        window_line_req = np.maximum(proportional_line, mandatory_line)
        window_line_req = np.minimum(window_line_req, current_line_cap)

        proportional_total = remaining_total * H / max(remaining_steps, 1)
        mandatory_total = max(
            0.0,
            remaining_total - float(np.sum(future_line_cap)),
        )
        window_total_req = max(
            proportional_total,
            mandatory_total,
            float(np.sum(window_line_req)),
        )
        window_total_req = min(
            window_total_req,
            float(np.sum(current_line_cap)),
            float(remaining_total),
        )

        return float(window_total_req), window_line_req

    def _compute_on_age(self, committed_u, prev_u, t_end):
        L, K, _ = committed_u.shape
        ages = np.zeros((L, K), dtype=int)

        for l in range(L):
            for k in range(K):
                if prev_u[l, k] == 0:
                    continue
                age = 0
                for t in range(t_end - 1, -1, -1):
                    if committed_u[l, k, t] == 1:
                        age += 1
                    else:
                        break
                ages[l, k] = age

        return ages

    def _build_subproblem(
        self,
        p,
        H,
        prev_u,
        prev_prod_out,
        prev_on_age,
        prev_sos,
        prev_soc,
        price_w,
        pv_w,
        window_total_req,
        window_line_req,
        future_steps,
        maint_w,
        is_first,
        is_last,
    ):
        L = p.num_lines
        K = p.num_processes
        S = p.num_buffers
        dt = float(p.dt)
        transfer_time = int(p.transfer_time)

        L_set = range(L)
        K_set = range(K)

        m = pyo.ConcreteModel()
        m.L = pyo.RangeSet(0, L - 1)
        m.K = pyo.RangeSet(0, K - 1)
        m.S = pyo.RangeSet(0, S - 1)
        m.T = pyo.RangeSet(0, H - 1)

        m.u = pyo.Var(m.L, m.K, m.T, within=pyo.Binary)
        m.mu = pyo.Var(m.L, m.K, m.T, within=pyo.Binary)
        m.nu = pyo.Var(m.L, m.K, m.T, within=pyo.Binary)
        m.prod_in = pyo.Var(m.L, m.K, m.T, within=pyo.NonNegativeReals)
        m.prod_out = pyo.Var(m.L, m.K, m.T, within=pyo.NonNegativeReals)
        m.P_sub = pyo.Var(m.L, m.K, m.T, within=pyo.NonNegativeReals)
        m.SoS = pyo.Var(m.L, m.S, m.T, within=pyo.NonNegativeReals)
        m.transfer = pyo.Var(m.L, m.L, m.S, m.T, within=pyo.NonNegativeReals)
        m.Y = pyo.Var(m.L, within=pyo.NonNegativeReals)

        m.P_grid = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.P_pv = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.P_ch = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.P_dis = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.SoC = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.u_ch = pyo.Var(m.T, within=pyo.Binary)
        m.u_dis = pyo.Var(m.T, within=pyo.Binary)
        m.P_prod_total = pyo.Var(m.T, within=pyo.NonNegativeReals)

        m.MaterialConversion = pyo.Constraint(
            m.L,
            m.K,
            m.T,
            rule=lambda m, l, k, t: m.prod_out[l, k, t] == p.eta[l][k] * m.prod_in[l, k, t],
        )

        m.ProdLower = pyo.Constraint(
            m.L,
            m.K,
            m.T,
            rule=lambda m, l, k, t: (
                m.prod_out[l, k, t] >= p.production_rate_min[l][k] * m.u[l, k, t]
            ),
        )

        m.ProdUpper = pyo.Constraint(
            m.L,
            m.K,
            m.T,
            rule=lambda m, l, k, t: (
                m.prod_out[l, k, t] <= p.production_rate_max[l][k] * m.u[l, k, t]
            ),
        )

        m.PowerConsumption = pyo.Constraint(
            m.L,
            m.K,
            m.T,
            rule=lambda m, l, k, t: m.P_sub[l, k, t] == p.power_per_ton[l][k] * m.prod_out[l, k, t],
        )

        def startup_shutdown_rule(m, l, k, t):
            previous = prev_u[l, k] if t == 0 else m.u[l, k, t - 1]
            return m.u[l, k, t] - previous == m.mu[l, k, t] - m.nu[l, k, t]

        m.StartupShutdown = pyo.Constraint(
            m.L,
            m.K,
            m.T,
            rule=startup_shutdown_rule,
        )

        m.Exclusive = pyo.Constraint(
            m.L,
            m.K,
            m.T,
            rule=lambda m, l, k, t: m.mu[l, k, t] + m.nu[l, k, t] <= 1,
        )

        def min_on_time_rule(m, l, k, t):
            on_time = int(p.on_time[l][k])
            if on_time <= 1 or t + on_time > H:
                return pyo.Constraint.Skip
            return sum(m.u[l, k, tau] for tau in range(t, t + on_time)) >= on_time * m.mu[l, k, t]

        m.MinOnTime = pyo.Constraint(
            m.L,
            m.K,
            m.T,
            rule=min_on_time_rule,
        )

        for l in L_set:
            for k in K_set:
                if prev_u[l, k] == 1:
                    remaining_on = max(
                        0,
                        int(p.on_time[l][k]) - int(prev_on_age[l, k]),
                    )
                    for t in range(min(remaining_on, H)):
                        m.add_component(
                            f"InitialOn_L{l}_K{k}_T{t}",
                            pyo.Constraint(expr=m.u[l, k, t] == 1),
                        )

                if not is_last:
                    on_time = int(p.on_time[l][k])
                    if on_time > 1:
                        for t in range(max(0, H - on_time + 1), H):
                            m.add_component(
                                f"TailStart_L{l}_K{k}_T{t}",
                                pyo.Constraint(expr=m.mu[l, k, t] == 0),
                            )

        def storage_dynamics_rule(m, l, s, t):
            previous = prev_sos[l, s] if t == 0 else m.SoS[l, s, t - 1]
            inflow = m.prod_out[l, s, t] * dt
            outflow = m.prod_in[l, s + 1, t] * dt
            transfer_out = sum(m.transfer[l, j, s, t] * dt for j in L_set if j != l)
            arrival_t = t - transfer_time
            if arrival_t >= 0:
                transfer_in = sum(m.transfer[i, l, s, arrival_t] * dt for i in L_set if i != l)
            else:
                transfer_in = 0.0
            return m.SoS[l, s, t] == previous + inflow - outflow - transfer_out + transfer_in

        m.StorageDynamics = pyo.Constraint(
            m.L,
            m.S,
            m.T,
            rule=storage_dynamics_rule,
        )

        m.StorageLower = pyo.Constraint(
            m.L,
            m.S,
            m.T,
            rule=lambda m, l, s, t: m.SoS[l, s, t] >= p.SoS_min[l][s],
        )

        m.StorageUpper = pyo.Constraint(
            m.L,
            m.S,
            m.T,
            rule=lambda m, l, s, t: m.SoS[l, s, t] <= p.SoS_max[l][s],
        )

        def transfer_limit_rule(m, i, j, s, t):
            if i == j:
                return m.transfer[i, j, s, t] == 0
            if transfer_time > 0 and t >= H - transfer_time:
                return m.transfer[i, j, s, t] == 0
            return m.transfer[i, j, s, t] <= p.transfer_rate_max

        m.TransferLimit = pyo.Constraint(
            m.L,
            m.L,
            m.S,
            m.T,
            rule=transfer_limit_rule,
        )

        def transfer_availability_rule(m, i, j, s, t):
            if i == j:
                return pyo.Constraint.Skip
            previous = prev_sos[i, s] if t == 0 else m.SoS[i, s, t - 1]
            total_out = sum(m.transfer[i, j_prime, s, t] for j_prime in L_set if j_prime != i)
            return total_out * dt <= previous - p.SoS_min[i][s]

        m.TransferAvailability = pyo.Constraint(
            m.L,
            m.L,
            m.S,
            m.T,
            rule=transfer_availability_rule,
        )

        for maint_l, local_start, local_end in maint_w:
            for k in K_set:
                for t in range(local_start, local_end + 1):
                    m.add_component(
                        f"Maintenance_L{maint_l}_K{k}_T{t}",
                        pyo.Constraint(expr=m.u[maint_l, k, t] == 0),
                    )

        def ramp_up_rule(m, l, k, t):
            if t == 0:
                if is_first:
                    return pyo.Constraint.Skip
                return (
                    m.prod_out[l, k, t] - prev_prod_out[l, k]
                    <= p.ramp_rate[l][k] + p.production_rate_max[l][k] * m.mu[l, k, t]
                )
            return (
                m.prod_out[l, k, t] - m.prod_out[l, k, t - 1]
                <= p.ramp_rate[l][k] + p.production_rate_max[l][k] * m.mu[l, k, t]
            )

        m.RampUp = pyo.Constraint(
            m.L,
            m.K,
            m.T,
            rule=ramp_up_rule,
        )

        def ramp_down_rule(m, l, k, t):
            if t == 0:
                if is_first:
                    return pyo.Constraint.Skip
                return (
                    prev_prod_out[l, k] - m.prod_out[l, k, t]
                    <= p.ramp_rate[l][k] + p.production_rate_max[l][k] * m.nu[l, k, t]
                )
            return (
                m.prod_out[l, k, t - 1] - m.prod_out[l, k, t]
                <= p.ramp_rate[l][k] + p.production_rate_max[l][k] * m.nu[l, k, t]
            )

        m.RampDown = pyo.Constraint(
            m.L,
            m.K,
            m.T,
            rule=ramp_down_rule,
        )

        last_process = K - 1
        m.LineOutput = pyo.Constraint(
            m.L,
            rule=lambda m, l: m.Y[l] == sum(m.prod_out[l, last_process, t] * dt for t in m.T),
        )

        m.LineMinOutput = pyo.Constraint(
            m.L,
            rule=lambda m, l: m.Y[l] >= float(window_line_req[l]),
        )

        m.TotalProduction = pyo.Constraint(expr=sum(m.Y[l] for l in m.L) >= float(window_total_req))

        m.PVLimit = pyo.Constraint(
            m.T,
            rule=lambda m, t: m.P_pv[t] <= float(pv_w[t]),
        )

        def soc_rule(m, t):
            previous = prev_soc if t == 0 else m.SoC[t - 1]
            return m.SoC[t] == previous + p.eta_ch * m.P_ch[t] * dt / p.E_max - m.P_dis[t] * dt / (
                p.eta_dis * p.E_max
            )

        m.SoCDynamics = pyo.Constraint(m.T, rule=soc_rule)

        m.SoCLower = pyo.Constraint(
            m.T,
            rule=lambda m, t: m.SoC[t] >= p.SoC_min,
        )

        m.SoCUpper = pyo.Constraint(
            m.T,
            rule=lambda m, t: m.SoC[t] <= p.SoC_max,
        )

        m.ChargeLimit = pyo.Constraint(
            m.T,
            rule=lambda m, t: m.P_ch[t] <= m.u_ch[t] * p.P_ch_max,
        )

        m.DischargeLimit = pyo.Constraint(
            m.T,
            rule=lambda m, t: m.P_dis[t] <= m.u_dis[t] * p.P_dis_max,
        )

        m.ChargeDischargeExclusive = pyo.Constraint(
            m.T,
            rule=lambda m, t: m.u_ch[t] + m.u_dis[t] <= 1,
        )

        if is_last:
            m.SoCTerminal = pyo.Constraint(expr=m.SoC[H - 1] == p.SoC0)
        else:
            max_future_increase = p.eta_ch * p.P_ch_max * dt * future_steps / p.E_max
            max_future_decrease = p.P_dis_max * dt * future_steps / (p.eta_dis * p.E_max)
            terminal_lower = max(
                float(p.SoC_min),
                float(p.SoC0) - max_future_increase,
            )
            terminal_upper = min(
                float(p.SoC_max),
                float(p.SoC0) + max_future_decrease,
            )
            m.SoCTerminalLower = pyo.Constraint(expr=m.SoC[H - 1] >= terminal_lower)
            m.SoCTerminalUpper = pyo.Constraint(expr=m.SoC[H - 1] <= terminal_upper)

        m.TotalPower = pyo.Constraint(
            m.T,
            rule=lambda m, t: m.P_prod_total[t] == sum(m.P_sub[l, k, t] for l in m.L for k in m.K),
        )

        m.PowerBalance = pyo.Constraint(
            m.T,
            rule=lambda m, t: m.P_grid[t] + m.P_pv[t] + m.P_dis[t] == m.P_prod_total[t] + m.P_ch[t],
        )

        m.GridLimit = pyo.Constraint(
            m.T,
            rule=lambda m, t: m.P_grid[t] <= p.P_grid_max,
        )

        def objective_rule(m):
            power_cost = sum(float(price_w[t]) * m.P_grid[t] * dt for t in m.T)
            startup_cost = sum(
                p.cost_startup[l][k] * m.mu[l, k, t] for l in m.L for k in m.K for t in m.T
            )
            shutdown_cost = sum(
                p.cost_shutdown[l][k] * m.nu[l, k, t] for l in m.L for k in m.K for t in m.T
            )
            transfer_cost = sum(
                p.transfer_cost_per_ton * m.transfer[i, j, s, t] * dt
                for i in m.L
                for j in m.L
                for s in m.S
                for t in m.T
                if i != j
            )
            return power_cost + startup_cost + shutdown_cost + transfer_cost

        m.Obj = pyo.Objective(
            rule=objective_rule,
            sense=pyo.minimize,
        )

        return m

    def _evaluate_fixed(self, u, p, price, pv_output, q_req):
        L = p.num_lines
        K = p.num_processes
        T = p.num_steps

        eval_model = IndustrialModel(p)
        m = eval_model.build(price, pv_output, q_req)

        for l in range(L):
            for k in range(K):
                for t in range(T):
                    value = int(u[l, k, t])
                    previous = p.u0[l][k] if t == 0 else int(u[l, k, t - 1])
                    m.u[l, k, t].fix(value)
                    m.mu[l, k, t].fix(max(0, value - previous))
                    m.nu[l, k, t].fix(max(0, previous - value))
                    m.u[l, k, t].set_value(value)
                    m.mu[l, k, t].set_value(max(0, value - previous))
                    m.nu[l, k, t].set_value(max(0, previous - value))

        solver = self._make_solver()

        result = self._solve_model(m, solver, warmstart=True)
        if not result["has_solution"]:
            return float("inf"), None

        eval_model.model = m
        sol = eval_model.extract_results()
        return float(sol["objective"]), sol

    def _make_solver(self):
        solver = create_solver(self.time_limit)

        solver.options["Seed"] = int(self.seed)

        return solver

    @staticmethod
    def _solve_model(m, solver, warmstart=False):
        logger = logging.getLogger("pyomo.core")
        old_level = logger.level
        logger.setLevel(logging.ERROR)

        try:
            results = solve_model(
                solver,
                m,
                tee=False,
                warmstart=warmstart,
                load_solutions=False,
            )
        except Exception as exc:
            return {
                "status": f"exception:{type(exc).__name__}",
                "has_solution": False,
            }
        finally:
            logger.setLevel(old_level)

        status = str(results.solver.termination_condition).lower()
        has_solution = len(results.solution) > 0

        if has_solution:
            try:
                m.solutions.load_from(results)
            except Exception:
                has_solution = False

        return {
            "status": status,
            "has_solution": has_solution,
        }

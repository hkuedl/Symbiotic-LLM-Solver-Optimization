"""Reference optimization method."""

import logging
import time

import numpy as np
import pyomo.environ as pyo

from common.solver_config import DEFAULT_TIME_LIMIT, create_solver, solve_model, validate_time_limit
from models import IndustrialModel


class LagrangianRelaxationSolver:
    """LagrangianRelaxationSolver: reference optimization method."""

    def __init__(
        self,
        max_iter=50,
        step_init=0.5,
        step_decay=0.95,
        recovery_interval=3,
        convergence_gap=0.01,
        recovery_fix_ratios=(0.9, 0.7, 0.5),
        seed=42,
        time_limit=DEFAULT_TIME_LIMIT,
    ):
        self.time_limit = validate_time_limit(time_limit)
        self.max_iter = max_iter
        self.step_init = step_init
        self.step_decay = step_decay
        self.recovery_interval = recovery_interval
        self.convergence_gap = convergence_gap
        self.recovery_fix_ratios = tuple(recovery_fix_ratios)
        self.seed = seed

        self.solve_time = 0.0
        self.history = []

    def solve(self, model_obj, price, pv_output, q_req):
        """Run the method and return the extracted solution, or None on failure."""
        p = model_obj.p
        if model_obj.model is None:
            raise RuntimeError("Model not built yet.")

        T = p.num_steps
        L = p.num_lines
        dt = float(p.dt)
        price_arr = np.asarray(price, dtype=float)
        pv_arr = np.asarray(pv_output, dtype=float)

        if len(price_arr) != T or len(pv_arr) != T:
            raise ValueError("price and pv_output must both have num_steps entries.")

        total_start = time.time()
        self.history = []

        lambda_t = price_arr.copy()
        mu_prod = 0.0

        print("[LR] Building line subproblems and energy subproblem...", flush=True)
        build_start = time.time()
        line_models = [self._build_line_subproblem(p, l) for l in range(L)]
        energy_model = self._build_energy_model(p, price_arr, pv_arr)
        print(f"[LR] Subproblems built in {time.time() - build_start:.1f}s", flush=True)

        best_ub = float("inf")
        best_lb = -float("inf")
        best_sol = None
        best_u = None

        for it in range(self.max_iter):
            iter_start = time.time()

            line_results = []
            all_optimal = True
            failed = False

            for l in range(L):
                result = self._solve_line_subproblem(line_models[l], lambda_t, mu_prod, p, l)
                line_results.append(result)
                if not result["has_solution"]:
                    failed = True
                    break
                if not result["is_optimal"]:
                    all_optimal = False

            if failed:
                print(f"  Iter {it}: line subproblem failed.", flush=True)
                break

            energy_result = self._solve_energy_subproblem(energy_model, lambda_t, p)

            if not energy_result["has_solution"]:
                print(
                    f"  Iter {it}: energy subproblem failed ({energy_result['status']}).",
                    flush=True,
                )
                break
            if not energy_result["is_optimal"]:
                all_optimal = False

            sub_time = time.time() - iter_start

            u_lr = np.zeros((L, p.num_processes, T), dtype=int)
            p_prod_total = np.zeros(T)
            dual_value = energy_result["objective"]
            total_Y = 0.0

            for l in range(L):
                u_lr[l] = line_results[l]["u"]
                p_prod_total += line_results[l]["P_line"]
                dual_value += line_results[l]["objective"]
                total_Y += line_results[l]["Y"]

            dual_value += mu_prod * float(q_req)

            p_grid_lr = energy_result["P_grid"]
            p_pv_lr = energy_result["P_pv"]
            p_ch_lr = energy_result["P_ch"]
            p_dis_lr = energy_result["P_dis"]

            strict_lb_available = all_optimal
            if strict_lb_available:
                best_lb = max(best_lb, dual_value)

            do_recovery = it % self.recovery_interval == 0 or it == self.max_iter - 1

            primal_obj = float("inf")
            recovered_sol = None
            recovered_u = None
            recovery_time = 0.0
            used_fix_ratio = None

            if do_recovery:
                recovery_start = time.time()
                primal_obj, recovered_sol, recovered_u, used_fix_ratio = self._recover_primal(
                    u_lr,
                    p,
                    price,
                    pv_output,
                    q_req,
                )
                recovery_time = time.time() - recovery_start

                if recovered_sol is not None and primal_obj < best_ub - 1e-6:
                    best_ub = primal_obj
                    best_sol = recovered_sol
                    best_u = recovered_u.copy()

            # --- Gap ---
            if np.isfinite(best_ub) and np.isfinite(best_lb):
                gap = max(0.0, (best_ub - best_lb) / max(abs(best_ub), 1e-10))
            else:
                gap = float("inf")

            status = (
                "feasible"
                if recovered_sol is not None
                else ("skip" if not do_recovery else "recovery_failed")
            )

            self.history.append(
                {
                    "iter": it,
                    "dual_value": dual_value,
                    "strict_lb": best_lb if np.isfinite(best_lb) else None,
                    "ub": best_ub if np.isfinite(best_ub) else None,
                    "gap": gap,
                    "status": status,
                    "total_Y": total_Y,
                    "mu_prod": mu_prod,
                    "recovery_fix_ratio": used_fix_ratio,
                    "sub_time": sub_time,
                    "recovery_time": recovery_time,
                }
            )

            lb_text = f"${best_lb:,.0f}" if np.isfinite(best_lb) else "N/A"
            ub_text = f"${best_ub:,.0f}" if np.isfinite(best_ub) else "N/A"
            gap_text = f"{gap:.2%}" if np.isfinite(gap) else "N/A"

            recovery_text = ""
            if do_recovery:
                ratio_text = f", fixed={used_fix_ratio:.0%}" if used_fix_ratio is not None else ""
                recovery_text = f", recovery={recovery_time:.1f}s{ratio_text}"

            dual_tag = "strict" if strict_lb_available else "heuristic"
            print(
                f"  Iter {it}: dual=${dual_value:,.0f} ({dual_tag}), "
                f"LB={lb_text}, UB={ub_text}, gap={gap_text}, "
                f"Y={total_Y:,.0f}, mu={mu_prod:.2f}, "
                f"sub={sub_time:.1f}s{recovery_text}",
                flush=True,
            )

            if np.isfinite(gap) and strict_lb_available and gap <= self.convergence_gap:
                print(f"  Gap <= {self.convergence_gap:.2%}; converged.", flush=True)
                break

            sg_power = (p_prod_total + p_ch_lr - p_grid_lr - p_pv_lr - p_dis_lr) * dt

            sg_prod = float(q_req) - total_Y

            sg_norm2 = float(np.dot(sg_power, sg_power)) + sg_prod**2
            if sg_norm2 < 1e-12:
                print("  Subgradient is approximately zero.", flush=True)
                break

            alpha = self.step_init * (self.step_decay**it)

            if strict_lb_available and np.isfinite(best_ub) and best_ub > dual_value:
                step = alpha * (best_ub - dual_value) / sg_norm2
            else:
                step = alpha / np.sqrt(sg_norm2)

            lambda_t = lambda_t + step * sg_power
            mu_prod = max(0.0, mu_prod + step * sg_prod)

        self.solve_time = time.time() - total_start

        lb_text = f"${best_lb:,.0f}" if np.isfinite(best_lb) else "N/A"
        ub_text = f"${best_ub:,.0f}" if np.isfinite(best_ub) else "N/A"
        if np.isfinite(best_ub) and np.isfinite(best_lb):
            final_gap = max(
                0.0,
                (best_ub - best_lb) / max(abs(best_ub), 1e-10),
            )
            gap_text = f"{final_gap:.2%}"
        else:
            gap_text = "N/A"

        print(
            f"[LR] Done. UB={ub_text}, LB={lb_text}, gap={gap_text}, time={self.solve_time:.1f}s",
            flush=True,
        )

        return best_sol

    # ================================================================

    # ================================================================

    def _build_line_subproblem(self, p, line_id):
        """Build the relaxed scheduling subproblem for one production line."""
        K = p.num_processes
        S = p.num_buffers
        T = p.num_steps
        dt = float(p.dt)
        l = line_id

        m = pyo.ConcreteModel(name=f"ISS_Line_{l}_Subproblem")
        m.K = pyo.RangeSet(0, K - 1)
        m.S = pyo.RangeSet(0, S - 1)
        m.T = pyo.RangeSet(0, T - 1)

        m.lam = pyo.Param(m.T, initialize={t: 0.0 for t in range(T)}, mutable=True)
        m.mu_prod = pyo.Param(initialize=0.0, mutable=True)

        m.u = pyo.Var(m.K, m.T, within=pyo.Binary)
        m.mu = pyo.Var(m.K, m.T, within=pyo.Binary)
        m.nu = pyo.Var(m.K, m.T, within=pyo.Binary)
        m.prod_in = pyo.Var(m.K, m.T, within=pyo.NonNegativeReals)
        m.prod_out = pyo.Var(m.K, m.T, within=pyo.NonNegativeReals)
        m.P_sub = pyo.Var(m.K, m.T, within=pyo.NonNegativeReals)
        m.SoS = pyo.Var(m.S, m.T, within=pyo.NonNegativeReals)
        m.Y = pyo.Var(within=pyo.NonNegativeReals)

        m.MaterialConversion = pyo.Constraint(
            m.K,
            m.T,
            rule=lambda m, k, t: m.prod_out[k, t] == p.eta[l][k] * m.prod_in[k, t],
        )

        m.ProdLower = pyo.Constraint(
            m.K,
            m.T,
            rule=lambda m, k, t: m.prod_out[k, t] >= p.production_rate_min[l][k] * m.u[k, t],
        )

        m.ProdUpper = pyo.Constraint(
            m.K,
            m.T,
            rule=lambda m, k, t: m.prod_out[k, t] <= p.production_rate_max[l][k] * m.u[k, t],
        )

        m.PowerConsumption = pyo.Constraint(
            m.K,
            m.T,
            rule=lambda m, k, t: m.P_sub[k, t] == p.power_per_ton[l][k] * m.prod_out[k, t],
        )

        def startup_shutdown_rule(m, k, t):
            prev = p.u0[l][k] if t == 0 else m.u[k, t - 1]
            return m.u[k, t] - prev == m.mu[k, t] - m.nu[k, t]

        m.StartupShutdown = pyo.Constraint(m.K, m.T, rule=startup_shutdown_rule)

        m.Exclusive = pyo.Constraint(
            m.K,
            m.T,
            rule=lambda m, k, t: m.mu[k, t] + m.nu[k, t] <= 1,
        )

        def min_on_time_rule(m, k, t):
            on_time = int(p.on_time[l][k])
            if on_time <= 1 or t + on_time > T:
                return pyo.Constraint.Skip
            return sum(m.u[k, tau] for tau in range(t, t + on_time)) >= on_time * m.mu[k, t]

        m.MinOnTime = pyo.Constraint(m.K, m.T, rule=min_on_time_rule)

        def storage_dynamics_rule(m, s, t):
            prev = p.SoS0[l][s] if t == 0 else m.SoS[s, t - 1]
            inflow = m.prod_out[s, t] * dt
            outflow = m.prod_in[s + 1, t] * dt
            return m.SoS[s, t] == prev + inflow - outflow

        m.StorageDynamics = pyo.Constraint(m.S, m.T, rule=storage_dynamics_rule)

        m.StorageLower = pyo.Constraint(
            m.S,
            m.T,
            rule=lambda m, s, t: m.SoS[s, t] >= p.SoS_min[l][s],
        )
        m.StorageUpper = pyo.Constraint(
            m.S,
            m.T,
            rule=lambda m, s, t: m.SoS[s, t] <= p.SoS_max[l][s],
        )

        for maint_line, t_start, t_end in p.maintenance_windows:
            if maint_line == l:
                for k in range(K):
                    for t in range(t_start, min(t_end + 1, T)):
                        m.add_component(
                            f"Maintenance_K{k}_T{t}",
                            pyo.Constraint(expr=m.u[k, t] == 0),
                        )

        def ramp_up_rule(m, k, t):
            if t == 0:
                return pyo.Constraint.Skip
            return (
                m.prod_out[k, t] - m.prod_out[k, t - 1]
                <= p.ramp_rate[l][k] + p.production_rate_max[l][k] * m.mu[k, t]
            )

        m.RampUp = pyo.Constraint(m.K, m.T, rule=ramp_up_rule)

        def ramp_down_rule(m, k, t):
            if t == 0:
                return pyo.Constraint.Skip
            return (
                m.prod_out[k, t - 1] - m.prod_out[k, t]
                <= p.ramp_rate[l][k] + p.production_rate_max[l][k] * m.nu[k, t]
            )

        m.RampDown = pyo.Constraint(m.K, m.T, rule=ramp_down_rule)

        last_process = K - 1
        m.LineOutput = pyo.Constraint(
            rule=lambda m: m.Y == sum(m.prod_out[last_process, t] * dt for t in m.T),
        )

        m.LineMinOutput = pyo.Constraint(
            rule=lambda m: m.Y >= p.q_req_line_min[l],
        )

        def objective_rule(m):
            startup_cost = sum(p.cost_startup[l][k] * m.mu[k, t] for k in m.K for t in m.T)
            shutdown_cost = sum(p.cost_shutdown[l][k] * m.nu[k, t] for k in m.K for t in m.T)
            lr_power = sum(m.lam[t] * m.P_sub[k, t] * dt for k in m.K for t in m.T)
            lr_prod = -m.mu_prod * m.Y
            return startup_cost + shutdown_cost + lr_power + lr_prod

        m.Obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        return m

    def _build_energy_model(self, p, price_arr, pv_arr):
        T = p.num_steps
        dt = float(p.dt)

        m = pyo.ConcreteModel(name="ISS_Energy_Subproblem")
        m.T = pyo.RangeSet(0, T - 1)

        m.lam = pyo.Param(m.T, initialize={t: 0.0 for t in range(T)}, mutable=True)
        m.price = pyo.Param(m.T, initialize={t: float(price_arr[t]) for t in range(T)})
        m.pv_max = pyo.Param(m.T, initialize={t: float(pv_arr[t]) for t in range(T)})

        m.P_grid = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.P_pv = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.P_ch = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.P_dis = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.SoC = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.u_ch = pyo.Var(m.T, within=pyo.Binary)
        m.u_dis = pyo.Var(m.T, within=pyo.Binary)

        m.GridLimit = pyo.Constraint(m.T, rule=lambda m, t: m.P_grid[t] <= p.P_grid_max)
        m.PVLimit = pyo.Constraint(m.T, rule=lambda m, t: m.P_pv[t] <= m.pv_max[t])
        m.ChargeLimit = pyo.Constraint(m.T, rule=lambda m, t: m.P_ch[t] <= m.u_ch[t] * p.P_ch_max)
        m.DischargeLimit = pyo.Constraint(
            m.T, rule=lambda m, t: m.P_dis[t] <= m.u_dis[t] * p.P_dis_max
        )
        m.ChargeDischargeExclusive = pyo.Constraint(
            m.T, rule=lambda m, t: m.u_ch[t] + m.u_dis[t] <= 1
        )

        def soc_rule(m, t):
            prev = p.SoC0 if t == 0 else m.SoC[t - 1]
            return m.SoC[t] == prev + p.eta_ch * m.P_ch[t] * dt / p.E_max - m.P_dis[t] * dt / (
                p.eta_dis * p.E_max
            )

        m.SoCDynamics = pyo.Constraint(m.T, rule=soc_rule)

        m.SoCLower = pyo.Constraint(m.T, rule=lambda m, t: m.SoC[t] >= p.SoC_min)
        m.SoCUpper = pyo.Constraint(m.T, rule=lambda m, t: m.SoC[t] <= p.SoC_max)
        m.SoCTerminal = pyo.Constraint(expr=m.SoC[T - 1] == p.SoC0)

        def objective_rule(m):
            grid_cost = sum(m.price[t] * m.P_grid[t] * dt for t in m.T)
            lr_supply = sum(
                -m.lam[t] * (m.P_grid[t] + m.P_pv[t] + m.P_dis[t] - m.P_ch[t]) * dt for t in m.T
            )
            return grid_cost + lr_supply

        m.Obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        return m

    # ================================================================

    # ================================================================

    def _solve_line_subproblem(self, m, lambda_t, mu_prod, p, line_id):
        T = p.num_steps
        K = p.num_processes

        for t in range(T):
            m.lam[t] = float(lambda_t[t])
        m.mu_prod = float(mu_prod)

        solver = self._make_solver()

        result = self._solve_model(m, solver, warmstart=True)
        if not result["has_solution"]:
            return {
                **result,
                "objective": float("inf"),
                "u": None,
                "P_line": None,
                "Y": 0.0,
            }

        u = np.zeros((K, T), dtype=int)
        p_line = np.zeros(T)

        for k in range(K):
            for t in range(T):
                u[k, t] = int(round(pyo.value(m.u[k, t])))
                p_line[t] += pyo.value(m.P_sub[k, t])

        return {
            **result,
            "objective": float(pyo.value(m.Obj)),
            "u": u,
            "P_line": p_line,
            "Y": float(pyo.value(m.Y)),
        }

    def _solve_energy_subproblem(self, m, lambda_t, p):
        T = p.num_steps

        for t in range(T):
            m.lam[t] = float(lambda_t[t])

        solver = self._make_solver()

        result = self._solve_model(m, solver, warmstart=True)
        if not result["has_solution"]:
            return {
                **result,
                "objective": float("inf"),
                "P_grid": None,
                "P_pv": None,
                "P_ch": None,
                "P_dis": None,
            }

        return {
            **result,
            "objective": float(pyo.value(m.Obj)),
            "P_grid": np.array([pyo.value(m.P_grid[t]) for t in range(T)], dtype=float),
            "P_pv": np.array([pyo.value(m.P_pv[t]) for t in range(T)], dtype=float),
            "P_ch": np.array([pyo.value(m.P_ch[t]) for t in range(T)], dtype=float),
            "P_dis": np.array([pyo.value(m.P_dis[t]) for t in range(T)], dtype=float),
        }

    # ================================================================

    # ================================================================

    def _recover_primal(self, u_lr, p, price, pv_output, q_req):
        scores = self._stability_scores(u_lr)
        flat_indices = [
            (l, k, t)
            for l in range(p.num_lines)
            for k in range(p.num_processes)
            for t in range(p.num_steps)
        ]
        flat_indices.sort(
            key=lambda key: scores[key[0], key[1], key[2]],
            reverse=True,
        )

        ratios = self.recovery_fix_ratios

        for fix_ratio in ratios:
            eval_model = IndustrialModel(p)
            m = eval_model.build(price, pv_output, q_req)

            for l in range(p.num_lines):
                for k in range(p.num_processes):
                    for t in range(p.num_steps):
                        m.u[l, k, t].set_value(int(u_lr[l, k, t]))

            n_fix = int(len(flat_indices) * fix_ratio)
            for l, k, t in flat_indices[:n_fix]:
                m.u[l, k, t].fix(int(u_lr[l, k, t]))

            self._set_transition_warm_start(m, u_lr, p)

            solver = self._make_solver()

            result = self._solve_model(m, solver, warmstart=True)
            if not result["has_solution"]:
                continue

            eval_model.model = m
            try:
                sol = eval_model.extract_results()
            except (ValueError, TypeError):
                continue

            recovered_u = np.zeros_like(u_lr, dtype=int)
            valid = True
            for l in range(p.num_lines):
                for k in range(p.num_processes):
                    for t in range(p.num_steps):
                        value = m.u[l, k, t].value
                        if value is None:
                            valid = False
                            break
                        recovered_u[l, k, t] = int(round(value))
                    if not valid:
                        break
                if not valid:
                    break

            if valid:
                return (
                    float(sol["objective"]),
                    sol,
                    recovered_u,
                    float(fix_ratio),
                )

        return float("inf"), None, None, None

    # ================================================================

    # ================================================================

    @staticmethod
    def _stability_scores(u):
        L, K, T = u.shape
        scores = np.zeros((L, K, T), dtype=float)

        for l in range(L):
            for k in range(K):
                start = 0
                while start < T:
                    value = u[l, k, start]
                    end = start + 1
                    while end < T and u[l, k, end] == value:
                        end += 1

                    block_length = end - start
                    for t in range(start, end):
                        distance_to_edge = min(t - start + 1, end - t)
                        scores[l, k, t] = block_length + distance_to_edge
                    start = end

        return scores

    @staticmethod
    def _set_transition_warm_start(m, u, p):
        for l in range(p.num_lines):
            for k in range(p.num_processes):
                for t in range(p.num_steps):
                    val = int(u[l, k, t])
                    prev = p.u0[l][k] if t == 0 else int(u[l, k, t - 1])
                    m.u[l, k, t].set_value(val)
                    m.mu[l, k, t].set_value(max(0, val - prev))
                    m.nu[l, k, t].set_value(max(0, prev - val))

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
            logger.setLevel(old_level)
            return {
                "status": f"exception:{type(exc).__name__}",
                "has_solution": False,
                "is_optimal": False,
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

        is_optimal = "optimal" in status and "locally" not in status

        return {
            "status": status,
            "has_solution": has_solution,
            "is_optimal": is_optimal,
        }

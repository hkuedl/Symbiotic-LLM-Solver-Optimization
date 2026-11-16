"""Reference optimization method."""

import logging
import time

import pyomo.environ as pyo

from baselines.uc.lr_uc import LagrangianRelaxationSolver
from common.solver_config import DEFAULT_TIME_LIMIT, create_solver, solve_model, validate_time_limit


class FNOSolver:
    """FNOSolver: reference optimization method."""

    def __init__(
        self,
        time_window=24,
        max_rounds=3,
        lr_warmup_iters=20,
        seed=42,
        time_limit=DEFAULT_TIME_LIMIT,
    ):
        self.time_limit = validate_time_limit(time_limit)
        self.time_window = time_window
        self.max_rounds = max_rounds
        self.lr_warmup_iters = lr_warmup_iters
        self.seed = seed
        self.solve_time = 0

    def solve(self, model_obj, demand_bus, demand_total, reserve, wind=None, solar=None):
        """Run the method and return the extracted solution, or None on failure."""
        m = model_obj.model
        p = model_obj.p
        if m is None:
            raise RuntimeError("Model not built yet.")

        total_start = time.time()
        solver = create_solver(self.time_limit)
        G, T = p.num_generators, p.num_steps

        print(f"[FNO] Phase 1: LR warmup ({self.lr_warmup_iters} iters)")
        lr_solver = LagrangianRelaxationSolver(seed=self.seed, time_limit=self.time_limit)
        lr_u, lr_obj = lr_solver.warmup(
            model_obj,
            demand_bus,
            reserve,
            wind,
            solar,
            warmup_iters=self.lr_warmup_iters,
            recovery_interval=3,
        )

        if lr_obj < float("inf"):
            best_sol = self._evaluate_u(lr_u, model_obj, solver)
        else:
            best_sol = None

        if best_sol is None:
            print("[FNO] Failed to obtain initial feasible solution.")
            self.solve_time = time.time() - total_start
            return None

        best_obj = best_sol["objective"]
        print(f"[FNO] Phase 1 done: obj=${best_obj:,.0f}")

        print(
            f"[FNO] Phase 2: windowed optimization "
            f"(window={self.time_window}, rounds={self.max_rounds})"
        )

        solver.options["MIPFocus"] = 0

        for round_idx in range(self.max_rounds):
            improved = False
            w_start = 0
            round_start_obj = best_obj
            round_t0 = time.time()

            while w_start < p.num_steps:
                w_end = min(w_start + self.time_window, p.num_steps)
                self._fix_outside(m, best_sol, p, w_start, w_end)
                self._warm_start(m, best_sol, p)

                t0 = time.time()
                sub_result = solve_model(solver, m, tee=False, warmstart=True)
                sub_time = time.time() - t0
                sub_tc = str(sub_result.solver.termination_condition).lower()

                if sub_tc in ("optimal", "feasible", "locallyoptimal"):
                    new_obj = pyo.value(m.Obj)
                    delta = best_obj - new_obj
                    if new_obj < best_obj - 1e-6:
                        best_obj = new_obj
                        best_sol = model_obj.extract_results()
                        improved = True
                        print(
                            f"    Window {w_start:>3d}-{w_end:>3d}: "
                            f"${new_obj:,.0f} (↓${delta:,.0f}), "
                            f"{sub_tc}, {sub_time:.1f}s"
                        )
                    else:
                        print(
                            f"    Window {w_start:>3d}-{w_end:>3d}: "
                            f"no improve, {sub_tc}, {sub_time:.1f}s"
                        )
                else:
                    print(f"    Window {w_start:>3d}-{w_end:>3d}: {sub_tc}, {sub_time:.1f}s")

                self._unfix_all(m, p)
                w_start += self.time_window

            round_time = time.time() - round_t0
            improvement = round_start_obj - best_obj
            print(
                f"  Round {round_idx + 1}: obj=${best_obj:,.0f}, "
                f"improve=${improvement:,.0f}, "
                f"time={round_time:.1f}s",
                end="",
            )
            if not improved:
                print(" (converged)")
                break
            if round_idx >= 1 and improvement < best_obj * 1e-4:
                print(" (small improvement, stop)")
                break
            print()

        self._warm_start(m, best_sol, p)
        self.solve_time = time.time() - total_start
        print(f"[FNO] Done. obj=${best_obj:,.0f}, time={self.solve_time:.1f}s")
        return best_sol

    def _evaluate_u(self, u, model_obj, solver):
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
            return None
        return model_obj.extract_results()

    def _fix_outside(self, m, sol, p, w_start, w_end):
        G, T = p.num_generators, p.num_steps
        for g in range(G):
            for t in range(T):
                if w_start <= t < w_end:
                    m.u[g, t].unfix()
                    m.v[g, t].unfix()
                    m.w[g, t].unfix()
                else:
                    m.u[g, t].fix(round(sol["u"][g][t]))
                    m.v[g, t].fix(round(sol["v"][g][t]))
                    m.w[g, t].fix(round(sol["w"][g][t]))

        if p.has_storage:
            for t in range(T):
                if w_start <= t < w_end:
                    m.u_ch[t].unfix()
                    m.u_dis[t].unfix()
                else:
                    m.u_ch[t].fix(round(sol["u_ch"][t]))
                    m.u_dis[t].fix(round(sol["u_dis"][t]))

    def _unfix_all(self, m, p):
        G, T = p.num_generators, p.num_steps
        for g in range(G):
            for t in range(T):
                m.u[g, t].unfix()
                m.v[g, t].unfix()
                m.w[g, t].unfix()

        if p.has_storage:
            for t in range(T):
                m.u_ch[t].unfix()
                m.u_dis[t].unfix()

    def _warm_start(self, m, sol, p):
        G, T = p.num_generators, p.num_steps
        for g in range(G):
            for t in range(T):
                m.u[g, t].set_value(round(sol["u"][g][t]))
                m.v[g, t].set_value(round(sol["v"][g][t]))
                m.w[g, t].set_value(round(sol["w"][g][t]))

"""Reference optimization method."""

import time

import pyomo.environ as pyo

from common.solver_config import (
    DEFAULT_TIME_LIMIT,
    MIP_GAP,
    create_solver,
    solve_model,
    validate_time_limit,
)


class FNOSolver:
    """FNOSolver: reference optimization method."""

    def __init__(self, time_window=24, max_rounds=3, time_limit=DEFAULT_TIME_LIMIT):
        self.time_limit = validate_time_limit(time_limit)
        self.time_window = time_window
        self.max_rounds = max_rounds
        self.solve_time = 0

    def solve(self, model_obj):
        """Run the method and return the extracted solution, or None on failure."""
        m = model_obj.model
        p = model_obj.p
        if m is None:
            raise RuntimeError("Model not built yet.")

        total_start = time.time()
        solver = create_solver(self.time_limit)

        print(f"\n[FNO Phase 1] Initial solve (gap={MIP_GAP * 100:.1f}%, limit={self.time_limit}s)")
        solver.options["MIPFocus"] = 1

        result = solve_model(solver, m, tee=False)
        tc = str(result.solver.termination_condition).lower()

        if tc not in ("optimal", "feasible", "locallyoptimal"):
            print(f"[FNO] Failed: {tc}")
            return None

        best_sol = model_obj.extract_results()
        best_obj = best_sol["objective"]
        print(f"[FNO Phase 1] Initial obj = ${best_obj:,.0f}")

        print(
            f"[FNO Phase 2] Windowed optimization "
            f"(window={self.time_window}, rounds={self.max_rounds})"
        )

        solver.options["MIPFocus"] = 0

        for round_idx in range(self.max_rounds):
            improved = False
            w_start = 0

            while w_start < p.num_steps:
                w_end = min(w_start + self.time_window, p.num_steps)

                self._fix_outside(m, best_sol, p, w_start, w_end)

                sub_result = solve_model(solver, m, tee=False, warmstart=False)
                sub_tc = str(sub_result.solver.termination_condition).lower()

                if sub_tc in ("optimal", "feasible", "locallyoptimal"):
                    new_obj = pyo.value(m.Obj)
                    if new_obj < best_obj - 1e-6:
                        best_obj = new_obj
                        best_sol = model_obj.extract_results()
                        improved = True

                self._unfix_all(m, p)
                w_start += self.time_window

            print(f"  Round {round_idx + 1}: obj=${best_obj:,.0f}", end="")
            if not improved:
                print(" (converged)")
                break
            else:
                print()

        self.solve_time = time.time() - total_start
        print(f"[FNO Done] Final obj=${best_obj:,.0f}, time={self.solve_time:.1f}s\n")
        return best_sol

    def _fix_outside(self, m, sol, p, w_start, w_end):
        for l in range(p.num_lines):
            for k in range(p.num_processes):
                for t in range(p.num_steps):
                    if w_start <= t < w_end:
                        m.u[l, k, t].unfix()
                        m.mu[l, k, t].unfix()
                        m.nu[l, k, t].unfix()
                    else:
                        m.u[l, k, t].fix(round(sol["u"][l][k][t]))
                        m.mu[l, k, t].fix(round(sol["mu"][l][k][t]))
                        m.nu[l, k, t].fix(round(sol["nu"][l][k][t]))

        for t in range(p.num_steps):
            if w_start <= t < w_end:
                m.u_ch[t].unfix()
                m.u_dis[t].unfix()
            else:
                m.u_ch[t].fix(round(sol["u_ch"][t]))
                m.u_dis[t].fix(round(sol["u_dis"][t]))

    def _unfix_all(self, m, p):
        for l in range(p.num_lines):
            for k in range(p.num_processes):
                for t in range(p.num_steps):
                    m.u[l, k, t].unfix()
                    m.mu[l, k, t].unfix()
                    m.nu[l, k, t].unfix()
        for t in range(p.num_steps):
            m.u_ch[t].unfix()
            m.u_dis[t].unfix()

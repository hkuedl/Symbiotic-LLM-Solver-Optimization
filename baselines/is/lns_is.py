"""Reference optimization method."""

import random
import time

import numpy as np
import pyomo.environ as pyo

from common.solver_config import (
    DEFAULT_TIME_LIMIT,
    MIP_GAP,
    create_solver,
    solve_model,
    validate_time_limit,
)
from models import IndustrialModel


class LocalSearchSolver:
    """LocalSearchSolver: reference optimization method."""

    def __init__(self, max_iter=30, destroy_ratio=0.25, seed=42, time_limit=DEFAULT_TIME_LIMIT):
        self.time_limit = validate_time_limit(time_limit)
        self.max_iter = max_iter
        self.destroy_ratio = destroy_ratio
        self.seed = seed
        self.solve_time = 0
        self.history = []

    def solve(self, model_obj, price=None, pv_output=None, q_req=None):
        """Run the method and return the extracted solution, or None on failure."""
        p = model_obj.p
        if model_obj.model is None:
            raise RuntimeError("Model not built yet.")

        total_start = time.time()
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.history = []

        L, K, T = p.num_lines, p.num_processes, p.num_steps
        total_vars = L * K * T

        print(f"[LNS] Phase 1: Initial solution (gap={MIP_GAP * 100:.0f}%)")
        best_u, best_obj, best_sol = self._get_initial_solution(p, price, pv_output, q_req)

        if best_sol is None:
            self.solve_time = time.time() - total_start
            return None

        self.history.append({"iter": 0, "obj": best_obj, "status": "initial"})
        print(f"[LNS] Initial obj=${best_obj:,.0f}")

        n_destroy = int(total_vars * self.destroy_ratio)
        print(
            f"[LNS] Phase 2: Destroy & Repair "
            f"(max_iter={self.max_iter}, destroy={n_destroy}/{total_vars})"
        )

        all_keys = [(l, k, t) for l in range(L) for k in range(K) for t in range(T)]
        no_improve = 0

        for iteration in range(1, self.max_iter + 1):
            free_set = set(random.sample(all_keys, n_destroy))

            t0 = time.time()
            new_obj, new_sol, new_u = self._repair(best_u, free_set, p, price, pv_output, q_req)
            repair_time = time.time() - t0

            if new_sol is not None and new_obj < best_obj - 1e-2:
                improvement = best_obj - new_obj
                best_obj = new_obj
                best_u = new_u.copy()
                best_sol = new_sol
                no_improve = 0
                self.history.append(
                    {"iter": iteration, "obj": best_obj, "status": "improved", "time": repair_time}
                )
                print(
                    f"  Iter {iteration}: obj=${best_obj:,.0f} "
                    f"(↓${improvement:,.0f}), time={repair_time:.1f}s"
                )
            else:
                no_improve += 1
                status = "infeasible" if new_sol is None else "no_improve"
                self.history.append(
                    {"iter": iteration, "obj": best_obj, "status": status, "time": repair_time}
                )
                if iteration % 5 == 0:
                    print(
                        f"  Iter {iteration}: no improvement "
                        f"(stall={no_improve}), time={repair_time:.1f}s"
                    )

            if no_improve >= 10:
                print(f"  Converged at iter {iteration}.")
                break

        self.solve_time = time.time() - total_start
        print(
            f"[LNS] Done. obj=${best_obj:,.0f}, "
            f"prod={best_sol['total_production']:,.0f}ton, "
            f"iters={len(self.history)}, time={self.solve_time:.1f}s"
        )
        return best_sol

    def _get_initial_solution(self, p, price, pv_output, q_req):
        model_obj = IndustrialModel(p)
        m = model_obj.build(price, pv_output, q_req)

        solver = create_solver(self.time_limit)
        solver.options["MIPFocus"] = 1

        t0 = time.time()
        results = solve_model(solver, m, tee=False)
        init_time = time.time() - t0
        status = str(results.solver.termination_condition).lower()

        has_solution = len(results.solution) > 0
        if status not in ("optimal", "feasible", "locallyoptimal") and has_solution:
            status = "feasible"

        print(f"  [Init] {status}, time={init_time:.1f}s")

        if status not in ("optimal", "feasible", "locallyoptimal"):
            return None, float("inf"), None

        model_obj.model = m
        sol = model_obj.extract_results()

        L, K, T = p.num_lines, p.num_processes, p.num_steps
        u = np.zeros((L, K, T), dtype=int)
        for l in range(L):
            for k in range(K):
                for t in range(T):
                    u[l, k, t] = int(round(sol["u"][l][k][t]))

        return u, sol["objective"], sol

    def _repair(self, u, free_set, p, price, pv_output, q_req):
        L, K, T = p.num_lines, p.num_processes, p.num_steps

        model_obj = IndustrialModel(p)
        m = model_obj.build(price, pv_output, q_req)

        for l in range(L):
            for k in range(K):
                for t in range(T):
                    if (l, k, t) not in free_set:
                        m.u[l, k, t].fix(int(u[l, k, t]))
                    else:
                        m.u[l, k, t].set_value(int(u[l, k, t]))

        solver = create_solver(self.time_limit)

        results = solve_model(solver, m, tee=False, warmstart=True)
        status = str(results.solver.termination_condition).lower()

        has_solution = len(results.solution) > 0
        if status not in ("optimal", "feasible", "locallyoptimal") and has_solution:
            status = "feasible"

        if status not in ("optimal", "feasible", "locallyoptimal"):
            return float("inf"), None, None

        model_obj.model = m
        sol = model_obj.extract_results()

        new_u = np.zeros((L, K, T), dtype=int)
        for l in range(L):
            for k in range(K):
                for t in range(T):
                    new_u[l, k, t] = int(round(pyo.value(m.u[l, k, t])))

        return sol["objective"], sol, new_u

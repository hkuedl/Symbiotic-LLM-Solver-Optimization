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


class SimulatedAnnealingSolver:
    """SimulatedAnnealingSolver: reference optimization method."""

    def __init__(
        self,
        max_iter=200,
        T0=5000.0,
        alpha=0.95,
        perturb_ratio=0.15,
        seed=42,
        time_limit=DEFAULT_TIME_LIMIT,
    ):
        self.time_limit = validate_time_limit(time_limit)
        self.max_iter = max_iter
        self.T0 = T0
        self.alpha = alpha
        self.perturb_ratio = perturb_ratio
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

        print(f"[SA] Phase 1: Initial solution (gap={MIP_GAP * 100:.0f}%)")
        current_u, current_obj, current_sol = self._get_initial_solution(p, price, pv_output, q_req)

        if current_sol is None:
            self.solve_time = time.time() - total_start
            return None

        best_u = current_u.copy()
        best_obj = current_obj
        best_sol = current_sol

        self.history.append(
            {
                "iter": 0,
                "obj": current_obj,
                "best_obj": best_obj,
                "temp": self.T0,
                "status": "initial",
            }
        )
        print(f"[SA] Initial obj=${current_obj:,.0f}")

        n_perturb = max(1, int(total_vars * self.perturb_ratio))
        print(
            f"[SA] Phase 2: Simulated Annealing "
            f"(max_iter={self.max_iter}, T0={self.T0:.0f}, "
            f"alpha={self.alpha}, perturb={n_perturb}/{total_vars})"
        )

        all_keys = [(l, k, t) for l in range(L) for k in range(K) for t in range(T)]

        temp = self.T0
        no_improve_count = 0
        accept_count = 0

        for iteration in range(1, self.max_iter + 1):
            perturb_set = set(random.sample(all_keys, n_perturb))
            neighbor_u = current_u.copy()

            for l, k, t in perturb_set:
                neighbor_u[l, k, t] = 1 - neighbor_u[l, k, t]

            t0 = time.time()
            neighbor_obj, neighbor_sol, neighbor_u_fixed = self._repair(
                neighbor_u, perturb_set, p, price, pv_output, q_req
            )
            repair_time = time.time() - t0

            if neighbor_sol is None:
                accept = False
                status = "infeasible"
            else:
                delta = neighbor_obj - current_obj

                if delta < -1e-6:
                    accept = True
                    status = "improved"
                elif temp > 1e-6:
                    prob = np.exp(-delta / temp)
                    accept = random.random() < prob
                    status = f"worse_accepted(p={prob:.3f})" if accept else "rejected"
                else:
                    accept = False
                    status = "rejected(T≈0)"

            if accept:
                current_u = neighbor_u_fixed.copy()
                current_obj = neighbor_obj
                current_sol = neighbor_sol
                accept_count += 1

            if neighbor_sol is not None and neighbor_obj < best_obj - 1e-2:
                improvement = best_obj - neighbor_obj
                best_obj = neighbor_obj
                best_u = neighbor_u_fixed.copy()
                best_sol = neighbor_sol
                no_improve_count = 0
                print(
                    f"  Iter {iteration}: obj=${neighbor_obj:,.0f} "
                    f"(↓${improvement:,.0f}) ★, T={temp:.1f}, "
                    f"time={repair_time:.1f}s"
                )
            else:
                no_improve_count += 1

            self.history.append(
                {
                    "iter": iteration,
                    "obj": current_obj,
                    "best_obj": best_obj,
                    "neighbor_obj": neighbor_obj if neighbor_sol else float("inf"),
                    "temp": temp,
                    "accept": accept,
                    "status": status,
                    "time": repair_time,
                }
            )

            if iteration % 10 == 0:
                accept_rate = accept_count / 10
                print(
                    f"  Iter {iteration}: best=${best_obj:,.0f}, "
                    f"current=${current_obj:,.0f}, T={temp:.1f}, "
                    f"accept_rate={accept_rate:.0%}, stall={no_improve_count}"
                )
                accept_count = 0

            temp *= self.alpha

            if temp < 1.0 and no_improve_count >= 30:
                print(f"  Converged at iter {iteration} (T={temp:.2f}, stall={no_improve_count})")
                break

        self.solve_time = time.time() - total_start
        print(
            f"[SA] Done. obj=${best_obj:,.0f}, "
            f"prod={best_sol['total_production']:,.0f}ton, "
            f"iters={len(self.history)}, time={self.solve_time:.1f}s"
        )
        return best_sol

    def _get_initial_solution(self, p, price, pv_output, q_req):
        """Construct the initial feasible schedule for neighborhood search."""
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

    def _repair(self, u, perturb_set, p, price, pv_output, q_req):
        """Repair a candidate schedule using the original feasibility procedure."""
        L, K, T = p.num_lines, p.num_processes, p.num_steps

        model_obj = IndustrialModel(p)
        m = model_obj.build(price, pv_output, q_req)

        for l in range(L):
            for k in range(K):
                for t in range(T):
                    if (l, k, t) not in perturb_set:
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

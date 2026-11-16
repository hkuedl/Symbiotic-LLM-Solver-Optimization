"""Reference optimization method."""

import random
import time
from collections import deque

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


class TabuSearchSolver:
    """TabuSearchSolver: reference optimization method."""

    def __init__(
        self,
        max_iter=200,
        tabu_tenure=5,
        n_candidates=5,
        perturb_ratio=0.15,
        diversify_ratio=0.3,
        diversify_interval=20,
        seed=42,
        time_limit=DEFAULT_TIME_LIMIT,
    ):
        self.time_limit = validate_time_limit(time_limit)
        self.max_iter = max_iter
        self.tabu_tenure = tabu_tenure
        self.n_candidates = n_candidates
        self.perturb_ratio = perturb_ratio
        self.diversify_ratio = diversify_ratio
        self.diversify_interval = diversify_interval
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

        print(f"[Tabu] Phase 1: Initial solution (gap={MIP_GAP * 100:.0f}%)")
        current_u, current_obj, current_sol = self._get_initial_solution(p, price, pv_output, q_req)

        if current_sol is None:
            self.solve_time = time.time() - total_start
            return None

        best_u = current_u.copy()
        best_obj = current_obj
        best_sol = current_sol

        tabu_list = deque(maxlen=self.tabu_tenure)
        current_hash = self._hash_solution(current_u)
        tabu_list.append(current_hash)

        self.history.append(
            {
                "iter": 0,
                "obj": current_obj,
                "best_obj": best_obj,
                "status": "initial",
                "tabu_size": len(tabu_list),
            }
        )
        print(f"[Tabu] Initial obj=${current_obj:,.0f}")

        n_perturb = max(1, int(total_vars * self.perturb_ratio))
        n_diversify = max(1, int(total_vars * self.diversify_ratio))

        print(
            f"[Tabu] Phase 2: Tabu Search "
            f"(max_iter={self.max_iter}, tenure={self.tabu_tenure}, "
            f"candidates={self.n_candidates}, perturb={n_perturb}/{total_vars})"
        )

        all_keys = [(l, k, t) for l in range(L) for k in range(K) for t in range(T)]

        no_improve_count = 0
        diversify_count = 0

        for iteration in range(1, self.max_iter + 1):
            if no_improve_count >= self.diversify_interval:
                n_move = n_diversify
                diversify_mode = True
                diversify_count += 1
            else:
                n_move = n_perturb
                diversify_mode = False

            t0 = time.time()
            candidates = []

            for c in range(self.n_candidates):
                perturb_set = set(random.sample(all_keys, n_move))
                neighbor_u = current_u.copy()

                for l, k, t in perturb_set:
                    if random.random() < 0.8:
                        neighbor_u[l, k, t] = 1 - neighbor_u[l, k, t]
                    else:
                        neighbor_u[l, k, t] = random.randint(0, 1)

                neighbor_obj, neighbor_sol, neighbor_u_fixed = self._repair(
                    neighbor_u, perturb_set, p, price, pv_output, q_req
                )

                if neighbor_sol is not None:
                    neighbor_hash = self._hash_solution(neighbor_u_fixed)
                    candidates.append(
                        {
                            "u": neighbor_u_fixed,
                            "obj": neighbor_obj,
                            "sol": neighbor_sol,
                            "hash": neighbor_hash,
                            "is_tabu": neighbor_hash in tabu_list,
                        }
                    )

            search_time = time.time() - t0

            if not candidates:
                print(f"  Iter {iteration}: No feasible candidate found")
                self.history.append(
                    {
                        "iter": iteration,
                        "obj": current_obj,
                        "best_obj": best_obj,
                        "status": "no_candidate",
                        "tabu_size": len(tabu_list),
                        "time": search_time,
                    }
                )
                continue

            selected = self._select_candidate(candidates, best_obj)

            if selected is None:
                print(f"  Iter {iteration}: All candidates tabu, no aspiration")
                self.history.append(
                    {
                        "iter": iteration,
                        "obj": current_obj,
                        "best_obj": best_obj,
                        "status": "all_tabu",
                        "tabu_size": len(tabu_list),
                        "time": search_time,
                    }
                )
                continue

            current_u = selected["u"].copy()
            current_obj = selected["obj"]
            current_sol = selected["sol"]
            current_hash = selected["hash"]

            tabu_list.append(current_hash)

            if current_obj < best_obj - 1e-2:
                improvement = best_obj - current_obj
                best_obj = current_obj
                best_u = current_u.copy()
                best_sol = current_sol
                no_improve_count = 0
                status = "improved★"
                print(
                    f"  Iter {iteration}: obj=${current_obj:,.0f} "
                    f"(↓${improvement:,.0f}) ★, "
                    f"tabu_size={len(tabu_list)}, time={search_time:.1f}s"
                )
            else:
                no_improve_count += 1
                if selected["is_tabu"]:
                    status = "aspired"
                else:
                    status = "accepted"

            self.history.append(
                {
                    "iter": iteration,
                    "obj": current_obj,
                    "best_obj": best_obj,
                    "status": status,
                    "tabu_size": len(tabu_list),
                    "diversify": diversify_mode,
                    "n_candidates": len(candidates),
                    "time": search_time,
                }
            )

            if iteration % 10 == 0:
                diversify_text = f", diversify={diversify_count}" if diversify_count > 0 else ""
                print(
                    f"  Iter {iteration}: best=${best_obj:,.0f}, "
                    f"current=${current_obj:,.0f}, "
                    f"stall={no_improve_count}, tabu={len(tabu_list)}"
                    f"{diversify_text}"
                )

            if no_improve_count >= 3 * self.diversify_interval:
                print(f"  Converged at iter {iteration} (stall={no_improve_count})")
                break

        self.solve_time = time.time() - total_start
        print(
            f"[Tabu] Done. obj=${best_obj:,.0f}, "
            f"prod={best_sol['total_production']:,.0f}ton, "
            f"iters={len(self.history)}, time={self.solve_time:.1f}s, "
            f"diversifications={diversify_count}"
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

    def _hash_solution(self, u):
        """Compute the schedule signature used by the tabu list."""

        L, K, T = u.shape
        fingerprint = []

        for l in range(L):
            for k in range(K):
                segments = []
                start = None
                for t in range(T):
                    if u[l, k, t] == 1:
                        if start is None:
                            start = t
                    else:
                        if start is not None:
                            segments.append((start, t - 1))
                            start = None
                if start is not None:
                    segments.append((start, T - 1))

                fingerprint.append((l, k, tuple(segments)))

        return hash(tuple(fingerprint))

    def _select_candidate(self, candidates, best_obj):

        non_tabu = [c for c in candidates if not c["is_tabu"]]
        tabu_candidates = [c for c in candidates if c["is_tabu"]]

        if non_tabu:
            non_tabu.sort(key=lambda c: c["obj"])
            return non_tabu[0]

        aspired = [c for c in tabu_candidates if c["obj"] < best_obj - 1e-2]
        if aspired:
            aspired.sort(key=lambda c: c["obj"])
            return aspired[0]

        if tabu_candidates:
            tabu_candidates.sort(key=lambda c: c["obj"])
            return tabu_candidates[0]

        return None

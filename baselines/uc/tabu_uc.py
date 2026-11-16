"""Reference optimization method."""

import logging
import random
import time
from collections import deque

import numpy as np
import pyomo.environ as pyo

from baselines.uc.lr_uc import LagrangianRelaxationSolver
from common.solver_config import DEFAULT_TIME_LIMIT, create_solver, solve_model, validate_time_limit
from models.uc.model_uc import UnitCommitmentModel


class TabuSearchSolver:
    """TabuSearchSolver: reference optimization method."""

    def __init__(
        self,
        max_iter=100,
        n_candidates=3,
        tabu_tenure=10,
        perturb_ratio=0.05,
        diversify_ratio=0.1,
        diversify_interval=10,
        lr_warmup_iters=25,
        seed=42,
        time_limit=DEFAULT_TIME_LIMIT,
    ):
        self.time_limit = validate_time_limit(time_limit)
        self.max_iter = max_iter
        self.n_candidates = n_candidates
        self.tabu_tenure = tabu_tenure
        self.perturb_ratio = perturb_ratio
        self.diversify_ratio = diversify_ratio
        self.diversify_interval = diversify_interval
        self.lr_warmup_iters = lr_warmup_iters
        self.seed = seed
        self.solve_time = 0
        self.history = []

    def solve(self, model_obj, demand_bus, reserve, wind=None, solar=None):
        """Run the method and return the extracted solution, or None on failure."""
        p = model_obj.p
        if model_obj.model is None:
            raise RuntimeError("Model not built yet.")

        total_start = time.time()
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.history = []

        G, T = p.num_generators, p.num_steps
        total_vars = G * T

        self._p = p
        self._demand_bus = demand_bus
        self._reserve = reserve
        self._wind = wind
        self._solar = solar

        demand_total = np.array(
            [sum(demand_bus[n][t] for n in range(p.num_buses)) for t in range(T)]
        )
        self._demand_total = demand_total

        self._base_model_obj = UnitCommitmentModel(p)
        self._base_m = self._base_model_obj.build(demand_bus, reserve, wind, solar)
        self._solver = create_solver(self.time_limit)

        # Phase 1: LR warmup
        print(f"[TS] Phase 1: LR warmup ({self.lr_warmup_iters} iters)")
        current_u, current_obj, current_sol = self._get_initial_solution(
            model_obj, demand_bus, reserve, wind, solar
        )

        if current_sol is None:
            self._cleanup()
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
        print(f"[TS] Seed obj=${best_obj:,.0f}")

        # Phase 2: Tabu Search
        n_perturb = max(1, int(total_vars * self.perturb_ratio))
        n_diversify = max(1, int(total_vars * self.diversify_ratio))

        print(
            f"[TS] Phase 2: Tabu Search "
            f"(max_iter={self.max_iter}, candidates={self.n_candidates}, "
            f"tenure={self.tabu_tenure}, "
            f"perturb={n_perturb}/{total_vars})"
        )

        all_keys = [(g, t) for g in range(G) for t in range(T)]

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
                perturb_set = self._select_perturb_set(
                    current_u, p, all_keys, n_move, diversify_mode
                )

                neighbor_u = current_u.copy()
                for g, t in perturb_set:
                    if random.random() < 0.8:
                        neighbor_u[g, t] = 1 - neighbor_u[g, t]
                    else:
                        neighbor_u[g, t] = random.randint(0, 1)

                neighbor_obj, neighbor_sol, neighbor_u_fixed = self._repair(
                    neighbor_u, perturb_set, p
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
                no_improve_count += 1
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
                if iteration % 10 == 0:
                    print(f"  Iter {iteration}: no feasible candidate, stall={no_improve_count}")
                continue

            selected = self._select_candidate(candidates, best_obj)

            if selected is None:
                no_improve_count += 1
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
            tabu_list.append(selected["hash"])

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
                    f"time={search_time:.1f}s"
                )
            else:
                no_improve_count += 1
                status = "aspired" if selected["is_tabu"] else "accepted"

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
                div_text = f", diversify={diversify_count}" if diversify_count > 0 else ""
                print(
                    f"  Iter {iteration}: best=${best_obj:,.0f}, "
                    f"current=${current_obj:,.0f}, "
                    f"stall={no_improve_count}, "
                    f"tabu={len(tabu_list)}{div_text}"
                )

            if no_improve_count >= 3 * self.diversify_interval:
                print(f"  Converged at iter {iteration} (stall={no_improve_count})")
                break

        self._cleanup()
        self.solve_time = time.time() - total_start
        print(
            f"[TS] Done. best=${best_obj:,.0f}, "
            f"iters={len(self.history)}, "
            f"diversifications={diversify_count}, "
            f"time={self.solve_time:.1f}s"
        )
        return best_sol

    # ------------------------------------------------------------------
    #  Cleanup cached model
    # ------------------------------------------------------------------
    def _cleanup(self):
        self._base_model_obj = None
        self._base_m = None
        self._solver = None

    # ------------------------------------------------------------------
    #  Initial solution via LR warmup
    # ------------------------------------------------------------------
    def _get_initial_solution(self, model_obj, demand_bus, reserve, wind, solar):
        p = model_obj.p

        lr = LagrangianRelaxationSolver(time_limit=self.time_limit)
        best_u, best_ub = lr.warmup(
            model_obj,
            demand_bus,
            reserve,
            wind,
            solar,
            warmup_iters=self.lr_warmup_iters,
            recovery_interval=3,
            verbose=True,
        )

        if best_u is None:
            return None, float("inf"), None

        obj, sol, u_fixed = self._repair(best_u, set(), p)

        if sol is None:
            return None, float("inf"), None

        return u_fixed, obj, sol

    # ------------------------------------------------------------------
    #  Intelligent perturbation set selection
    # ------------------------------------------------------------------
    def _select_perturb_set(self, u, p, all_keys, n_move, diversify_mode):
        G, T = p.num_generators, p.num_steps

        if diversify_mode:
            return set(random.sample(all_keys, min(n_move, len(all_keys))))

        perturb_set = set()
        strategy = random.random()

        if strategy < 0.4:
            n_gens = random.randint(1, min(3, G))
            gens = random.sample(range(G), n_gens)
            for g in gens:
                for t in range(T):
                    perturb_set.add((g, t))

        elif strategy < 0.7:
            window = random.randint(max(1, T // 8), max(1, T // 3))
            t_start = random.randint(0, max(0, T - window))
            n_gens = random.randint(max(1, G // 5), max(1, G // 2))
            gens = random.sample(range(G), n_gens)
            for g in gens:
                for t in range(t_start, min(t_start + window, T)):
                    perturb_set.add((g, t))

        else:
            perturb_set = set(random.sample(all_keys, min(n_move, len(all_keys))))

        max_perturb = int(G * T * 0.5)
        if len(perturb_set) > max_perturb:
            perturb_set = set(random.sample(list(perturb_set), max_perturb))
        if len(perturb_set) < 1:
            perturb_set = set(random.sample(all_keys, min(n_move, len(all_keys))))

        return perturb_set

    # ------------------------------------------------------------------
    #  Repair via sub-MIP (reuse pre-built model)
    # ------------------------------------------------------------------
    def _repair(self, u, perturb_set, p):
        G, T = p.num_generators, p.num_steps
        m = self._base_m

        for g in range(G):
            for t in range(T):
                m.u[g, t].unfix()
                m.v[g, t].unfix()
                m.w[g, t].unfix()

        for g in range(G):
            for t in range(T):
                val = int(u[g, t])
                if t == 0:
                    prev = p.u0[g]
                else:
                    prev = int(u[g, t - 1])

                if (g, t) not in perturb_set:
                    m.u[g, t].fix(val)

                    prev_is_fixed = (t == 0) or ((g, t - 1) not in perturb_set)
                    if prev_is_fixed:
                        m.v[g, t].fix(max(0, val - prev))
                        m.w[g, t].fix(max(0, prev - val))
                    else:
                        m.v[g, t].set_value(max(0, val - prev))
                        m.w[g, t].set_value(max(0, prev - val))
                else:
                    m.u[g, t].set_value(val)
                    m.v[g, t].set_value(max(0, val - prev))
                    m.w[g, t].set_value(max(0, prev - val))

        logging.getLogger("pyomo.core").setLevel(logging.ERROR)
        results = solve_model(self._solver, m, tee=False, warmstart=True)
        logging.getLogger("pyomo.core").setLevel(logging.WARNING)

        status = str(results.solver.termination_condition).lower()
        has_solution = len(results.solution) > 0
        if status not in ("optimal", "feasible", "locallyoptimal") and has_solution:
            status = "feasible"

        if status not in ("optimal", "feasible", "locallyoptimal"):
            return float("inf"), None, None

        self._base_model_obj.model = m
        sol = self._base_model_obj.extract_results()

        new_u = np.zeros((G, T), dtype=int)
        for g in range(G):
            for t in range(T):
                new_u[g, t] = int(round(pyo.value(m.u[g, t])))

        return sol["objective"], sol, new_u

    # ------------------------------------------------------------------
    #  Solution hashing for tabu list
    # ------------------------------------------------------------------
    @staticmethod
    def _hash_solution(u):
        G, T = u.shape
        fingerprint = []
        for g in range(G):
            segments = []
            start = None
            for t in range(T):
                if u[g, t] == 1:
                    if start is None:
                        start = t
                else:
                    if start is not None:
                        segments.append((start, t - 1))
                        start = None
            if start is not None:
                segments.append((start, T - 1))
            fingerprint.append((g, tuple(segments)))
        return hash(tuple(fingerprint))

    # ------------------------------------------------------------------
    #  Candidate selection with aspiration
    # ------------------------------------------------------------------
    @staticmethod
    def _select_candidate(candidates, best_obj):
        non_tabu = [c for c in candidates if not c["is_tabu"]]
        tabu_cands = [c for c in candidates if c["is_tabu"]]

        if non_tabu:
            non_tabu.sort(key=lambda c: c["obj"])
            return non_tabu[0]

        aspired = [c for c in tabu_cands if c["obj"] < best_obj - 1e-2]
        if aspired:
            aspired.sort(key=lambda c: c["obj"])
            return aspired[0]

        if tabu_cands:
            tabu_cands.sort(key=lambda c: c["obj"])
            return tabu_cands[0]

        return None

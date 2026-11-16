"""Reference optimization method."""

import logging
import math
import random
import time

import numpy as np

from baselines.uc.lr_uc import LagrangianRelaxationSolver
from common.solver_config import DEFAULT_TIME_LIMIT, create_solver, solve_model, validate_time_limit


class SimulatedAnnealingSolver:
    """SimulatedAnnealingSolver: reference optimization method."""

    def __init__(
        self,
        T_start=50000.0,
        T_end=100.0,
        cooling_rate=0.95,
        max_iter=200,
        reheat_interval=50,
        reheat_factor=1.5,
        stall_limit=60,
        lr_warmup_iters=15,
        seed=42,
        time_limit=DEFAULT_TIME_LIMIT,
    ):
        self.time_limit = validate_time_limit(time_limit)
        self.T_start = T_start
        self.T_end = T_end
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.reheat_interval = reheat_interval
        self.reheat_factor = reheat_factor
        self.stall_limit = stall_limit
        self.lr_warmup_iters = lr_warmup_iters
        self.seed = seed
        self.solve_time = 0
        self.history = []
        self._solver = None
        self._demand_total = None
        self._reserve = None
        self._switching_gens = []
        self._baseload_gens = []
        self._offline_gens = []
        self._online_by_cost = []

    def solve(self, model_obj, demand_bus, demand_total, reserve, wind=None, solar=None):
        """Run the method and return the extracted solution, or None on failure."""
        p = model_obj.p
        if model_obj.model is None:
            raise RuntimeError("Model not built yet.")

        total_start = time.time()
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.history = []

        G, T = p.num_generators, p.num_steps
        self._demand_total = list(demand_total)
        self._reserve = list(reserve)
        self._solver = create_solver(self.time_limit)

        print(f"[SA] Phase 1: LR warmup ({self.lr_warmup_iters} iters)")
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
            seed_obj, seed_sol = self._evaluate(lr_u, model_obj)
        else:
            seed_sol = None

        if seed_sol is None:
            print("[SA] Failed to obtain initial feasible solution.")
            self.solve_time = time.time() - total_start
            return None

        print(f"[SA] Seed obj=${seed_obj:,.0f}")
        self._analyze_seed(lr_u, p)

        current_u = lr_u.copy()
        current_obj = seed_obj
        current_sol = seed_sol
        best_u = lr_u.copy()
        best_obj = seed_obj
        best_sol = seed_sol

        print(
            f"[SA] Phase 2: annealing "
            f"(T0={self.T_start:.0f}, Tf={self.T_end:.0f}, "
            f"α={self.cooling_rate}, max_iter={self.max_iter})"
        )

        temp = self.T_start
        no_improve = 0
        n_accepted = 0
        n_improved = 0
        n_total = 0

        nbr_names = ["shift", "swap", "window", "insert", "decommit"]
        nbr_weights = [3.0, 5.0, 2.0, 1.0, 3.0]
        nbr_success = [0, 0, 0, 0, 0]
        nbr_tries = [0, 0, 0, 0, 0]
        nbr_funcs = [
            self._neighbor_shift,
            self._neighbor_swap,
            self._neighbor_window,
            self._neighbor_insert,
            self._neighbor_decommit,
        ]

        for it in range(self.max_iter):
            iter_start = time.time()

            nbr_idx = self._weighted_choice(nbr_weights)
            nbr_tries[nbr_idx] += 1

            neighbor_u = current_u.copy()
            changed = nbr_funcs[nbr_idx](neighbor_u, p)

            if not changed:
                self.history.append(
                    {
                        "iter": it,
                        "temp": temp,
                        "best_obj": best_obj,
                        "current_obj": current_obj,
                        "action": "skip",
                        "nbr": nbr_names[nbr_idx],
                    }
                )
                temp *= self.cooling_rate
                continue

            self._repair(neighbor_u, p)

            t_eval = time.time()
            nbr_obj, nbr_sol = self._evaluate(neighbor_u, model_obj)
            eval_time = time.time() - t_eval
            n_total += 1

            if nbr_sol is None:
                self.history.append(
                    {
                        "iter": it,
                        "temp": temp,
                        "best_obj": best_obj,
                        "current_obj": current_obj,
                        "action": "infeasible",
                        "nbr": nbr_names[nbr_idx],
                        "eval_time": eval_time,
                    }
                )
                print(
                    f"  Iter {it}: infeasible, T={temp:.0f}, {nbr_names[nbr_idx]}, {eval_time:.0f}s"
                )
                temp *= self.cooling_rate
                continue

            delta = nbr_obj - current_obj
            accept = False

            if delta < 0:
                accept = True
                n_improved += 1
                nbr_success[nbr_idx] += 1
            elif temp > 0:
                if random.random() < math.exp(-delta / temp):
                    accept = True

            if accept:
                current_u = neighbor_u
                current_obj = nbr_obj
                current_sol = nbr_sol
                n_accepted += 1

            if nbr_obj < best_obj:
                best_u = neighbor_u.copy()
                best_obj = nbr_obj
                best_sol = nbr_sol
                no_improve = 0
                action, star = "new_best", " ★"
            elif accept:
                no_improve += 1
                action, star = "accept", ""
            else:
                no_improve += 1
                action, star = "reject", ""

            self.history.append(
                {
                    "iter": it,
                    "temp": temp,
                    "best_obj": best_obj,
                    "current_obj": current_obj,
                    "nbr_obj": nbr_obj,
                    "delta": delta,
                    "action": action,
                    "nbr": nbr_names[nbr_idx],
                    "eval_time": eval_time,
                }
            )

            print(
                f"  Iter {it}: ${nbr_obj:,.0f} (Δ={delta:+,.0f}), "
                f"{action}{star}, T={temp:.0f}, {nbr_names[nbr_idx]}, "
                f"best=${best_obj:,.0f}, {time.time() - iter_start:.0f}s"
            )

            temp *= self.cooling_rate

            if (
                no_improve > 0
                and no_improve % self.reheat_interval == 0
                and temp < self.T_start * 0.5
            ):
                old_temp = temp
                temp = min(temp * self.reheat_factor, self.T_start * 0.3)
                print(f"  [Reheat] T: {old_temp:.0f} → {temp:.0f}")

                current_u = best_u.copy()
                current_obj = best_obj
                current_sol = best_sol

            if (it + 1) % 20 == 0:
                self._update_weights(nbr_weights, nbr_success, nbr_tries)
                print(f"  [Weights] {dict(zip(nbr_names, [f'{w:.1f}' for w in nbr_weights]))}")
                nbr_success = [0] * 5
                nbr_tries = [0] * 5

            if temp < self.T_end:
                print(f"  Temperature reached T_end={self.T_end:.0f}")
                break
            if no_improve >= self.stall_limit:
                print(f"  No improvement for {self.stall_limit} iters")
                break

        self.solve_time = time.time() - total_start
        print(
            f"\n[SA] Done. best=${best_obj:,.0f}, iters={it + 1}, "
            f"accept={n_accepted / max(n_total, 1):.0%}, "
            f"improve={n_improved / max(n_total, 1):.0%}, "
            f"time={self.solve_time:.1f}s"
        )

        return best_sol

    def _analyze_seed(self, seed_u, p):
        """Classify generators by their commitment pattern in the seed solution."""
        G, T = p.num_generators, p.num_steps
        self._switching_gens = []
        self._baseload_gens = []
        self._offline_gens = []

        for g in range(G):
            on = int(np.sum(seed_u[g, :]))
            if on == T:
                self._baseload_gens.append(g)
            elif on == 0:
                self._offline_gens.append(g)
            else:
                self._switching_gens.append(g)

        self._online_by_cost = sorted(
            [g for g in range(G) if np.sum(seed_u[g, :]) > 0], key=lambda g: -p.cost_b[g]
        )

        print(
            f"[SA] Seed: baseload={len(self._baseload_gens)}, "
            f"marginal={len(self._switching_gens)}, "
            f"offline={len(self._offline_gens)}"
        )

    def _neighbor_shift(self, u, p):
        """Shift a randomly selected startup or shutdown transition."""
        G, T = p.num_generators, p.num_steps
        cands = self._switching_gens if self._switching_gens else list(range(G))
        random.shuffle(cands)

        for g in cands:
            trans = [t for t in range(1, T) if u[g, t] != u[g, t - 1]]
            if not trans:
                continue

            tt = random.choice(trans)
            shift = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            tn = max(1, min(T - 1, tt + shift))

            if tn == tt:
                continue

            if u[g, tt] == 1 and u[g, tt - 1] == 0:
                rng = range(tt, min(tn, T)) if shift > 0 else range(max(tn, 0), tt)
                val = 0 if shift > 0 else 1
            elif u[g, tt] == 0 and u[g, tt - 1] == 1:
                rng = range(tt, min(tn, T)) if shift > 0 else range(max(tn, 0), tt)
                val = 1 if shift > 0 else 0
            else:
                continue

            for t in rng:
                u[g, t] = val
            return True

        return False

    def _neighbor_swap(self, u, p):
        """Replace an expensive online generator with a cheaper alternative."""
        G, T = p.num_generators, p.num_steps

        exp = [g for g in self._online_by_cost if 0 < np.sum(u[g, :]) < T]
        if not exp:
            return False

        g_off = random.choice(exp[: max(1, len(exp) // 3)])

        segs = []
        t = 0
        while t < T:
            if u[g_off, t] == 1:
                ts = t
                while t < T and u[g_off, t] == 1:
                    t += 1
                segs.append((ts, t))
            else:
                t += 1

        if not segs:
            return False

        ss, se = random.choice(segs)

        cheaper = sorted(
            [
                g
                for g in range(G)
                if g != g_off
                and p.cost_b[g] < p.cost_b[g_off]
                and p.P_max[g] >= p.P_min[g_off] * 0.5
                and np.sum(u[g, ss:se]) < (se - ss)
            ],
            key=lambda g: p.cost_b[g],
        )

        if not cheaper:
            return False

        for t in range(ss, se):
            u[g_off, t] = 0
            u[cheaper[0], t] = 1

        return True

    def _neighbor_window(self, u, p):
        """Flip generator states within a randomly selected time window."""
        G, T = p.num_generators, p.num_steps

        win = random.randint(4, min(12, T))
        ts = random.randint(0, T - win)

        cands = self._switching_gens if self._switching_gens else list(range(G))
        if not cands:
            return False

        for g in random.sample(cands, min(random.randint(1, 3), len(cands))):
            for t in range(ts, ts + win):
                u[g, t] = 1 - u[g, t]

        return True

    def _neighbor_insert(self, u, p):
        """Insert an online interval or remove an existing interval."""
        G, T = p.num_generators, p.num_steps

        if random.random() < 0.5 and self._offline_gens:
            g = random.choice(self._offline_gens)
            mu = p.min_up_time[g]
            if T - mu < 0:
                return False
            ts = random.randint(0, T - mu)
            for t in range(ts, ts + mu):
                u[g, t] = 1
            return True

        elif self._switching_gens:
            g = random.choice(self._switching_gens)
            segs = []
            t = 0
            while t < T:
                if u[g, t] == 1:
                    ts = t
                    while t < T and u[g, t] == 1:
                        t += 1
                    segs.append((ts, t))
                else:
                    t += 1
            if not segs:
                return False
            ss, se = random.choice(segs)
            for t in range(ss, se):
                u[g, t] = 0
            return True

        return False

    def _neighbor_decommit(self, u, p):
        """Switch off selected high-cost generators."""
        G, T = p.num_generators, p.num_steps

        online = [g for g in range(G) if np.sum(u[g, :]) > 0]
        if not online:
            return False

        pool = sorted(online, key=lambda g: -p.cost_b[g])[: max(1, len(online) // 4)]
        targets = random.sample(pool, min(random.randint(1, 3), len(pool)))

        changed = False
        for g in targets:
            if random.random() < 0.6:
                for t in range(T):
                    if u[g, t] == 1:
                        u[g, t] = 0
                        changed = True
            else:
                on_t = [t for t in range(T) if u[g, t] == 1]
                if not on_t:
                    continue
                win = random.randint(p.min_down_time[g], min(len(on_t), T // 2))
                si = random.randint(0, max(0, len(on_t) - win))
                for idx in range(si, min(si + win, len(on_t))):
                    u[g, on_t[idx]] = 0
                    changed = True

        return changed

    @staticmethod
    def _weighted_choice(weights):
        """Sample an operator index in proportion to its weight."""
        r = random.random() * sum(weights)
        c = 0
        for i, w in enumerate(weights):
            c += w
            if r <= c:
                return i
        return len(weights) - 1

    @staticmethod
    def _update_weights(weights, successes, tries):
        """Adapt operator weights using observed success rates."""
        for i in range(len(weights)):
            if tries[i] > 0:
                success_rate = successes[i] / tries[i]

                weights[i] = max(0.5, 0.7 * weights[i] + 0.3 * (1.0 + success_rate * 10.0))

    def _repair(self, u, p):
        """Repair a candidate schedule using the original feasibility procedure."""
        G, T = p.num_generators, p.num_steps

        for g in range(G):
            sh = p.initial_status_steps[g]
            if sh > 0:
                for t in range(min(max(0, p.min_up_time[g] - sh), T)):
                    u[g, t] = 1
            elif sh < 0:
                for t in range(min(max(0, p.min_down_time[g] - abs(sh)), T)):
                    u[g, t] = 0

        for g in range(G):
            for _ in range(10):
                if not self._fix_min_on(u, g, p.min_up_time[g], T) and not self._fix_min_off(
                    u, g, p.min_down_time[g], T
                ):
                    break

        self._fix_reserve(u, p)

    @staticmethod
    def _fix_min_on(u, g, min_up, T):
        """Repair online intervals shorter than the minimum uptime."""
        changed = False
        t = 0
        while t < T:
            if u[g, t] == 1:
                ts = t
                while t < T and u[g, t] == 1:
                    t += 1

                if t - ts < min_up:
                    changed = True
                    if ts + min_up <= T:
                        for tt in range(t, ts + min_up):
                            u[g, tt] = 1
                        t = ts + min_up
                    else:
                        for tt in range(ts, t):
                            u[g, tt] = 0
            else:
                t += 1
        return changed

    @staticmethod
    def _fix_min_off(u, g, min_down, T):
        """Repair offline intervals shorter than the minimum downtime."""
        changed = False
        t = 0
        while t < T:
            if u[g, t] == 0:
                ts = t
                while t < T and u[g, t] == 0:
                    t += 1

                if t - ts < min_down:
                    changed = True
                    if ts + min_down <= T:
                        for tt in range(t, ts + min_down):
                            u[g, tt] = 0
                        t = ts + min_down
                    else:
                        for tt in range(ts, t):
                            u[g, tt] = 1
            else:
                t += 1
        return changed

    def _fix_reserve(self, u, p):
        """Commit low-cost generators until reserve requirements are met."""
        G, T = p.num_generators, p.num_steps
        order = sorted(range(G), key=lambda g: p.cost_b[g])
        touched = set()

        for t in range(T):
            cap = sum(p.P_max[g] for g in range(G) if u[g, t] == 1)
            need = self._demand_total[t] + self._reserve[t]

            if cap >= need:
                continue

            for g in order:
                if u[g, t] == 1:
                    continue
                u[g, t] = 1
                touched.add(g)
                cap += p.P_max[g]
                if cap >= need:
                    break

        for g in touched:
            for _ in range(10):
                if not self._fix_min_on(u, g, p.min_up_time[g], T) and not self._fix_min_off(
                    u, g, p.min_down_time[g], T
                ):
                    break

    def _evaluate(self, u, model_obj):
        """Evaluate a candidate schedule and extract its solution once."""
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
        results = solve_model(self._solver, m, tee=False, load_solutions=False)
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

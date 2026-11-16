"""Reference optimization method."""

import logging
import time

import numpy as np
import pyomo.environ as pyo

from common.solver_config import DEFAULT_TIME_LIMIT, create_solver, solve_model, validate_time_limit


class LagrangianRelaxationSolver:
    """LagrangianRelaxationSolver: reference optimization method."""

    def __init__(
        self,
        max_iter=50,
        step_init=0.5,
        step_decay=0.95,
        recovery_interval=1,
        convergence_gap=0.01,
        seed=42,
        time_limit=DEFAULT_TIME_LIMIT,
    ):
        self.time_limit = validate_time_limit(time_limit)
        self.max_iter = max_iter
        self.step_init = step_init
        self.step_decay = step_decay
        self.recovery_interval = recovery_interval
        self.convergence_gap = convergence_gap
        self.seed = seed
        self.solve_time = 0
        self.history = []

    def solve(self, model_obj, demand_bus, reserve, wind, solar):
        """Run the method and return the extracted solution, or None on failure."""
        p = model_obj.p
        G, T = p.num_generators, p.num_steps
        dt = p.dt
        N = p.num_buses

        total_start = time.time()
        self.history = []

        demand_total = np.array([sum(demand_bus[n][t] for n in range(N)) for t in range(T)])
        reserve_arr = np.array(reserve)

        re_avail = np.zeros(T)
        if wind is not None:
            for bv in wind.values():
                re_avail += np.array(bv)
        if solar is not None:
            for bv in solar.values():
                re_avail += np.array(bv)

        solver = create_solver(self.time_limit)

        avg_mc = float(np.mean(p.cost_b))
        lambda_t = np.full(T, avg_mc)
        mu_t = np.zeros(T)

        print(f"[LR] Building {G} generator subproblems...")
        t_build = time.time()
        gen_models = [self._build_gen_model(g, p) for g in range(G)]
        stor_model = None
        if p.has_storage:
            stor_model = self._build_stor_model(p)
        print(f"[LR] Subproblems built in {time.time() - t_build:.1f}s")

        best_ub = float("inf")
        best_lb = -float("inf")
        best_sol = None

        for it in range(self.max_iter):
            iter_start = time.time()

            u_lr, P_lr, gen_obj_sum, n_fail = self._solve_subproblems(
                gen_models, lambda_t, mu_t, solver, p
            )
            sub_time = time.time() - iter_start

            if n_fail > 0:
                print(f"  WARNING: {n_fail}/{G} subproblems failed")

            re_used = np.where(lambda_t + p.curtailment_cost >= 0, re_avail, 0.0)
            re_curt = re_avail - re_used
            re_obj = float(np.sum((p.curtailment_cost * re_curt - lambda_t * re_used) * dt))

            stor_net = np.zeros(T)
            stor_obj = 0.0
            if p.has_storage and stor_model is not None:
                stor_obj, stor_net = self._solve_storage(stor_model, lambda_t, solver, p)

            constant = float(np.sum(lambda_t * demand_total + mu_t * (demand_total + reserve_arr)))
            dual_bound = gen_obj_sum + re_obj + stor_obj + constant

            if dual_bound > best_lb:
                best_lb = dual_bound

            primal_obj = float("inf")
            sol = None
            do_recovery = (it % self.recovery_interval == 0) or it == self.max_iter - 1

            fix_time = 0.0
            if do_recovery:
                u_fixed = u_lr.copy()
                self._repair(u_fixed, p, demand_total, reserve_arr)
                t_fix = time.time()
                primal_obj, sol = self._evaluate_fixed(u_fixed, model_obj, solver)
                fix_time = time.time() - t_fix
                if sol is not None and primal_obj < best_ub:
                    best_ub = primal_obj
                    best_sol = sol

            gap = (
                (best_ub - best_lb) / max(abs(best_ub), 1e-10)
                if best_ub < float("inf")
                else float("inf")
            )
            iter_time = time.time() - iter_start

            status = (
                "feasible" if sol is not None else ("skip" if not do_recovery else "infeasible")
            )
            self.history.append(
                {
                    "iter": it,
                    "lb": best_lb,
                    "ub": best_ub,
                    "gap": gap,
                    "status": status,
                    "objective": primal_obj if sol is not None else None,
                    "solve_time": iter_time,
                    "n_changed": 0,
                }
            )

            ub_str = f"${best_ub:,.0f}" if best_ub < float("inf") else "N/A"
            fix_str = ""
            if do_recovery:
                star = " ★" if (sol is not None and primal_obj <= best_ub) else ""
                fix_str = f", fix={fix_time:.0f}s{star}"
            print(
                f"  Iter {it}: LB=${best_lb:,.0f}, UB={ub_str}, "
                f"gap={gap:.2%}, sub={sub_time:.0f}s{fix_str}"
            )

            if gap < self.convergence_gap and best_ub < float("inf"):
                print(f"  Gap < {self.convergence_gap:.0%}, converged.")
                break

            total_gen = np.sum(P_lr, axis=0)
            sg_bal = demand_total - total_gen - re_used - stor_net
            cap_online = np.array(
                [sum(p.P_max[g] for g in range(G) if u_lr[g, t] == 1) for t in range(T)]
            )
            sg_res = demand_total + reserve_arr - cap_online
            sg_norm2 = float(np.sum(sg_bal**2) + np.sum(np.maximum(0.0, sg_res) ** 2))

            if sg_norm2 < 1e-10:
                print("  Subgradient ≈ 0, converged.")
                break

            alpha = self.step_init * (self.step_decay**it)
            if best_ub < float("inf") and best_ub > dual_bound:
                step = alpha * (best_ub - dual_bound) / sg_norm2
            else:
                step = alpha / np.sqrt(sg_norm2)

            lambda_t += step * sg_bal
            mu_t = np.maximum(0.0, mu_t + step * sg_res)

        self.solve_time = time.time() - total_start
        ub_str = f"${best_ub:,.0f}" if best_ub < float("inf") else "N/A"
        lb_str = f"${best_lb:,.0f}" if best_lb > -float("inf") else "N/A"
        final_gap = (
            (best_ub - best_lb) / max(abs(best_ub), 1e-10)
            if best_ub < float("inf")
            else float("inf")
        )
        print(
            f"[LR] Done. UB={ub_str}, LB={lb_str}, gap={final_gap:.2%}, time={self.solve_time:.1f}s"
        )
        return best_sol

    def warmup(
        self,
        model_obj,
        demand_bus,
        reserve,
        wind,
        solar,
        warmup_iters=15,
        recovery_interval=3,
        verbose=False,
    ):
        p = model_obj.p
        G, T = p.num_generators, p.num_steps
        dt = p.dt
        N = p.num_buses

        t0 = time.time()

        demand_total = np.array([sum(demand_bus[n][t] for n in range(N)) for t in range(T)])
        reserve_arr = np.array(reserve)

        re_avail = np.zeros(T)
        if wind is not None:
            for bv in wind.values():
                re_avail += np.array(bv)
        if solar is not None:
            for bv in solar.values():
                re_avail += np.array(bv)

        solver = create_solver(self.time_limit)

        avg_mc = float(np.mean(p.cost_b))
        lambda_t = np.full(T, avg_mc)
        mu_t = np.zeros(T)

        gen_models = [self._build_gen_model(g, p) for g in range(G)]

        best_u = None
        best_ub = float("inf")

        for it in range(warmup_iters):
            u_lr, P_lr, gen_obj_sum, n_fail = self._solve_subproblems(
                gen_models, lambda_t, mu_t, solver, p
            )

            re_used = np.where(lambda_t + p.curtailment_cost >= 0, re_avail, 0.0)
            re_curt = re_avail - re_used
            re_obj = float(np.sum((p.curtailment_cost * re_curt - lambda_t * re_used) * dt))

            constant = float(np.sum(lambda_t * demand_total + mu_t * (demand_total + reserve_arr)))
            dual_bound = gen_obj_sum + re_obj + constant

            do_recovery = (it % recovery_interval == 0) or (it == warmup_iters - 1)
            if do_recovery:
                u_fixed = u_lr.copy()
                self._repair(u_fixed, p, demand_total, reserve_arr)
                primal_obj, sol = self._evaluate_fixed(u_fixed, model_obj, solver)
                if sol is not None and primal_obj < best_ub:
                    best_ub = primal_obj
                    best_u = u_fixed.copy()
                    if verbose:
                        print(f"  [LR-Warmup] iter {it}: UB=${best_ub:,.0f} ★")

            total_gen = np.sum(P_lr, axis=0)
            sg_bal = demand_total - total_gen - re_used
            cap_online = np.array(
                [sum(p.P_max[g] for g in range(G) if u_lr[g, t] == 1) for t in range(T)]
            )
            sg_res = demand_total + reserve_arr - cap_online

            sg_norm2 = float(np.sum(sg_bal**2) + np.sum(np.maximum(0.0, sg_res) ** 2))

            if sg_norm2 < 1e-10:
                break

            alpha = self.step_init * (self.step_decay**it)
            if best_ub < float("inf") and best_ub > dual_bound:
                step = alpha * (best_ub - dual_bound) / sg_norm2
            else:
                step = alpha / np.sqrt(sg_norm2)

            lambda_t += step * sg_bal
            mu_t = np.maximum(0.0, mu_t + step * sg_res)

        if best_u is None:
            best_u = u_lr.copy()

        warmup_time = time.time() - t0
        ub_str = f"${best_ub:,.0f}" if best_ub < float("inf") else "N/A"
        print(f"  [LR-Warmup] Done: UB={ub_str}, {warmup_time:.1f}s")
        return best_u, best_ub

    def _init_multipliers(self, p, net_demand, reserve_arr):
        G, T = p.num_generators, p.num_steps
        mc_list = []
        for g in range(G):
            if p.use_piecewise and p.pw_slopes is not None:
                mc_g = p.pw_slopes[g][-1] if p.pw_slopes[g] else p.cost_b[g]
            else:
                mc_g = p.cost_a[g] * p.P_max[g] + p.cost_b[g]
            mc_list.append(mc_g)

        gen_order = sorted(range(G), key=lambda g: mc_list[g])
        peak_net = float(np.max(net_demand + reserve_arr))
        cumcap = 0.0
        clearing_price = mc_list[gen_order[0]]
        for g in gen_order:
            cumcap += p.P_max[g]
            clearing_price = mc_list[g]
            if cumcap >= peak_net:
                break

        net_min = float(np.min(net_demand))
        net_max = float(np.max(net_demand))
        if net_max > net_min:
            load_shape = (net_demand - net_min) / (net_max - net_min)
            lambda_t = clearing_price * (0.3 + 0.6 * load_shape)
        else:
            lambda_t = np.full(T, clearing_price * 0.7)

        mu_t = np.full(T, clearing_price * 0.05)
        return lambda_t, mu_t, clearing_price

    def _solve_subproblems(self, gen_models, lambda_t, mu_t, solver, p):
        G, T = p.num_generators, p.num_steps
        u_lr = np.zeros((G, T), dtype=int)
        P_lr = np.zeros((G, T))
        gen_obj_sum = 0.0
        n_fail = 0

        for g in range(G):
            gm = gen_models[g]
            for t in range(T):
                gm.lam[t] = float(lambda_t[t])
                gm.mu[t] = float(mu_t[t])
            logging.getLogger("pyomo.core").setLevel(logging.ERROR)
            try:
                res = solve_model(solver, gm, tee=False, load_solutions=True)
                status = str(res.solver.termination_condition).lower()
                if status in ("optimal", "feasible", "locallyoptimal", "maxtimelimit"):
                    gen_obj_sum += pyo.value(gm.Obj)
                    for t in range(T):
                        u_lr[g, t] = int(round(pyo.value(gm.u[t])))
                        P_lr[g, t] = pyo.value(gm.P[t])
                else:
                    n_fail += 1
            except Exception:
                n_fail += 1
            logging.getLogger("pyomo.core").setLevel(logging.WARNING)
        return u_lr, P_lr, gen_obj_sum, n_fail

    def _solve_storage(self, stor_model, lambda_t, solver, p):
        T = p.num_steps
        stor_net = np.zeros(T)
        stor_obj = 0.0
        for t in range(T):
            stor_model.lam[t] = float(lambda_t[t])
        try:
            solve_model(solver, stor_model, tee=False, load_solutions=True)
            stor_obj = pyo.value(stor_model.Obj)
            for t in range(T):
                stor_net[t] = pyo.value(stor_model.P_dis[t]) - pyo.value(stor_model.P_ch[t])
        except Exception:
            pass
        return stor_obj, stor_net

    def _build_gen_model(self, g, p):
        T = p.num_steps
        dt = p.dt
        Pmax_g = p.P_max[g]
        UT = p.min_up_time[g]
        DT = p.min_down_time[g]

        m = pyo.ConcreteModel(name=f"Gen_{g}")
        m.T = pyo.RangeSet(0, T - 1)
        m.lam = pyo.Param(m.T, initialize={t: 0.0 for t in range(T)}, mutable=True)
        m.mu = pyo.Param(m.T, initialize={t: 0.0 for t in range(T)}, mutable=True)
        m.u = pyo.Var(m.T, within=pyo.Binary)
        m.v = pyo.Var(m.T, within=pyo.Binary)
        m.w = pyo.Var(m.T, within=pyo.Binary)
        m.P = pyo.Var(m.T, within=pyo.NonNegativeReals)

        use_pw = p.use_piecewise and p.pw_slopes is not None
        if use_pw:
            S = p.num_segments
            m.SEG = pyo.RangeSet(0, S - 1)
            m.delta = pyo.Var(m.SEG, m.T, within=pyo.NonNegativeReals)

        m.StartupShutdown = pyo.Constraint(
            m.T, rule=lambda m, t: m.u[t] - (p.u0[g] if t == 0 else m.u[t - 1]) == m.v[t] - m.w[t]
        )
        m.Exclusive = pyo.Constraint(m.T, rule=lambda m, t: m.v[t] + m.w[t] <= 1)

        def min_up_rule(m, t):
            if UT <= 1 or t + UT > T:
                return pyo.Constraint.Skip
            return sum(m.u[tau] for tau in range(t, t + UT)) >= UT * m.v[t]

        m.MinUp = pyo.Constraint(m.T, rule=min_up_rule)

        def min_down_rule(m, t):
            if DT <= 1 or t + DT > T:
                return pyo.Constraint.Skip
            return sum(1 - m.u[tau] for tau in range(t, t + DT)) >= DT * m.w[t]

        m.MinDown = pyo.Constraint(m.T, rule=min_down_rule)

        status_h = p.initial_status_steps[g]
        if status_h > 0:
            for t in range(min(max(0, UT - status_h), T)):
                m.add_component(f"InitUp_{t}", pyo.Constraint(expr=m.u[t] == 1))
        elif status_h < 0:
            for t in range(min(max(0, DT - abs(status_h)), T)):
                m.add_component(f"InitDown_{t}", pyo.Constraint(expr=m.u[t] == 0))

        m.PLower = pyo.Constraint(m.T, rule=lambda m, t: m.P[t] >= p.P_min[g] * m.u[t])
        m.PUpper = pyo.Constraint(m.T, rule=lambda m, t: m.P[t] <= Pmax_g * m.u[t])

        def ramp_up_rule(m, t):
            prev_P = p.P0[g] if t == 0 else m.P[t - 1]
            prev_u = p.u0[g] if t == 0 else m.u[t - 1]
            return m.P[t] - prev_P <= p.ramp_up[g] * prev_u + p.startup_ramp[g] * m.v[t]

        m.RampUp = pyo.Constraint(m.T, rule=ramp_up_rule)

        def ramp_dn_rule(m, t):
            prev_P = p.P0[g] if t == 0 else m.P[t - 1]
            return prev_P - m.P[t] <= p.ramp_down[g] * m.u[t] + p.shutdown_ramp[g] * m.w[t]

        m.RampDown = pyo.Constraint(m.T, rule=ramp_dn_rule)

        if use_pw:
            S = p.num_segments
            m.SegWidth = pyo.Constraint(
                m.SEG, m.T, rule=lambda m, s, t: m.delta[s, t] <= p.pw_widths[g][s] * m.u[t]
            )
            m.SegSum = pyo.Constraint(
                m.T,
                rule=lambda m, t: (
                    m.P[t] == p.P_min[g] * m.u[t] + sum(m.delta[s, t] for s in range(S))
                ),
            )

        def obj_rule(m):
            if use_pw:
                fuel = sum(
                    (
                        p.cost_c[g] * m.u[t]
                        + sum(p.pw_slopes[g][s] * m.delta[s, t] for s in range(p.num_segments))
                    )
                    * dt
                    for t in m.T
                )
            else:
                fuel = sum(
                    (p.cost_a[g] * Pmax_g * m.P[t] + p.cost_b[g] * m.P[t] + p.cost_c[g] * m.u[t])
                    * dt
                    for t in m.T
                )
            startup = sum(p.cost_startup[g] * m.v[t] for t in m.T)
            shutdown = sum(p.cost_shutdown[g] * m.w[t] for t in m.T)
            lr_term = sum(-m.lam[t] * m.P[t] - m.mu[t] * Pmax_g * m.u[t] for t in m.T)
            return fuel + startup + shutdown + lr_term

        m.Obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        return m

    def _build_stor_model(self, p):
        T = p.num_steps
        dt = p.dt
        m = pyo.ConcreteModel(name="Storage")
        m.T = pyo.RangeSet(0, T - 1)
        m.lam = pyo.Param(m.T, initialize={t: 0.0 for t in range(T)}, mutable=True)
        m.P_ch = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.P_dis = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.SoC = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.u_ch = pyo.Var(m.T, within=pyo.Binary)
        m.u_dis = pyo.Var(m.T, within=pyo.Binary)
        m.SoCDyn = pyo.Constraint(
            m.T,
            rule=lambda m, t: (
                m.SoC[t]
                == (p.SoC0 if t == 0 else m.SoC[t - 1])
                + p.eta_ch * m.P_ch[t] * dt / p.E_max
                - m.P_dis[t] * dt / (p.eta_dis * p.E_max)
            ),
        )
        m.SoCLo = pyo.Constraint(m.T, rule=lambda m, t: m.SoC[t] >= p.SoC_min)
        m.SoCHi = pyo.Constraint(m.T, rule=lambda m, t: m.SoC[t] <= p.SoC_max)
        m.SoCEnd = pyo.Constraint(rule=lambda m: m.SoC[T - 1] == p.SoC0)
        m.ChLim = pyo.Constraint(m.T, rule=lambda m, t: m.P_ch[t] <= m.u_ch[t] * p.P_ch_max)
        m.DisLim = pyo.Constraint(m.T, rule=lambda m, t: m.P_dis[t] <= m.u_dis[t] * p.P_dis_max)
        m.ChDisExcl = pyo.Constraint(m.T, rule=lambda m, t: m.u_ch[t] + m.u_dis[t] <= 1)
        m.Obj = pyo.Objective(
            rule=lambda m: sum(-m.lam[t] * (m.P_dis[t] - m.P_ch[t]) for t in m.T),
            sense=pyo.minimize,
        )
        return m

    def _evaluate_fixed(self, u, model_obj, solver):
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
            return float("inf"), None
        sol = model_obj.extract_results()
        return sol["objective"], sol

    def _repair(self, u, p, demand_total, reserve):
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
                c1 = self._fix_min_on(u, g, p.min_up_time[g], T)
                c2 = self._fix_min_off(u, g, p.min_down_time[g], T)
                if not c1 and not c2:
                    break
        gen_order = sorted(range(G), key=lambda g: p.cost_b[g])
        touched = set()
        for t in range(T):
            cap = sum(p.P_max[g] for g in range(G) if u[g, t] == 1)
            need = demand_total[t] + reserve[t]
            if cap >= need:
                continue
            for g in gen_order:
                if u[g, t] == 1:
                    continue
                u[g, t] = 1
                touched.add(g)
                cap += p.P_max[g]
                if cap >= need:
                    break
        for g in touched:
            for _ in range(10):
                c1 = self._fix_min_on(u, g, p.min_up_time[g], T)
                c2 = self._fix_min_off(u, g, p.min_down_time[g], T)
                if not c1 and not c2:
                    break

    @staticmethod
    def _fix_min_on(u, g, min_up, T):
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

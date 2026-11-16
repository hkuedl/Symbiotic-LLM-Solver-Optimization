"""Uncertainty-aware LLM sampling and solver-guided refinement."""

import asyncio
import json
import math
import re
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import gurobipy as gp
import numpy as np
import pyomo.environ as pyo

from common.llm_client import create_client
from common.llm_is import postprocess, print_llm_usage, robust_json_parse, to_3d
from common.llm_requests import ChatRequests, logprob_content
from common.solver_config import (
    DEFAULT_TIME_LIMIT,
    configure_gurobi,
    create_solver,
    solve_model,
    validate_time_limit,
)
from models import IndustrialModel
from ours import IndustrialUncertaintyQuantifier as UncertaintyQuantifier


class OursSolver:
    """OursSolver: uncertainty-aware llm sampling and solver-guided refinement."""

    def __init__(
        self,
        api_key=None,
        base_url=None,
        model="gpt-4o",
        n_samples=5,
        strategy="epistemic_threshold",
        strategy_params=None,
        max_refine_iter=5,
        time_limit=DEFAULT_TIME_LIMIT,
    ):
        self.time_limit = validate_time_limit(time_limit)
        self.client = create_client(api_key, base_url, asynchronous=True)
        self.requests = ChatRequests(self.client)
        self._temp_dir = TemporaryDirectory(prefix="llm_solver_")
        self.model = model
        self.n_samples = n_samples
        self.strategy = strategy
        self.strategy_params = strategy_params or {"tau_I": 0.1}
        self.max_refine_iter = max_refine_iter
        self.solve_time = 0
        self.llm_time = 0
        self.token_usage = {"prompt": 0, "completion": 0}
        self.uq = None
        self.history = []
        self.iter_timings = []

    def solve(self, model_obj, price, pv_output, q_req):
        """Run the method and return the extracted solution, or None on failure."""
        m = model_obj.model
        p = model_obj.p
        if m is None:
            raise RuntimeError("Model not built yet.")

        total_start = time.time()
        total_vars = p.num_lines * p.num_processes * p.num_steps

        self.uq = UncertaintyQuantifier(p.num_lines, p.num_processes, p.num_steps)
        self.history = []
        self.iter_timings = []
        self.token_usage = {"prompt": 0, "completion": 0}

        print(f"[UQ] Phase 1: LLM Sampling ({self.n_samples} samples)...")
        t0 = time.time()
        self._mc_sampling(p, price, pv_output, q_req)
        self.llm_time = time.time() - t0

        if self.uq.N == 0:
            print("[UQ] No valid samples. Aborting.")
            return None

        print(f"[UQ] Sampling total: {self.llm_time:.1f}s, valid={self.uq.N}/{self.n_samples}")
        self.uq.summary()

        fixes = self.uq.select_variables_to_fix(strategy=self.strategy, **self.strategy_params)
        fixes = postprocess(fixes, p)
        fixed_keys = set(fixes.keys())
        free_keys = set()
        for l in range(p.num_lines):
            for k in range(p.num_processes):
                for t in range(p.num_steps):
                    if (l, k, t) not in fixed_keys:
                        free_keys.add((l, k, t))

        print(f"[UQ] fixed={len(fixed_keys)}/{total_vars}, free={len(free_keys)}")

        current_fixes = self._postprocess_maintenance(dict(fixes), fixed_keys, p)
        best_fixes = dict(current_fixes)
        best_obj = float("inf")
        best_solution = None
        last_solution = None
        prev_snapshot = None
        infeasible_count = 0

        messages = [
            {"role": "system", "content": self._build_system_prompt(p)},
            {"role": "user", "content": self._build_user_prompt(price, pv_output, q_req)},
        ]

        for iteration in range(1, self.max_refine_iter + 1):
            print(f"\n--- Iteration {iteration}/{self.max_refine_iter} ---")

            fixed_on = sum(1 for k in fixed_keys if current_fixes.get(k) == 1)
            fixed_off = sum(1 for k in fixed_keys if current_fixes.get(k) == 0)
            free_count = total_vars - len(fixed_keys)
            print(
                f"  Fixed: {len(fixed_keys)}/{total_vars} "
                f"(ON={fixed_on}, OFF={fixed_off}), "
                f"Free: {free_count} ({free_count / total_vars * 100:.1f}%)"
            )

            iter_timing = {"iter": iteration}
            solve_t0 = time.time()

            sch_obj, milp_m, results, status = self._solve_partial(
                current_fixes, fixed_keys, p, price, pv_output, q_req, warm_start_sol=last_solution
            )

            iter_timing["solve_time"] = time.time() - solve_t0
            iter_timing["status"] = status

            if status not in ("optimal", "feasible", "locallyoptimal", "infeasible"):
                self.history.append({"iter": iteration, "status": status})
                iter_timing["llm_refine_time"] = 0
                iter_timing["n_changed"] = 0
                self.iter_timings.append(iter_timing)
                break
            if status == "infeasible":
                self.history.append({"iter": iteration, "status": "infeasible"})
                infeasible_count += 1
                print(f"  INFEASIBLE (count={infeasible_count})")

                if infeasible_count >= 2 and best_fixes is not None:
                    print("  Reverting to best known feasible solution")
                    current_fixes = dict(best_fixes)
                    infeasible_count = 0
                    iter_timing["llm_refine_time"] = 0
                    iter_timing["n_changed"] = 0
                    self.iter_timings.append(iter_timing)
                    continue

                feedback = self._infeasibility_feedback(
                    milp_m, current_fixes, fixed_keys, p, price, pv_output, q_req, prev_snapshot
                )
                fb_type = "infeasible"
            else:
                infeasible_count = 0
                solution = self._extract_solution(sch_obj)
                last_solution = solution
                obj = solution["objective"]
                self.history.append({"iter": iteration, "status": "feasible", "obj": obj})
                iter_timing["objective"] = obj
                print(f"  obj=${obj:,.0f}, prod={solution['total_production']:,.0f}ton")

                if obj < best_obj:
                    best_obj = obj
                    best_solution = solution
                    best_fixes = dict(current_fixes)
                    print("  ★ New best!")
                elif obj > best_obj * 1.005:
                    print("  Worsened, reverting to best")
                    current_fixes = dict(best_fixes)
                    iter_timing["llm_refine_time"] = 0
                    iter_timing["n_changed"] = 0
                    self.iter_timings.append(iter_timing)
                    continue

                dual_info = self._extract_dual_info(
                    milp_m, current_fixes, fixed_keys, free_keys, p, price, pv_output, q_req
                )

                if dual_info is None:
                    iter_timing["llm_refine_time"] = 0
                    iter_timing["n_changed"] = 0
                    self.iter_timings.append(iter_timing)
                    break

                n_signals = self._count_improvement_signals(dual_info, current_fixes, fixed_keys)
                if n_signals == 0:
                    print("  Converged.")
                    iter_timing["llm_refine_time"] = 0
                    iter_timing["n_changed"] = 0
                    self.iter_timings.append(iter_timing)
                    break

                feedback = self._optimality_feedback(
                    dual_info, current_fixes, fixed_keys, p, price, solution, q_req
                )
                fb_type = "optimality"

            if iteration == self.max_refine_iter:
                iter_timing["llm_refine_time"] = 0
                iter_timing["n_changed"] = 0
                self.iter_timings.append(iter_timing)
                break

            if (
                len(self.history) >= 2
                and all(h["status"] == "feasible" for h in self.history[-2:])
                and self.history[-1]["obj"] >= self.history[-2]["obj"]
            ):
                print(" Converged (no improvement).")
                iter_timing["llm_refine_time"] = 0
                iter_timing["n_changed"] = 0
                self.iter_timings.append(iter_timing)
                break

            llm_t0 = time.time()
            n_changed, prev_snapshot = self._llm_refine(
                messages,
                feedback,
                fb_type,
                iteration,
                current_fixes,
                fixed_keys,
                p,
                price,
                pv_output,
                q_req,
            )
            iter_timing["llm_refine_time"] = time.time() - llm_t0
            iter_timing["n_changed"] = n_changed
            self.iter_timings.append(iter_timing)

            if n_changed == 0:
                break

            current_fixes = self._postprocess_maintenance(current_fixes, fixed_keys, p)

            fixed_on = sum(1 for k in fixed_keys if current_fixes.get(k) == 1)
            fixed_off = sum(1 for k in fixed_keys if current_fixes.get(k) == 0)
            free_count = total_vars - len(fixed_keys)
            print(
                f"  After refine: Fixed={len(fixed_keys)} "
                f"(ON={fixed_on}, OFF={fixed_off}), "
                f"Free={free_count} ({free_count / total_vars * 100:.1f}%)"
            )

        self.solve_time = time.time() - total_start
        print(f"\n[UQ] Done. Best=${best_obj:,.0f}, time={self.solve_time:.1f}s")
        self._print_history()
        return best_solution

    def _run_async(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            return asyncio.run(coro)

    def _mc_sampling(self, p, price, pv_output, q_req):
        """Collect samples with token probabilities when the endpoint supports them."""
        system_prompt = self._build_system_prompt(p)
        user_prompt = self._build_user_prompt(price, pv_output, q_req)

        async def _single_sample(idx):
            response = await self.requests.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            return idx, response, bool(logprob_content(response))

        async def _run_all():
            tasks = [_single_sample(i) for i in range(self.n_samples)]
            return await asyncio.gather(*tasks)

        t0 = time.time()
        results = self._run_async(_run_all())
        wall_time = time.time() - t0
        print(f"  All {self.n_samples} samples completed in {wall_time:.1f}s")

        target = (p.num_lines, p.num_processes, p.num_steps)
        total = p.num_lines * p.num_processes * p.num_steps

        for idx, response, has_lp in sorted(results, key=lambda x: x[0]):
            usage = response.usage
            self.token_usage["prompt"] += usage.prompt_tokens
            self.token_usage["completion"] += usage.completion_tokens

            raw = response.choices[0].message.content
            try:
                parsed = robust_json_parse(raw)
            except (ValueError, json.JSONDecodeError) as e:
                print(f"  Sample {idx + 1}: SKIPPED ({str(e)[:60]})")
                continue
            u_arr = to_3d(parsed.get("u", []), target, fill=0)
            u_list = u_arr.astype(int).tolist()

            logprob_probs = None
            if has_lp:
                logprob_probs = self._extract_logprob_per_variable(logprob_content(response), p)

            self.uq.add_sample(u_list, logprob_probs)

            on_count = int(np.sum(u_arr))
            lp_count = len(logprob_probs) if logprob_probs else 0
            print(
                f"  Sample {idx + 1}: ON={on_count}/{total}, "
                f"logprobs={lp_count}, tokens={usage.total_tokens}"
            )

    def _extract_logprob_per_variable(self, logprobs_content, p):
        if logprobs_content is None:
            return None

        K, T = p.num_processes, p.num_steps
        total_vars = p.num_lines * K * T

        full_text = ""
        token_positions = []
        for idx, item in enumerate(logprobs_content):
            start = len(full_text)
            full_text += item.token
            end = len(full_text)
            token_positions.append((start, end, idx))

        u_pattern = re.search(r'"u"\s*:\s*\[', full_text)
        if u_pattern is None:
            return None

        u_array_start = u_pattern.end() - 1
        depth = 0
        u_array_end = u_array_start
        for ci in range(u_array_start, len(full_text)):
            if full_text[ci] == "[":
                depth += 1
            elif full_text[ci] == "]":
                depth -= 1
                if depth == 0:
                    u_array_end = ci + 1
                    break

        digit_pattern = re.compile(r"(?<=[\[,\s])([01])(?=[,\]\s])")
        digit_matches = list(digit_pattern.finditer(full_text, u_array_start, u_array_end))

        if len(digit_matches) < total_vars * 0.5:
            return None

        probs = {}
        var_idx = 0
        for match in digit_matches:
            if var_idx >= total_vars:
                break
            char_pos = match.start(1)
            digit_str = match.group(1)

            tok_idx = None
            for start, end, tidx in token_positions:
                if start <= char_pos < end:
                    tok_idx = tidx
                    break
            if tok_idx is None:
                var_idx += 1
                continue

            item = logprobs_content[tok_idx]
            if digit_str == "1":
                p1 = math.exp(item.logprob)
                if item.top_logprobs:
                    for alt in item.top_logprobs:
                        if alt.token == item.token.replace("1", "0", 1):
                            p1 = p1 / (p1 + math.exp(alt.logprob))
                            break
            else:
                p1 = 0.0
                if item.top_logprobs:
                    p_self = math.exp(item.logprob)
                    for alt in item.top_logprobs:
                        if alt.token == item.token.replace("0", "1", 1):
                            p1_raw = math.exp(alt.logprob)
                            p1 = p1_raw / (p_self + p1_raw)
                            break

            l = var_idx // (K * T)
            k = (var_idx // T) % K
            t = var_idx % T
            probs[(l, k, t)] = float(np.clip(p1, 1e-10, 1 - 1e-10))
            var_idx += 1

        return probs if probs else None

    def _solve_partial(self, fixes, fixed_keys, p, price, pv_output, q_req, warm_start_sol=None):
        model_obj = IndustrialModel(p)
        m = model_obj.build(price, pv_output, q_req)
        for l, k, t in fixed_keys:
            m.u[l, k, t].fix(fixes[(l, k, t)])
        if warm_start_sol is not None:
            self._set_warm_start(m, warm_start_sol, p)
        solver = create_solver(self.time_limit)

        t0 = time.time()
        results = solve_model(solver, m, tee=False, warmstart=(warm_start_sol is not None))
        solve_time = time.time() - t0
        status = str(results.solver.termination_condition).lower()
        has_solution = len(results.solution) > 0
        if status not in ("optimal", "feasible", "locallyoptimal") and has_solution:
            status = "feasible"
        print(f"  [MILP] {status}, time={solve_time:.1f}s" + (" (warm)" if warm_start_sol else ""))
        model_obj.model = m
        return model_obj, m, results, status

    def _set_warm_start(self, m, sol, p):
        for l in range(p.num_lines):
            for k in range(p.num_processes):
                for t in range(p.num_steps):
                    m.u[l, k, t].set_value(round(sol["u"][l][k][t]))
                    m.mu[l, k, t].set_value(round(sol["mu"][l][k][t]))
                    m.nu[l, k, t].set_value(round(sol["nu"][l][k][t]))
        for t in range(p.num_steps):
            m.u_ch[t].set_value(round(sol["u_ch"][t]))
            m.u_dis[t].set_value(round(sol["u_dis"][t]))

    def _extract_solution(self, model_obj):
        sol = model_obj.extract_results()
        if hasattr(model_obj, "cost_breakdown"):
            sol["cost_breakdown"] = model_obj.cost_breakdown()
        return sol

    def _extract_dual_info(self, milp_m, fixes, fixed_keys, free_keys, p, price, pv_output, q_req):
        """Solve the original diagnostic relaxation and collect economic signals."""
        all_u = {}
        for l in range(p.num_lines):
            for k in range(p.num_processes):
                for t in range(p.num_steps):
                    all_u[(l, k, t)] = int(round(pyo.value(milp_m.u[l, k, t])))

        model_lp = IndustrialModel(p)
        m_lp = model_lp.build(price, pv_output, q_req)

        for l, k, t in free_keys:
            m_lp.u[l, k, t].fix(all_u[(l, k, t)])

        for l, k, t in fixed_keys:
            m_lp.u[l, k, t].domain = pyo.NonNegativeReals
            m_lp.u[l, k, t].setub(1.0)
            m_lp.u[l, k, t].setlb(0.0)
            m_lp.u[l, k, t].value = fixes[(l, k, t)]

        for l in range(p.num_lines):
            for k in range(p.num_processes):
                for t in range(p.num_steps):
                    m_lp.mu[l, k, t].domain = pyo.NonNegativeReals
                    m_lp.mu[l, k, t].setub(1.0)
                    m_lp.nu[l, k, t].domain = pyo.NonNegativeReals
                    m_lp.nu[l, k, t].setub(1.0)

        for t in range(p.num_steps):
            m_lp.u_ch[t].domain = pyo.NonNegativeReals
            m_lp.u_ch[t].setub(1.0)
            m_lp.u_dis[t].domain = pyo.NonNegativeReals
            m_lp.u_dis[t].setub(1.0)

        m_lp.write(
            str(Path(self._temp_dir.name) / "_dual_lp.lp"),
            io_options={"symbolic_solver_labels": True},
        )

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            gm = gp.read(str(Path(self._temp_dir.name) / "_dual_lp.lp"), env=env)

            for v in gm.getVars():
                if v.VType != "C":
                    v.VType = "C"
            gm.update()
            configure_gurobi(gm, self.time_limit)
            gm.optimize()

            if gm.status != gp.GRB.OPTIMAL:
                return None

            lp_obj = gm.objVal

            info = {
                "lp_obj": lp_obj,
                "reduced_costs": {},
                "lp_values": {},
                "grid_shadow": {},
                "production_shadow": 0,
                "line_output_shadow": {},
            }

            for v in gm.getVars():
                if not (v.VarName.startswith("u(") or v.VarName.startswith("u[")):
                    continue
                nums = re.findall(r"\d+", v.VarName)
                if len(nums) < 3:
                    continue
                l, k, t = int(nums[0]), int(nums[1]), int(nums[2])
                if (l, k, t) in fixed_keys:
                    info["reduced_costs"][(l, k, t)] = v.RC
                    info["lp_values"][(l, k, t)] = v.X

            for c in gm.getConstrs():
                name, pi = c.ConstrName, c.Pi
                if abs(pi) < 1e-6:
                    continue
                if "GridLimit" in name:
                    nums = re.findall(r"\d+", name)
                    if nums:
                        info["grid_shadow"][int(nums[-1])] = pi
                elif "TotalProduction" in name:
                    info["production_shadow"] = pi
                elif "LineMinOutput" in name:
                    nums = re.findall(r"\d+", name)
                    if nums:
                        info["line_output_shadow"][int(nums[-1])] = pi

        return info

    def _count_improvement_signals(self, dual_info, fixes, fixed_keys):
        rc = dual_info["reduced_costs"]
        lp_vals = dual_info["lp_values"]
        n_frac = sum(1 for k in fixed_keys if abs(lp_vals.get(k, fixes[k]) - fixes[k]) > 0.05)
        n_rc = sum(1 for v in rc.values() if abs(v) > 0.5)
        return n_frac + n_rc

    def _infeasibility_feedback(
        self, m, fixes, fixed_keys, p, price, pv_output, q_req, prev_snapshot=None
    ):
        fb = ["INFEASIBLE. Revert or adjust changes to restore feasibility."]

        if prev_snapshot is not None:
            diff = []
            for key in fixed_keys:
                if fixes[key] != prev_snapshot.get(key, fixes[key]):
                    diff.append((key, prev_snapshot[key], fixes[key]))
            if diff:
                fb.append(f"\nLast changes ({len(diff)} vars) — revert some:")
                for (l, k, t), old, new in diff[:15]:
                    fb.append(f"  L{l}_P{k}_T{t}: {old} → {new} (revert to {old})")

        last_k = p.num_processes - 1
        total_max = 0
        for l in range(p.num_lines):
            fixed_off = sum(
                1
                for t in range(p.num_steps)
                if (l, last_k, t) in fixed_keys and fixes[(l, last_k, t)] == 0
            )
            max_on = p.num_steps - fixed_off
            total_max += max_on * p.dt * p.production_rate_max[l][last_k]

        if total_max < q_req:
            fb.append(f"\nProduction capacity {total_max:.0f} < required {q_req:.0f}.")
            off_candidates = sorted(
                [
                    (price[t], l, t)
                    for l in range(p.num_lines)
                    for t in range(p.num_steps)
                    if (l, last_k, t) in fixed_keys
                    and fixes[(l, last_k, t)] == 0
                    and not any(l == lid and ts <= t <= te for lid, ts, te in p.maintenance_windows)
                ],
                key=lambda x: x[0],
            )
            fb.append(f"Turn ON these P{last_k} (cheapest first):")
            for pr, l, t in off_candidates[:10]:
                fb.append(f"  L{l}_P{last_k}_T{t} → set 1 (price=${pr:.0f})")

        for t in range(p.num_steps):
            fp = sum(
                p.power_per_ton[l][k] * p.production_rate_min[l][k]
                for l in range(p.num_lines)
                for k in range(p.num_processes)
                if (l, k, t) in fixed_keys and fixes[(l, k, t)] == 1
            )
            avail = p.P_grid_max + pv_output[t] + p.P_dis_max
            if fp > avail:
                fb.append(f"\nPower overload step {t}: need {fp:.1f}MW > available {avail:.1f}MW.")
                on_at_t = [
                    (p.power_per_ton[l][k] * p.production_rate_min[l][k], l, k)
                    for l in range(p.num_lines)
                    for k in range(p.num_processes)
                    if (l, k, t) in fixed_keys and fixes[(l, k, t)] == 1
                ]
                on_at_t.sort(reverse=True)
                fb.append("Turn OFF one of (highest power first):")
                for pw, l, k in on_at_t[:5]:
                    fb.append(f"  L{l}_P{k}_T{t} → set 0 (power={pw:.1f}MW)")
                break

        for l in range(p.num_lines):
            for s in range(p.num_buffers):
                up_k, dn_k = s, s + 1
                t_on_up = next(
                    (
                        t
                        for t in range(p.num_steps)
                        if (l, up_k, t) in fixed_keys and fixes[(l, up_k, t)] == 1
                    ),
                    None,
                )
                t_on_dn = next(
                    (
                        t
                        for t in range(p.num_steps)
                        if (l, dn_k, t) in fixed_keys and fixes[(l, dn_k, t)] == 1
                    ),
                    None,
                )
                if t_on_up is not None and t_on_dn is not None:
                    gap = t_on_dn - t_on_up
                    min_lead = max(
                        1,
                        int(
                            np.ceil(
                                (p.production_rate_min[l][dn_k] / p.eta[l][dn_k])
                                / (p.production_rate_max[l][up_k] * p.eta[l][up_k])
                            )
                        ),
                    )
                    if gap < min_lead:
                        fb.append(
                            f"\nBUFFER: L{l} P{dn_k} starts step {t_on_dn}, "
                            f"P{up_k} starts step {t_on_up}. "
                            f"Gap={gap} < required={min_lead}. "
                            f"Delay P{dn_k} or release steps {t_on_dn}-{t_on_up + min_lead}."
                        )

        fb.append("\nPrioritize reverting last changes or releasing uncertain variables.")

        return "\n".join(fb)

    def _optimality_feedback(self, dual_info, fixes, fixed_keys, p, price, solution, q_req):
        fb = []

        cb = solution.get("cost_breakdown", {})
        if cb:
            fb.append(
                f"Objective={solution['objective']:.0f} "
                f"(power={cb.get('power_cost', 0):.0f}, "
                f"SU={cb.get('startup_cost', 0):.0f}, "
                f"SD={cb.get('shutdown_cost', 0):.0f}, "
                f"transfer={cb.get('transfer_cost', 0):.0f})"
            )
        else:
            fb.append(f"Objective: ${solution['objective']:,.0f}")

        fb.append(f"Production: {solution['total_production']:.0f}/{q_req:.0f} ton")
        fb.append(f"Lines: {[f'{y:.0f}' for y in solution['Y']]}")
        fb.append(
            f"LP lower bound: {dual_info['lp_obj']:.0f} "
            f"(improvement room: {solution['objective'] - dual_info['lp_obj']:.0f})"
        )

        gs = dual_info["grid_shadow"]
        if gs:
            fb.append(f"\nGrid BINDING at {len(gs)} steps: {sorted(gs.keys())[:15]}")

        ps = dual_info["production_shadow"]
        if abs(ps) > 0.01:
            fb.append(
                f"Production shadow: {ps:.2f} $/ton"
                f" {'(tight, do NOT reduce ON)' if ps < -0.01 else ''}"
            )
        else:
            fb.append("Production NOT binding (can reduce ON to save cost).")

        for l_idx, pi in dual_info["line_output_shadow"].items():
            if abs(pi) > 0.01:
                fb.append(f"  Line {l_idx} min output binding (shadow={pi:.2f})")

        rc = dual_info["reduced_costs"]
        lp_vals = dual_info["lp_values"]

        fractional = []
        for l, k, t in fixed_keys:
            fix_val = fixes[(l, k, t)]
            lp_val = lp_vals.get((l, k, t), fix_val)
            rc_val = rc.get((l, k, t), 0)
            deviation = abs(lp_val - fix_val)
            if deviation > 0.05:
                power_cost_t = p.power_per_ton[l][k] * p.production_rate_min[l][k] * p.dt * price[t]
                fractional.append(
                    (deviation, rc_val, l, k, t, fix_val, lp_val, price[t], power_cost_t)
                )
        fractional.sort(reverse=True)

        if fractional:
            should_off = [
                (d, r, l, k, t, pr, pc)
                for d, r, l, k, t, fv, lv, pr, pc in fractional
                if fv == 1 and lv < 0.5
            ]
            should_on = [
                (d, r, l, k, t, pr, pc)
                for d, r, l, k, t, fv, lv, pr, pc in fractional
                if fv == 0 and lv > 0.5
            ]

            if should_off:
                blocks = self._aggregate_blocks(should_off)
                total_saving = sum(b["est_saving"] for b in blocks[:10])
                fb.append(
                    f"\nSET to OFF ({len(should_off)} vars, est. saving ~${total_saving:,.0f}):"
                )
                for b in blocks[:10]:
                    fb.append(
                        f"  L{b['l']}_P{b['k']} steps {b['t_start']}-{b['t_end']}: "
                        f"save ~${b['est_saving']:,.0f}, avg_price=${b['avg_price']:.0f} → set 0"
                    )

            if should_on:
                blocks = self._aggregate_blocks(should_on)
                fb.append(f"\nSET to ON ({len(should_on)} vars):")
                for b in blocks[:10]:
                    fb.append(
                        f"  L{b['l']}_P{b['k']} steps {b['t_start']}-{b['t_end']}: "
                        f"cost ~${b['est_saving']:,.0f}, avg_price=${b['avg_price']:.0f} → set 1"
                    )

            release_candidates = []
            for l, k, t in fixed_keys:
                fv = fixes[(l, k, t)]
                lv = lp_vals.get((l, k, t), fv)
                rc_val = rc.get((l, k, t), 0)
                deviation = abs(lv - fv)
                if 0.01 < deviation <= 0.05 or (deviation <= 0.05 and abs(rc_val) > 1.0):
                    release_candidates.append(
                        {
                            "l": l,
                            "k": k,
                            "t": t,
                            "rc": rc_val,
                            "price": price[t],
                        }
                    )
            release_candidates.sort(key=lambda x: -abs(x["rc"]))

            if release_candidates:
                release_blocks = self._aggregate_release_blocks(release_candidates)
                fb.append(
                    f"\nRELEASE (let solver decide, {len(release_candidates)} borderline vars):"
                )
                for b in release_blocks[:10]:
                    fb.append(f"  L{b['l']}_P{b['k']} steps {b['t_start']}-{b['t_end']} → release")

        else:
            should_off_rc = sorted(
                [
                    (rc.get(k, 0), k[0], k[1], k[2], price[k[2]])
                    for k in fixed_keys
                    if fixes[k] == 1 and rc.get(k, 0) > 0.5
                ],
                reverse=True,
            )
            should_on_rc = sorted(
                [
                    (abs(rc.get(k, 0)), k[0], k[1], k[2], price[k[2]])
                    for k in fixed_keys
                    if fixes[k] == 0 and rc.get(k, 0) < -0.5
                ],
                reverse=True,
            )
            if should_off_rc:
                fb.append("\nBy RC — SET to OFF:")
                for rv, l, k, t, pr in should_off_rc[:8]:
                    fb.append(f"  L{l}_P{k}_T{t}: RC={rv:.1f}, price={pr:.0f} → set 0")
            if should_on_rc:
                fb.append("\nBy RC — SET to ON:")
                for rv, l, k, t, pr in should_on_rc[:8]:
                    fb.append(f"  L{l}_P{k}_T{t}: RC={rv:.1f}, price={pr:.0f} → set 1")

            mid_rc = sorted(
                [
                    (abs(rc.get(k, 0)), k[0], k[1], k[2])
                    for k in fixed_keys
                    if 0.1 < abs(rc.get(k, 0)) <= 0.5
                ],
                reverse=True,
            )
            if mid_rc:
                fb.append(f"\nRELEASE (mid-range RC, {len(mid_rc)} vars):")
                for rv, l, k, t in mid_rc[:10]:
                    fb.append(f"  L{l}_P{k}_T{t} → release")

            if not should_off_rc and not should_on_rc and not mid_rc:
                fb.append("\nNo improvement signals found.")

        q75 = float(np.percentile(price, 75))
        exp_on = [
            (price[t], l, k, t)
            for (l, k, t) in fixed_keys
            if fixes[(l, k, t)] == 1 and price[t] > q75
        ]
        exp_on.sort(reverse=True)
        if exp_on:
            fb.append(f"\nFixed ON in expensive periods (>{q75:.0f}$/MWh): {len(exp_on)} vars")
            for pr, l, k, t in exp_on[:5]:
                fb.append(f"  L{l}_P{k}_T{t} price={pr:.0f}")

        structural = self._structural_feedback(solution, p, price, q_req)
        if structural:
            fb.append(f"\n{structural}")

        prod_binding = abs(ps) > 0.01
        if prod_binding:
            fb.append("\nWARNING: Production is tight. Do not turn off final process.")

        fb.append(
            f"\nProduction must >= {q_req:.0f} ton, each line >= {min(p.q_req_line_min):.0f}."
        )

        return "\n".join(fb)

    def _structural_feedback(self, solution, p, price, q_req):
        lines = []
        for l in range(p.num_lines):
            if solution["Y"][l] < 1:
                lines.append(
                    {
                        "line": l,
                        "output": 0,
                        "cost_per_ton": float("inf"),
                        "avg_price_on": 0,
                        "on_steps": 0,
                        "chain_eta": 0,
                    }
                )
                continue

            last_k = p.num_processes - 1
            on_steps = sum(1 for t in range(p.num_steps) if solution["u"][l][last_k][t] > 0.5)

            if on_steps == 0:
                continue

            prices_on = [price[t] for t in range(p.num_steps) if solution["u"][l][last_k][t] > 0.5]
            avg_price_on = sum(prices_on) / len(prices_on)

            chain_eta = 1.0
            for k in range(p.num_processes):
                chain_eta *= p.eta[l][k]

            total_power = sum(
                p.power_per_ton[l][k] * solution["prod_out"][l][k][t] * p.dt
                for k in range(p.num_processes)
                for t in range(p.num_steps)
            )
            cost_per_ton = (
                total_power * avg_price_on / solution["Y"][l]
                if solution["Y"][l] > 0
                else float("inf")
            )

            lines.append(
                {
                    "line": l,
                    "output": solution["Y"][l],
                    "cost_per_ton": cost_per_ton,
                    "avg_price_on": avg_price_on,
                    "on_steps": on_steps,
                    "chain_eta": chain_eta,
                }
            )

        lines.sort(key=lambda x: x["cost_per_ton"])

        fb = ["LINE ANALYSIS (cheapest first):"]
        for info in lines:
            if info["on_steps"] == 0:
                continue
            fb.append(
                f"  Line {info['line']}: "
                f"output={info['output']:.0f}ton, "
                f"~${info['cost_per_ton']:.1f}/ton, "
                f"avg_price=${info['avg_price_on']:.0f}, "
                f"eta={info['chain_eta']:.3f}, "
                f"on={info['on_steps']}steps"
            )

        if len(lines) >= 2:
            cheapest = lines[0]
            costliest = lines[-1]
            if (
                costliest["cost_per_ton"] > cheapest["cost_per_ton"] * 1.15
                and costliest["output"] > 0
            ):
                fb.append(
                    f"\n  SUGGESTION: Line {costliest['line']} is {costliest['cost_per_ton'] / cheapest['cost_per_ton']:.0%} "
                    f"more expensive than Line {cheapest['line']}. "
                    f"Consider shifting production from Line {costliest['line']} to Line {cheapest['line']}."
                )

            surplus = solution["total_production"] - q_req
            if surplus > 100:
                fb.append(
                    f"\n  SUGGESTION: Over-producing by {surplus:.0f} ton. "
                    f"Reduce Line {costliest['line']} output to save cost."
                )

        return "\n".join(fb)

    def _aggregate_blocks(self, items):
        by_grp = {}
        for dev, rc_v, l, k, t, pr, pc in items:
            by_grp.setdefault((l, k), []).append((t, dev, rc_v, pr, pc))
        blocks = []
        for (l, k), entries in by_grp.items():
            entries.sort()
            i = 0
            while i < len(entries):
                t_s = entries[i][0]
                total_dev = entries[i][1]
                total_price = entries[i][3]
                total_pc = entries[i][4]
                j = i + 1
                while j < len(entries) and entries[j][0] == entries[j - 1][0] + 1:
                    total_dev += entries[j][1]
                    total_price += entries[j][3]
                    total_pc += entries[j][4]
                    j += 1
                n_steps = j - i
                blocks.append(
                    {
                        "l": l,
                        "k": k,
                        "t_start": t_s,
                        "t_end": entries[j - 1][0],
                        "n_steps": n_steps,
                        "total_dev": total_dev,
                        "avg_price": total_price / n_steps,
                        "est_saving": total_pc,
                    }
                )
                i = j
        blocks.sort(key=lambda x: -x["est_saving"])
        return blocks

    def _aggregate_release_blocks(self, items):
        by_grp = {}
        for c in items:
            by_grp.setdefault((c["l"], c["k"]), []).append(c["t"])
        blocks = []
        for (l, k), times in by_grp.items():
            times.sort()
            i = 0
            while i < len(times):
                t_s = times[i]
                j = i + 1
                while j < len(times) and times[j] == times[j - 1] + 1:
                    j += 1
                blocks.append(
                    {
                        "l": l,
                        "k": k,
                        "t_start": t_s,
                        "t_end": times[j - 1],
                        "n_steps": j - i,
                    }
                )
                i = j
        blocks.sort(key=lambda x: -x["n_steps"])
        return blocks

    def _llm_refine(
        self,
        messages,
        feedback,
        fb_type,
        iteration,
        current_fixes,
        fixed_keys,
        p,
        price,
        pv_output,
        q_req,
    ):
        action = "fix infeasibility" if fb_type == "infeasible" else "reduce cost"
        prompt_msg = (
            f"=== SOLVER FEEDBACK (Iter {iteration}, {fb_type}) ===\n\n"
            f"{feedback}\n\n"
            f"Change some fixed variables to {action}.\n"
            f'Output: JSON with "changes" list. Each entry:\n'
            f'  {{"l": int, "k": int, "t_start": int, "t_end": int, '
            f'"action": "set"|"release", "value": 0 or 1}}\n'
            f'  "set": fix the variable to value 0 or 1.\n'
            f'  "release": remove from fixed set, let solver decide optimally. '
            f"value is ignored for release.\n"
            f"Use release for uncertain variables. Empty list if none.\n"
            f"Output ONLY the JSON object, no other text."
        )
        messages.append({"role": "user", "content": prompt_msg})

        async def _call():
            return await self.requests.acreate(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
            )

        t0 = time.time()
        response = self._run_async(_call())
        llm_time = time.time() - t0

        usage = response.usage
        self.token_usage["prompt"] += usage.prompt_tokens
        self.token_usage["completion"] += usage.completion_tokens
        print_llm_usage(usage, llm_time)

        raw_resp = response.choices[0].message.content
        messages.append({"role": "assistant", "content": raw_resp})

        try:
            parsed = robust_json_parse(raw_resp)
            if isinstance(parsed, list):
                changes = parsed
            elif isinstance(parsed, dict):
                changes = parsed.get("changes", [])
            else:
                changes = []
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  LLM refine parse failed: {str(e)[:60]}")
            changes = []
        changes = parsed.get("changes", [])

        changes = self._validate_changes(
            changes, current_fixes, fixed_keys, p, price, pv_output, q_req
        )

        snapshot = dict(current_fixes)
        n_changed = 0
        n_released = 0

        for c in changes:
            l, k = c.get("l", -1), c.get("k", -1)
            t_start = max(0, c.get("t_start", -1))
            t_end = min(c.get("t_end", t_start), p.num_steps - 1)
            action_type = c.get("action", "set")
            value = int(c.get("value", 0))

            for t in range(t_start, t_end + 1):
                if (l, k, t) not in fixed_keys:
                    continue
                if action_type == "release":
                    fixed_keys.discard((l, k, t))
                    current_fixes.pop((l, k, t), None)
                    n_released += 1
                elif current_fixes.get((l, k, t)) != value:
                    current_fixes[(l, k, t)] = value
                    n_changed += 1

        print(f"  Applied {n_changed} changes, released {n_released} vars")
        return n_changed + n_released, snapshot

    def _validate_changes(self, changes, current_fixes, fixed_keys, p, price, pv_output, q_req):
        """Validate proposed changes before updating fixed assignments."""
        valid = []
        for c in changes:
            l, k = c.get("l", -1), c.get("k", -1)
            t_start = max(0, c.get("t_start", -1))
            t_end = min(c.get("t_end", t_start), p.num_steps - 1)
            value = int(c.get("value", 0))
            action_type = c.get("action", "set")

            if action_type == "release":
                valid.append(c)
                continue

            reject = False

            if value == 1:
                for lid, ts, te in p.maintenance_windows:
                    if l == lid and any(ts <= t <= te for t in range(t_start, t_end + 1)):
                        reject = True
                        break

            if not reject and value == 1:
                for t in range(t_start, t_end + 1):
                    forced_power = sum(
                        p.power_per_ton[ll][kk] * p.production_rate_min[ll][kk]
                        for ll in range(p.num_lines)
                        for kk in range(p.num_processes)
                        if ((ll, kk, t) in fixed_keys and current_fixes.get((ll, kk, t)) == 1)
                        or (ll == l and kk == k)
                    )
                    avail = p.P_grid_max + pv_output[t] + p.P_dis_max
                    if forced_power > avail * 0.95:
                        reject = True
                        break

            if not reject and value == 0 and k == p.num_processes - 1:
                last_k = p.num_processes - 1
                temp_fixes = dict(current_fixes)
                for t in range(t_start, t_end + 1):
                    temp_fixes[(l, last_k, t)] = 0
                total_max = 0
                for ll in range(p.num_lines):
                    on_steps = sum(
                        1 for t in range(p.num_steps) if temp_fixes.get((ll, last_k, t), 1) == 1
                    )
                    total_max += on_steps * p.dt * p.production_rate_max[ll][last_k]
                if total_max < q_req * 1.05:
                    reject = True

            if not reject:
                valid.append(c)
            else:
                print(f"Rejected: L{l}_P{k}_T{t_start}-{t_end}={value}")

        return valid

    def _postprocess_maintenance(self, fixes, fixed_keys, p):
        fixes = dict(fixes)
        for lid, ts, te in p.maintenance_windows:
            for k in range(p.num_processes):
                for t in range(ts, min(te + 1, p.num_steps)):
                    if (lid, k, t) in fixed_keys:
                        fixes[(lid, k, t)] = 0
        return fixes

    def _print_history(self):
        print(f"\n  {'Iter':>5s} {'Status':>12s} {'Objective':>12s}")
        print(f"  {'-' * 32}")
        for h in self.history:
            if h["status"] == "feasible":
                print(f"  {h['iter']:>5d} {'feasible':>12s} {h['obj']:>12,.0f}")
            else:
                print(f"  {h['iter']:>5d} {'INFEASIBLE':>12s} {'---':>12s}")

    def _build_system_prompt(self, p):
        is_claude = "claude" in self.model.lower()

        if is_claude:
            output_instruction = (
                f"Output: JSON with only one key:\n"
                f'- "u": a 3D array [{p.num_lines}][{p.num_processes}][{p.num_steps}], each value 0 or 1.\n'
                f"  u[l][k][t] = 1 means line l, process k is ON at step t; 0 means OFF.\n"
                f"You MUST output ONLY a valid JSON object, nothing else. "
                f"Do NOT ask questions. Do NOT explain. Do NOT refuse. "
                f"Use your best judgment for any ambiguity. "
                f"Your entire response must be parseable by json.loads()."
            )
        else:
            output_instruction = (
                f"Output: JSON with only one key:\n"
                f'- "u": a 3D array [{p.num_lines}][{p.num_processes}][{p.num_steps}], each value 0 or 1.\n'
                f"  u[l][k][t] = 1 means line l, process k is ON at step t; 0 means OFF.\n"
                f"Output ONLY the JSON object. No other text."
            )

        return f"""Task: You are a pre-decision advisor for a multi-line industrial scheduling system. Suggest on/off decisions for production processes to minimize electricity cost while meeting production targets. The MILP solver will optimally determine continuous variables.

System:
{p.num_lines} production lines, each with {p.num_processes} sequential processes.
Time horizon: {p.num_steps} steps, each {p.dt} hours. Index starts from 0.
Process chain: Process 0 → Buffer 0 → Process 1 → ... → Process {p.num_processes - 1} → final output.

Objective: minimize electricity cost + startup/shutdown costs.

Parameters [line][process]:
- max production rate (ton/h): {json.dumps(p.production_rate_max)}
- min production rate (ton/h): {json.dumps(p.production_rate_min)}
- efficiency: {json.dumps(p.eta)}
- power per ton (MW/ton): {json.dumps(p.power_per_ton)}
- min consecutive on steps: {json.dumps(p.on_time)}
- startup cost ($): {json.dumps(p.cost_startup)}
- shutdown cost ($): {json.dumps(p.cost_shutdown)}
- min production per line: {min(p.q_req_line_min)} ton
- maintenance windows: {json.dumps(p.maintenance_windows)}

Rules:
1. Each process must run at least minimum consecutive on steps once turned on.
2. Each process must be off during maintenance windows.
3. Process k+1 draws material from Buffer k; if buffer is empty it cannot produce.

Guidelines:
1. Prefer off during high-price periods.
2. Prefer on during low-price and high PV periods.

{output_instruction}"""

    def _build_user_prompt(self, price, pv_output, q_req):
        return f"""Price ($/MWh) [{len(price)} steps]:
{json.dumps([round(v, 2) for v in price])}

PV (MW) [{len(pv_output)} steps]:
{json.dumps([round(v, 2) for v in pv_output])}

Total production requirement: {q_req} ton."""

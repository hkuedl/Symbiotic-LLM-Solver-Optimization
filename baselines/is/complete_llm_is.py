"""Reference optimization method."""

import json
import time

from common.llm_client import create_client
from common.llm_is import fix_variables, postprocess, print_llm_usage, robust_json_parse, to_3d
from common.llm_requests import ChatRequests
from common.solver_config import DEFAULT_TIME_LIMIT, create_solver, solve_model, validate_time_limit


class LLMSolver:
    """LLMSolver: reference optimization method."""

    def __init__(self, api_key=None, base_url=None, model="gpt-4o", time_limit=DEFAULT_TIME_LIMIT):
        self.time_limit = validate_time_limit(time_limit)
        self.client = create_client(api_key, base_url)
        self.requests = ChatRequests(self.client)
        self.model = model
        self.solve_time = 0
        self.llm_time = 0
        self.token_usage = None

    def solve(self, model_obj, price, pv_output, q_req):
        """Run the method and return the extracted solution, or None on failure."""
        m = model_obj.model
        p = model_obj.p
        if m is None:
            raise RuntimeError("Model not built yet.")
        total_start = time.time()
        print(f"[LLM] Calling {self.model}...")
        t0 = time.time()
        u_arr = self._call_llm(p, price, pv_output, q_req)
        self.llm_time = time.time() - t0
        print_llm_usage(self.token_usage, self.llm_time)
        if u_arr is None:
            return None
        fixes = {}
        for l in range(p.num_lines):
            for k in range(p.num_processes):
                for t in range(p.num_steps):
                    fixes[(l, k, t)] = int(u_arr[l, k, t])
        fixes = postprocess(fixes, p)
        fix_variables(m, fixes, p)
        total = p.num_lines * p.num_processes * p.num_steps
        on_count = sum(fixes.values())
        print(f"[LLM] Fixed all {total} vars, ON={on_count}")
        solver = create_solver(self.time_limit)
        result = solve_model(solver, m, tee=False)
        tc = str(result.solver.termination_condition).lower()
        self.solve_time = time.time() - total_start
        if tc in ("optimal", "feasible", "locallyoptimal"):
            sol = model_obj.extract_results()
            print(f"[LLM] obj=${sol['objective']:,.0f}, time={self.solve_time:.1f}s")
            return sol
        else:
            print(f"[LLM] Infeasible: {tc}")
            return None

    def _call_llm(self, p, price, pv_output, q_req):
        system_prompt = self._build_system_prompt(p)
        user_prompt = self._build_user_prompt(price, pv_output, q_req)
        try:
            response = self.requests.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            print(f"  [LLM] API error: {e}")
            return None
        self.token_usage = response.usage
        raw = response.choices[0].message.content
        target = (p.num_lines, p.num_processes, p.num_steps)
        try:
            data = robust_json_parse(raw)
        except ValueError as e:
            print(f"  [LLM] Parse failed: {e}")
            return None
        if "u" not in data:
            return None
        return to_3d(data["u"], target, fill=0)

    def _build_system_prompt(self, p):
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

Output: JSON with only one key:
- "u": a 3D array [{p.num_lines}][{p.num_processes}][{p.num_steps}], each value 0 or 1.
  u[l][k][t] = 1 means line l, process k is ON at step t; 0 means OFF.
No other text."""

    def _build_user_prompt(self, price, pv_output, q_req):
        return f"""Price ($/MWh) [{len(price)} steps]:
{json.dumps([round(v, 2) for v in price])}

PV (MW) [{len(pv_output)} steps]:
{json.dumps([round(v, 2) for v in pv_output])}

Total production requirement: {q_req} ton."""

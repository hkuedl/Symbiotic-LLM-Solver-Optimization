"""Shared parsing and schedule postprocessing utilities."""

import itertools
import json
import re

import numpy as np


def robust_json_parse(raw):
    """Recover a JSON object from complete or truncated LLM output."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    if start != -1:
        partial = cleaned[start:]
        partial = re.sub(r"[,\s]+$", "", partial)
        partial = re.sub(r",\s*$", "", partial)
        open_sq = partial.count("[") - partial.count("]")
        open_cr = partial.count("{") - partial.count("}")
        if open_sq > 0 or open_cr > 0:
            try:
                return json.loads(partial + "]" * max(0, open_sq) + "}" * max(0, open_cr))
            except json.JSONDecodeError:
                pass

    cleaned_tail = re.sub(r"[,\s]+$", "", cleaned)
    open_sq = cleaned_tail.count("[") - cleaned_tail.count("]")
    open_cr = cleaned_tail.count("{") - cleaned_tail.count("}")
    suffix = "]" * max(0, open_sq) + "}" * max(0, open_cr)
    try:
        return json.loads(cleaned_tail + suffix)
    except json.JSONDecodeError:
        pass

    u_match = re.search(r'"u"\s*:\s*(\[[\s\S]*)', cleaned)
    if u_match:
        u_str = u_match.group(1).strip()
        u_str = re.sub(r"[,\s]+$", "", u_str)
        open_sq = u_str.count("[") - u_str.count("]")
        u_str += "]" * max(0, open_sq)
        try:
            return {"u": json.loads(u_str)}
        except json.JSONDecodeError:
            last_valid = len(u_str)
            for cut in range(len(u_str) - 1, 0, -1):
                if u_str[cut] in "],":
                    candidate = u_str[: cut + 1]
                    candidate = re.sub(r",\s*$", "", candidate)
                    o = candidate.count("[") - candidate.count("]")
                    candidate += "]" * max(0, o)
                    try:
                        return {"u": json.loads(candidate)}
                    except json.JSONDecodeError:
                        continue

    raise ValueError(f"Cannot parse. First 300 chars: {raw[:300]}")


def _ragged_to_3d(raw_list, target, fill):
    L, K, T = target
    result = np.full(target, fill, dtype=float)
    if not isinstance(raw_list, list):
        return result
    for l in range(min(len(raw_list), L)):
        if not isinstance(raw_list[l], list):
            continue
        for k in range(min(len(raw_list[l]), K)):
            if not isinstance(raw_list[l][k], list):
                if isinstance(raw_list[l][k], (int, float)):
                    result[l, k, :] = int(round(raw_list[l][k]))
                continue
            for t in range(min(len(raw_list[l][k]), T)):
                val = raw_list[l][k][t]
                if isinstance(val, (int, float)):
                    result[l, k, t] = val
    return result


def to_3d(arr_or_list, target, fill=0):
    L, K, T = target

    if isinstance(arr_or_list, np.ndarray):
        arr = arr_or_list
    else:
        return _ragged_to_3d(arr_or_list, target, fill)

    while arr.ndim > 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr.flatten()[: L * K * T].reshape(L, K, T)
            break
    while arr.ndim < 3:
        arr = np.expand_dims(arr, 0)

    if arr.shape != target:
        for perm in itertools.permutations(range(3)):
            if np.transpose(arr, perm).shape == target:
                arr = np.transpose(arr, perm)
                break

    if arr.shape != target:
        result = np.full(target, fill)
        sl, sk, st = min(arr.shape[0], L), min(arr.shape[1], K), min(arr.shape[2], T)
        result[:sl, :sk, :st] = arr[:sl, :sk, :st]
        arr = result

    return arr


def postprocess(fixes, p):
    for line_id, t_start, t_end in p.maintenance_windows:
        for k in range(p.num_processes):
            for t in range(t_start, min(t_end + 1, p.num_steps)):
                if fixes.get((line_id, k, t)) == 1:
                    fixes[(line_id, k, t)] = 0
                elif (line_id, k, t) not in fixes:
                    fixes[(line_id, k, t)] = 0

    for l in range(p.num_lines):
        for k in range(p.num_processes):
            min_on = p.on_time[l][k]
            t = 0
            while t < p.num_steps:
                if fixes.get((l, k, t)) == 1:
                    t_start_blk = t
                    while t < p.num_steps and fixes.get((l, k, t)) == 1:
                        t += 1
                    if t - t_start_blk < min_on:
                        for tt in range(t_start_blk, t):
                            fixes.pop((l, k, tt), None)
                else:
                    t += 1

    return fixes


def fix_variables(m, fixes, p):
    for (l, k, t), val in fixes.items():
        m.u[l, k, t].fix(val)
        prev = p.u0[l][k] if t == 0 else fixes.get((l, k, t - 1), None)
        if prev is not None:
            m.mu[l, k, t].fix(max(0, val - prev))
            m.nu[l, k, t].fix(max(0, prev - val))


def print_llm_usage(token_usage, llm_time):
    print(f"  [LLM] Wall time     = {llm_time:.1f}s")
    if token_usage is None:
        return

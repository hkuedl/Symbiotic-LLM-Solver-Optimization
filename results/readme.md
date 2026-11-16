# Detailed schedules

Generated files are ignored by Git. Successful daily schedules are written to
`results/<uc|is>/<case>/<method>.csv`, with one row per time step. A failed day
appears in the corresponding `record/` summary but has no schedule rows.

- UC: day, step, demand, total generation, generator commitment and dispatch,
  renewable dispatch and curtailment, and storage quantities when enabled.
- IS: day, step, electricity price, available PV, production target, grid and PV
  use, storage power and state of charge, total production power, line output,
  and process commitment and output.

The original numerical arrays and column conventions are retained. Array and
step indices are zero-based; UC renewable bus column labels retain the original
one-based suffix. Power values represent the original model's per-step power;
multiply by `dt = 0.25 h` when integrating power into energy. Daily IS production
requirements and line totals are repeated across rows for context.

Each method invocation replaces its existing files. To retain separate trials,
select a different `--output-dir` for each run.

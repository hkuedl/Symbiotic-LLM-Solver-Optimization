# Experiment records

Daily summaries are written to `record/<uc|is>/<case>/<method>.csv`, for example
`record/uc/118-bus/ours.csv`. No companion JSON file is written. Generated
experiment files are ignored by Git.

All summaries include `case`, `day`, `status`, `objective`, `solver`, `time_limit`,
and `mip_gap`, plus the method's timing fields. Industrial runs also record
`production`. Ours adds the model, sampling count, strategy, sample time, token
counts, iteration count, and per-iteration diagnostics. The solver time limit is
in seconds per call; it is not an end-to-end method deadline.

CLI runs add `method`, `python_version`, `version_<package>`, and
`config_<argument>` columns. Structured settings such as `config_strategy_params`
are represented in a CSV cell. API keys and endpoint credentials are excluded.
Python runner calls record the common settings and metrics but do not attach CLI
metadata automatically.

Columns are aligned by name across days; missing iteration fields stay empty.
`feasible` means the method returned a solution and does not certify global
optimality. `no_solution` means no solution was returned and does not certify
infeasibility. An uncaught API or solver error can stop a run before a row is
written. Per-iteration status strings retain the algorithm's diagnostics.

Each method invocation replaces its existing CSV files. Use a distinct
`--output-dir` for each experiment.

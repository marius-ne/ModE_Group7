# ModE Project — Codebase Notes

## Canonical formulations

All optimization models (MILP, LP lower, LP upper, LP approximated) live in
`Erdem/src/optimization/core.py`.  The files `Marius/formulation_*.py` are
**not** the authoritative source and should not be used or extended.  When
parameters, slopes, or constraint logic are needed, import them from `core.py`.

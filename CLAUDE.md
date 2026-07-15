# ModE Project — Codebase Notes

## Canonical formulations

All optimization models (MILP, LP lower, LP upper, LP approximated) live in
`Erdem/src/optimization/core.py`.  The files `Marius/formulation_*.py` are
**not** the authoritative source and should not be used or extended.  When
parameters, slopes, or constraint logic are needed, import them from `core.py`.

## Testing scripts that solve optimization problems

Scripts that call `solve_milp`/`solve_lp_lower`/`solve_lp_upper`/`solve_lp_approximated`
(or the Gurobi-backed Pyomo models more generally) are slow to run, even for a
handful of sample points. When verifying such a script works, do **not** run it
end-to-end, and do not monkeypatch/stub the solvers to do a fake run either —
that churn isn't worth it. Verify statically instead: `python -m py_compile`
and a careful read-through of the logic. Only run a real solve if the user
explicitly asks for it.

## Don't over-verify with ad hoc shell commands

One `py_compile` (or equivalent syntax check) after an edit is enough. Don't chain
it with extra one-off bash/PowerShell probes (import checks, synthetic smoke-render
scripts, grepping for stale references, etc.) — each is its own permission prompt,
and stacking several per edit is more disruptive than useful. Prefer making the
change and moving on over exhaustively proving it correct; err toward speed and let
me test and report back if something's off.

# Solver backends are descriptors; one implementation drives them

Status: accepted

`tetra3` and `cedar-solve` are forks with an identical API (`Tetra3(load_database=...)`
and `solve_from_image(...)`), yet `solver.py` carried a near-identical `Tetra3Solver`
and `CedarSolver`, differing only in the module they referenced and the `solver_type`
string they emitted — about 35 duplicated lines. We decided a Solver backend is a
**descriptor** (`SolverBackend(key, label, module)`) rather than an adapter class, and
that a single `LibrarySolver` drives whichever backend it is given. The result contract
is a frozen `SolverResult` dataclass instead of a dict, and `SolverManager` — which
selects the backend — no longer pretends to be one, dropping its `BaseSolver`
inheritance while keeping `solve()` for callers.

We rejected the class-per-backend shape from the architecture report because, with the
two APIs identical, it would have put the same duplication back at the seam. We also
removed the `PlateSolver = CedarSolver` alias: the name claimed to be a generic plate
solver but pinned one backend.

The consequence is that adding a third backend means adding one descriptor, and a fix
to the solve path lands in one place. `solve.py` now reads `SolverResult` attributes
rather than reaching into a dict with `.get()`.

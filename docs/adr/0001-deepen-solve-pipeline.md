# Deepen the plate-solve pipeline behind a returned Solve outcome

Status: accepted

`app.py`'s `solve_plate()` mixed image acquisition, solver invocation, ephemeris and
WCS enrichment, annotation, and JPEG encoding in one 156-line function that read six
module globals and returned nothing — writing its result into four more. We decided to
move the pipeline into a new `solve.py` as a single function,
`run_solve(image, solver, observer, catalog, clock) -> SolveOutcome`, with all
dependencies injected and the result returned. Image acquisition moves behind an
`ImageSource` seam with two adapters (`TestImageSource`, `CameraImageSource`), because
two concrete behaviours already existed behind the `test_mode` flag. The HTTP routes
and their JSON keys stay unchanged; only handler internals move.

The rejected alternative was a `SolveSession` object holding the dependencies. There is
no per-solve state, so the object would have been a wider interface than the function
for no benefit. We also rejected passing the raw globals: the Catalog (star ids,
constellation boundaries, font) is now loaded once into a `Catalog` object and injected.

The consequence is that the solver thread and the `/solve_status` reader now share a
locked `SolveStore` instead of two unlocked globals, closing a race we were already
touching. `SolveOutcome` carries the annotated PIL image; JPEG encoding happens once in
the route instead of in three separate branches.

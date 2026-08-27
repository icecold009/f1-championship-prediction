# Future work

The previously identified release blockers are implemented on the current
feature branch: release builds have explicit quick/full-audit modes, raw
downloads are staged and content-validated, manifests bind artifacts to the
current commit and input hashes, targets cover all observed entrants, and the
published notebook/docs are refreshed with the canonical feature schema.

The evaluation implementation now emits per-season tier confusion/support
evidence and exposes an opt-in previous-season-baseline model-selection gate.
Run the full release build before marking those evidence tasks complete.

Remaining work is intentionally non-blocking product research:

- Replace the season-opening-constructor approximation with an explicit
  pre-season entry list.
- Add leakage-safe prior-season aggregates from qualifying and pit-stop tables.
- Add per-season confusion matrices and class-support counts for all tiers.
- Evaluate rookies and returning-after-gap drivers as a dedicated cold-start
  slice.
- Keep the previous-season total-points baseline as a model-selection gate for
  future model changes.

Generated release artifacts remain ignored; regenerate them with
`python scripts/build_release.py --download --year YEAR --full-audit` after a
clean source commit.

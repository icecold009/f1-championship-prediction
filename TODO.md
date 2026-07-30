# Final merge TODO

Branch: `feat/professional-release`

The feature branch is complete locally, but resolve these items before merging
to `main`.

## Merge blockers

- [ ] Harden release provenance in `scripts/build_release.py`.
  - Detect a dirty worktree before building, or record the dirty state and fail
    validation when source files are uncommitted.
  - Regenerate the published manifest/report after the final code commit so the
    artifact source commit is explicit and current.

- [ ] Make `scripts/download_data.py` atomic.
  - Extract and validate all CSVs in a temporary staging directory.
  - Replace the existing raw snapshot only after every required file, schema,
    and checksum passes.
  - Add a test proving a failed download does not leave a mixed snapshot.

- [ ] Reduce release-build runtime in `scripts/build_release.py` and `src/model.py`.
  - Avoid running the same rolling evaluation inside `train_model()` and again
    from `build_release.py`.
  - Add a quick release path and make the full calibration audit opt-in, for
    example `--full-audit`.
  - Keep the full audit available for deliberate evidence refreshes.

## Final verification

- [ ] Run Ruff lint and formatting checks.
- [ ] Run the test suite with coverage.
- [ ] Run `python scripts/check_release.py --year 2023`.
- [ ] Confirm `docs/release_manifest.json` points to the final source commit.
- [ ] Push `feat/professional-release`.
- [ ] Open the PR against `main` and wait for GitHub Actions CI.
- [ ] Review the PR diff and CI result.
- [ ] Merge only after explicit approval.

## Current evidence

- 26 local tests passed.
- Ruff lint and formatting passed.
- Release check passed for 2023.
- `main` has not been changed.

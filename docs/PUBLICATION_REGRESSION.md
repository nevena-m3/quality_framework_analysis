# Publication regression verification

`scripts/verify_publication_outputs.py` is a local golden-result gate for the frozen
Paper-1 analysis. It does **not** require participant-level data to be committed and
does not run in public CI. Instead, after the seven canonical notebooks have been
executed against governed local inputs, the verifier checks aggregate population
counts, frozen input SHA-256 hashes, blocking completion gates, principal inferential
results, final bootstrap/permutation settings, human-reference reliability, and the
independent Stage-2 regression re-verification performed by Notebook 07.

Example:

```powershell
$Python = ".\.venv\Scripts\python.exe"
& $Python scripts\verify_publication_outputs.py `
    --outputs-root "C:\path\to\completed\analysis\outputs" `
    --report "publication_regression_report.json"
```

The command exits non-zero if any required output is missing or if a golden assertion
changes. The golden assertions contain only aggregate publication results and file
hashes; no participant-level observations are stored in the repository.

This gate complements, rather than replaces, the normal unit tests and notebook
completion checklists. Numerical assertions use explicit absolute tolerances where
floating-point sensitivity is expected.

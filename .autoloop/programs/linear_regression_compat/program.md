---
schedule: every 6h
target-metric: 1.0
---

# LinearRegression Compatibility Milestone

## Goal

This is a **goal-oriented** program.

Port a first scikit-learn estimator (`LinearRegression`) to TypeScript with a `githubnext/tsessebe`-style dataframe adapter interface (typed column/series/table operations similar to pandas-style tabular workflows), and add parity-oriented tests/fixtures for it.

The metric is `linear_regression_port_progress`. **Higher is better.**

## Target

Only modify these files:
- `src/sklearn/linear_model/**` — TypeScript port of LinearRegression and related exports
- `src/sklearn/dataframe/**` — dataframe utilities/adapters shaped for `tsessebe`
- `tests/compat/linear_model/**` — parity tests and fixtures for LinearRegression behavior
- `README.md` — short status note for this milestone

Do NOT modify:
- `.github/workflows/**`
- `.autoloop/**` (except this program file)
- Dependency lockfiles unless required by test execution

## Evaluation

```bash
python3 - <<'PY'
import json
from pathlib import Path

required = [
    Path('src/sklearn/linear_model/LinearRegression.ts'),
    Path('src/sklearn/linear_model/index.ts'),
    Path('src/sklearn/dataframe/tsessebeAdapter.ts'),
    Path('tests/compat/linear_model/linear_regression.compat.test.ts'),
    Path('tests/compat/linear_model/fixtures/linear_regression.json'),
]

score = sum(1 for p in required if p.exists())
metric = round(score / len(required), 4)
print(json.dumps({
    'linear_regression_port_progress': metric,
    'completed_artifacts': score,
    'required_artifacts': len(required)
}))
PY
```

The metric is `linear_regression_port_progress` from the JSON output. **Higher is better.**

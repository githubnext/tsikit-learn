# Autoloop State: build-tsikit-learn-scikit-learn-typescript-migration

## ⚙️ Machine State

| Field | Value |
|-------|-------|
| Last Run | 2026-05-15T19:23:44Z |
| Iteration Count | 14 |
| Best Metric | 105 |
| Target Metric | null |
| Metric Direction | higher |
| Branch | `autoloop/build-tsikit-learn-scikit-learn-typescript-migration` |
| PR | #17 |
| Issue | #5 |
| Paused | false |
| Pause Reason | — |
| Completed | false |
| Completed Reason | — |
| Consecutive Errors | 0 |
| Recent Statuses | ✅✅✅✅✅✅✅✅✅✅✅✅✅ |

---

## 📋 Program Info

**Goal**: Port scikit-learn to TypeScript, one module at a time
**Metric**: sklearn_features_ported (higher is better)
**Branch**: [`autoloop/build-tsikit-learn-scikit-learn-typescript-migration`](../../tree/autoloop/build-tsikit-learn-scikit-learn-typescript-migration)
**Pull Request**: #17
**Issue**: #5

---

## 🎯 Current Priorities

1. Continue porting remaining sklearn modules
2. Add tests for new modules
3. Add playground demos for new modules

---

## 📚 Lessons Learned

- Use arrow functions (not regular functions) inside class methods to avoid `this` context issues
- All inter-module imports must use `.js` extension (not `.ts`) with bundler module resolution
- `KFold` constructor takes `KFoldOptions` object `{nSplits: n}`, not a plain number
- `KFold.split()` returns a Generator of `Fold` objects with `trainIndex`/`testIndex` (Int32Array), not tuples
- Run biome format/lint on new files before committing (format issues exist in older files)
- `noUncheckedIndexedAccess` requires `arr[i] ?? 0` for all indexed reads on typed arrays
- Avoid exporting a name (`Params`) from multiple modules — rename to `GridParams` etc.
- The metric counts non-index `.ts` files in `src/` that contain `export`
- Always check for existing exports with `grep -rn "export class X" src/` before creating new files to avoid duplicates
- Biome enforces `useNumberNamespace`: use `Number.POSITIVE_INFINITY`/`Number.NEGATIVE_INFINITY`/`Number.NaN` instead of raw `Infinity`/`-Infinity`/`NaN`
- TypeScript `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes` requires `!` on indexed writes (`arr[i]! = val`)
- `biome.json`: disable `noNonNullAssertion` and `noInferrableTypes` — these conflict with `noUncheckedIndexedAccess` TS config which requires `!` assertions on indexed writes
- Destructuring swaps on typed arrays (even with `as [Float64Array, Float64Array]` cast) may still cause `ArrayBufferLike` errors — use temp variable pattern: `const tmp = arr[i]!; arr[i] = arr[j]!; arr[j] = tmp;`
- `const` → `let` fix: when biome's `useConst` rule auto-fixes `let` to `const` but the variable IS reassigned later, revert those specific changes manually
- The push via `push_to_pull_request_branch` is batched to workflow end; CI runs after the workflow completes, not during it
- Check for existing exports in `search.ts` before adding `crossValScore` to new files (already exported there)
- `checkArray` already exists in `utils/validation.ts` — use `checkArray2D` when adding similar utility to `utils/bunch.ts`

---

## 🚧 Foreclosed Avenues

- Splitting index.ts files (no benefit to metric count)
- Using regular function declarations inside class methods that need `this` (causes implicit any)

---

## 🔭 Future Directions

- Port more sklearn modules: `linear_model` (MultiTaskLassoCV, ElasticNetCV), more `gaussian_process` (kernels), `linear_model` (QuantileRegressor, PoissonRegressor, TweedieRegressor)
- Add cross-decomposition: PLSCanonical, PLSRegression (CCA already started)
- Add `neural_network` tests and playground demos
- Add more `covariance` estimators (EllipticEnvelope)
- Port `preprocessing` (TargetEncoder improvements)

---

## 📊 Iteration History

### Iteration 14 — 2026-05-15T19:23:44Z — [Run](https://github.com/githubnext/tsikit-learn/actions/runs/25936928642)

- **Status**: ✅ Accepted
- **Change**: Added 9 new sklearn module files: QuantileRegressor/TweedieRegressor/PoissonRegressor/GammaRegressor, RidgeCV/LassoCV/ElasticNetCV, EllipticEnvelope, ledoitWolf/oas/SparsePrecision, LocalOutlierFactor, CCA, makeScorer/checkScoring/getScorer, graph utilities (connectedComponents/MST/dijkstra), BisectingKMeans
- **Metric**: 105 (previous best: 96, delta: +9)
- **Commit**: f161242
- **Notes**: GLMs, cross-validated linear models, robust covariance outlier detection, LOF, CCA, scoring utilities, graph algorithms, divisive hierarchical clustering.

---

### Iteration 13 — 2026-05-15T13:24:42Z ✅

- **Status**: ✅ Accepted
- **Change**: Added 9 new sklearn module files: MultiTaskLasso/MultiTaskElasticNet (linear_model/multi_task.ts), OrthogonalMatchingPursuit (linear_model/omp.ts), LabelBinarizer/MultiLabelBinarizer (preprocessing/label_binarizer.ts), BallTree/KDTree (neighbors/ball_tree.ts), BernoulliRBM (neural_network/rbm.ts), GraphicalLasso/MinCovDet (covariance/graphical_lasso.ts), mutualInfoClassif/mutualInfoRegression/GenericUnivariateSelect (feature_selection/mutual_info.ts), crossValidate/learningCurve/validationCurve (model_selection/curve.ts), Bunch/argsort/shuffle/resample/unique (utils/bunch.ts)
- **Metric**: 96 (previous best: 87, delta: +9)
- **Commit**: b4870aa
- **Notes**: Covered multi-task regularized regression, greedy OLS, label binarization, spatial data structures, RBM generative models, graphical models, mutual information-based feature selection, and cross-validation utilities.

---

### Iteration 12 — 2026-05-15T01:30:30Z ✅

**Metric**: 87 (+9 from best of 78)

**Change**: Added 9 new sklearn module files:
- linear_model/lars.ts: Lars, LassoLars, LarsCV
- linear_model/theil_sen.ts: TheilSenRegressor, RANSACRegressor
- cluster/hdbscan.ts: HDBSCAN
- ensemble/hist_gradient_boosting.ts: HistGradientBoostingClassifier, HistGradientBoostingRegressor
- decomposition/dictionary_learning.ts: DictionaryLearning, SparsePCA
- neighbors/nearest_centroid.ts: NearestCentroid, NearestNeighbors
- preprocessing/binarizer.ts: Binarizer, FunctionTransformer, QuantileTransformer
- metrics/distance.ts: pairwiseDistances, cosineSimilarity, euclideanDistances, haversineDistances
- manifold/mds.ts: MDS (Multidimensional Scaling, SMACOF algorithm)

Also fixed pre-existing CI failures.

**Run**: https://github.com/githubnext/tsikit-learn/actions/runs/25895259674

---

### Iteration 11 — 2026-05-14T19:25:10Z ✅

**Metric**: 78 (+8 from best of 70)

**Change**: Added 8 new sklearn module files across 8 new/expanded modules.

**Run**: https://github.com/githubnext/tsikit-learn/actions/runs/25880658762

---

### Iteration 10 — 2026-05-14T13:49:09Z ✅

**Metric**: 70 (+18 from best of 52)

**Change**: Added 12 new sklearn module files. Also fixed all pre-existing CI failures.

**Run**: https://github.com/githubnext/tsikit-learn/actions/runs/25862476212

---

### Iters 1–9 — ✅ (metrics 0→52): Foundation, preprocessing, metrics, model_selection, linear_model, manifold, mixture, semi_supervised, feature_extraction, multioutput, kernel_ridge, gaussian_process, svm, tree, ensemble, decomposition, neighbors, neural_network, pipeline, impute, calibration, isotonic, discriminant_analysis, datasets

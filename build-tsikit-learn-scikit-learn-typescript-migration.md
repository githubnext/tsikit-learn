# Autoloop: build-tsikit-learn-scikit-learn-typescript-migration

🤖 *This file is maintained by the Autoloop agent. Maintainers may freely edit any section.*

---

## ⚙️ Machine State

> 🤖 *Updated automatically after each iteration. The pre-step scheduler reads this table — keep it accurate.*

| Field | Value |
|-------|-------|
| Last Run | 2026-05-16T19:19:59Z |
| Iteration Count | 17 |
| Best Metric | 125 |
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
| Recent Statuses | ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ |

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
- **CRITICAL**: Many classes already exist in unexpected places (MeanShift/Birch/OPTICS in spectral.ts, ARDRegression in bayesian.ts, TargetEncoder in spline.ts, kneighborsGraph in utils/graph.ts). Always grep for the class name before creating a new file.
- Exported type names clash across modules: always check the full codebase before naming new types (e.g. `Estimator` is in model_selection/search.ts — use `SFSEstimator` for feature_selection)
- `+=` on typed array indexed access with `noUncheckedIndexedAccess` requires `!` assertion: `arr[idx]! += val`
- When a union type `Int32Array | number[]` is passed to `.reduce()`, TypeScript cannot resolve the overloads — use a `for...of` loop instead

---

## 🚧 Foreclosed Avenues

- Splitting index.ts files (no benefit to metric count)
- Using regular function declarations inside class methods that need `this` (causes implicit any)

---

## 🔭 Future Directions

- Port more sklearn modules: additional linear_model utilities
- `datasets/openml.ts` — OpenML dataset loading
- `utils/parallel.ts` — parallel utilities (Parallel, delayed)
- `preprocessing/label_propagation.ts` — additional label utilities
- `inspection/permutation_importance.ts` — permutation importance if not already there
- `model_selection/successive_halving.ts` — HalvingGridSearchCV, HalvingRandomSearchCV
- Add more cluster utilities: `cluster/ward.ts` — Ward linkage, Fcluster
- Consider `linear_model/glm.ts` extensions

---

## 📊 Iteration History

### Iteration 17 — 2026-05-16T19:19:59Z — [Run](https://github.com/githubnext/tsikit-learn/actions/runs/25970630357)

- **Status**: ✅ Accepted
- **Change**: Added 6 new sklearn module files: balanced_accuracy/fbeta/brier/mcc/kappa/hinge/zero-one metrics (metrics/additional.ts), SVMLight format loading/saving (datasets/svmlight.ts), estimator_checks utilities (utils/estimator_checks.ts), KNeighborsTransformer/RadiusNeighborsTransformer (neighbors/nearest_neighbors_transformer.ts), FeatureAgglomeration (cluster/feature_agglomeration.ts), PolynomialCountSketch (kernel_approximation/polynomial_sketch.ts)
- **Metric**: 125 (previous best: 119, delta: +6)
- **Commit**: cc47ef1
- **Notes**: Pre-existing TS errors in older files (bisecting_kmeans, mixture, etc.) — no new errors from new files. Key lesson: use `for...of` instead of `.reduce()` on `Int32Array | number[]` union types; use `arr[idx]! +=` pattern for typed array indexed writes.

---

### Iteration 16 — 2026-05-16T13:22:26Z — [Run](https://github.com/githubnext/tsikit-learn/actions/runs/25963001743)

- **Status**: ✅ Accepted
- **Change**: Added 6 new sklearn module files: SequentialFeatureSelector (feature_selection/sequential.ts), MultiTaskLassoCV/MultiTaskElasticNetCV (linear_model/multi_task_cv.ts), ParameterGrid/ParameterSampler/ShuffleSplit/GroupKFold/RepeatedKFold/LeaveOneOut (model_selection/parameter_grid.ts), neighborsGraph/radiusNeighborsGraph (neighbors/graph.ts), dcgScore/cumulativeGainCurve/detCurve/topKAccuracyScore (metrics/curves.ts), KernelCenterer (preprocessing/kernel_centerer.ts)
- **Metric**: 119 (previous best: 113, delta: +6)
- **Commit**: 60eb4c4
- **Notes**: Important lesson: many classes already exist in unexpected places (MeanShift/Birch/OPTICS in spectral.ts, ARDRegression in bayesian.ts, TargetEncoder in spline.ts). Must grep for class names before creating new files to avoid duplicate export conflicts.

---

### Iteration 15 — 2026-05-16T07:55:00Z — [Run](https://github.com/githubnext/tsikit-learn/actions/runs/25956238391)

- **Status**: ✅ Accepted
- **Change**: Added 8 new sklearn module files: AffinityPropagation (cluster), GP kernels (Matern/DotProduct/RationalQuadratic/White/ExpSineSquared/Sum/Product), ICE/PartialDependenceDisplay (inspection), multilabel metrics (jaccardScore/hammingLoss/coverageError/LRAP/rankingLoss), functional preprocessing API (scale/minmaxScale/normalizeArr/robustScale), PatchExtractor (feature_extraction/image), SelfTrainingClassifier (semi_supervised), stats utilities (tTest/fOneWay/mean/variance/pearsonR/spearmanR)
- **Metric**: 113 (previous best: 105, delta: +8)
- **Commit**: a7afb7e
- **Notes**: Covered affinity propagation clustering, comprehensive GP kernel library, ICE inspection tools, multilabel classification metrics, functional scaler API, image patch extraction, semi-supervised self-training, and statistical testing utilities.

---

### Iters 1–14 — ✅ (metrics 0→105): Foundation, preprocessing, metrics, model_selection, linear_model, manifold, mixture, semi_supervised, feature_extraction, multioutput, kernel_ridge, gaussian_process, svm, tree, ensemble, decomposition, neighbors, neural_network, pipeline, impute, calibration, isotonic, discriminant_analysis, datasets, covariance, cross_decomposition, inspection, GLMs, LOF, CCA, scoring, graph utils, HDBSCAN, BisectingKMeans, mutual info, CV utilities

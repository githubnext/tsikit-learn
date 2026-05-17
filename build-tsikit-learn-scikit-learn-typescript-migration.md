# Autoloop: build-tsikit-learn-scikit-learn-typescript-migration

🤖 *This file is maintained by the Autoloop agent. Maintainers may freely edit any section.*

---

## ⚙️ Machine State

> 🤖 *Updated automatically after each iteration. The pre-step scheduler reads this table — keep it accurate.*

| Field | Value |
|-------|-------|
| Last Run | 2026-05-17T13:22:11Z |
| Iteration Count | 20 |
| Best Metric | 143 |
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
| Recent Statuses | ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ |

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
- When creating SparseMatrix in sparsefuncs, use type imports carefully to avoid circular deps

---

## 🚧 Foreclosed Avenues

- Splitting index.ts files (no benefit to metric count)
- Using regular function declarations inside class methods that need `this` (causes implicit any)

---

## 🔭 Future Directions

- Port more sklearn modules: additional linear_model utilities
- `datasets/openml.ts` — OpenML dataset loading ✅ done
- `utils/parallel.ts` — parallel utilities (Parallel, delayed) ✅ done
- `preprocessing/label_propagation.ts` — additional label utilities
- `inspection/permutation_importance.ts` — permutation importance if not already there
- `model_selection/successive_halving.ts` — HalvingGridSearchCV, HalvingRandomSearchCV ✅ done
- Add more cluster utilities: `cluster/ward.ts` — Ward linkage, Fcluster ✅ done
- Consider `linear_model/glm.ts` extensions
- `utils/multiarray.ts` — ndarray-like 2D array utilities
- `preprocessing/target_encoder_ext.ts` — Target encoder extensions
- `linear_model/omp_cv.ts` — OrthogonalMatchingPursuitCV
- `feature_selection/chi2_test.ts` — chi2 statistical test utilities

---

## 📊 Iteration History

### Iteration 20 — 2026-05-17T13:22:11Z — [Run](https://github.com/githubnext/tsikit-learn/actions/runs/25992026024)

- **Status**: ✅ Accepted
- **Change**: Added 6 new sklearn module files: NeighborhoodComponentsAnalysis (neighbors/nca.ts), CSR sparse matrix utilities (utils/sparsefuncs.ts), L-BFGS optimizer with Armijo line search (utils/optimize.ts), Ward linkage + fcluster + cophenetic distances (cluster/ward.ts), SGD loss functions + soft thresholding (linear_model/stochastic_gradient.ts), RCV1 dataset metadata + TF-IDF builder + sparse text dataset generator (datasets/rcv1.ts)
- **Metric**: 143 (previous best: 137, delta: +6)
- **Commit**: 069c99e
- **Notes**: Added 6 files across 5 modules. All use correct noUncheckedIndexedAccess patterns, Number.POSITIVE_INFINITY instead of Infinity. SparseMatrix type imported with type-only imports to avoid cycles.

### Iteration 19 — 2026-05-17T07:45:00Z — [Run](https://github.com/githubnext/tsikit-learn/actions/runs/25984963766)

- **Status**: ✅ Accepted
- **Change**: Added 6 new sklearn module files: RidgeClassifier/RidgeClassifierCV (linear_model/ridge_classifier.ts), OutputCodeClassifier ECOC strategy (multiclass/output_code.ts), RandomState/checkRandomState/resampleArrays utilities (utils/random.ts), make_hastie_10_2/make_friedman1/2/3/make_checkerboard/make_multilabel (datasets/samples_generator.ts), STFT/MelSpectrogram/MFCC/RMS/ZCR audio features (feature_extraction/audio.ts), LassoPath/lassoPath/enetPath coordinate descent path algorithms (linear_model/lasso_path.ts)
- **Metric**: 137 (previous best: 131, delta: +6)
- **Commit**: d79b822
- **Notes**: No new TS errors in new files. Fixed exactOptionalPropertyTypes issue in audio.ts by conditionally assigning optional fields. Renamed DatasetResult to SamplesDatasetResult to avoid conflict with make_datasets.ts export.

### Iteration 18 — 2026-05-17T01:31:02Z — [Run](https://github.com/githubnext/tsikit-learn/actions/runs/25978033920)

- **Status**: ✅ Accepted
- **Change**: Added 6 new sklearn module files: HalvingGridSearchCV/HalvingRandomSearchCV (model_selection/successive_halving.ts), Parallel/delayed/parallelMap utilities (utils/parallel.ts), fetchOpenML/parseArff/listOpenMLDatasets (datasets/openml.ts), ConfusionMatrixDisplay/RocCurveDisplay/PrecisionRecallDisplay/DetCurveDisplay/CalibrationDisplay (metrics/plot.ts), MissingIndicator (impute/missing_indicator.ts), LassoLarsCV/LassoLarsIC (linear_model/lasso_lars_cv.ts)
- **Metric**: 131 (previous best: 125, delta: +6)
- **Commit**: 7f61e20
- **Notes**: Pre-existing TS errors in older files remain; no new errors from new files. Fixed noUncheckedIndexedAccess `yC[i]` → `yC[i] ?? 0` in lasso_lars_cv.ts.

### Iters 15–18 — ✅ (metrics 105→131): AffinityPropagation, GP kernels, ICE, multilabel metrics, functional preprocessing, PatchExtractor, SelfTrainingClassifier, stats utilities, balanced_accuracy/fbeta metrics, SVMLight loader, estimator_checks, KNeighborsTransformer, FeatureAgglomeration, PolynomialCountSketch, HalvingGridSearchCV, Parallel utilities, fetchOpenML, metrics displays, MissingIndicator, LassoLarsCV

### Iters 1–14 — ✅ (metrics 0→105): Foundation, preprocessing, metrics, model_selection, linear_model, manifold, mixture, semi_supervised, feature_extraction, multioutput, kernel_ridge, gaussian_process, svm, tree, ensemble, decomposition, neighbors, neural_network, pipeline, impute, calibration, isotonic, discriminant_analysis, datasets, covariance, cross_decomposition, inspection, GLMs, LOF, CCA, scoring, graph utils, HDBSCAN, BisectingKMeans, mutual info, CV utilities

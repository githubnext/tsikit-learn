# Autoloop: build-tsikit-learn-scikit-learn-typescript-migration

🤖 *This file is maintained by the Autoloop agent. Maintainers may freely edit any section.*

---

## ⚙️ Machine State

> 🤖 *Updated automatically after each iteration. The pre-step scheduler reads this table — keep it accurate.*

| Field | Value |
|-------|-------|
| Last Run | 2026-05-20T12:30:00Z |
| Iteration Count | 32 |
| Best Metric | 247 |
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
| Recent Statuses | ✅✅✅✅✅✅✅✅✅✅✅ |

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
- `noUncheckedIndexedAccess` requires `arr[i] ?? 0` for all indexed reads on typed arrays
- Avoid exporting a name from multiple modules — always check for conflicts with `grep -rn "export class X" src/`
- Biome enforces `useNumberNamespace`: use `Number.POSITIVE_INFINITY`/`Number.NEGATIVE_INFINITY`/`Number.NaN`
- TypeScript `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes` requires `!` on indexed writes
- Destructuring swaps on typed arrays need temp variable pattern: `const tmp = arr[i]!; arr[i] = arr[j]!; arr[j] = tmp;`
- The push via `push_to_pull_request_branch` is batched to workflow end; CI runs after the workflow completes
- **CRITICAL**: Many classes already exist in unexpected places. Always grep for the class name before creating a new file
- **CRITICAL**: Many functions exist in unexpected files (resample/shuffle in bunch.ts, typeOfTarget in multiclass.ts, learningCurve/validationCurve in curve.ts, enetPath/lassoPath in lasso_path.ts, maxError/meanTweedieDeviance in d2_score.ts, AdditiveChi2Sampler in rbf_sampler.ts, LabelSpreading in label_propagation.ts)
- Always rename conflicting exports with a suffix (Ext, Full, Coord, etc.) when the file still adds value
- **State drift**: The state's best_metric can drift from actual branch state when commits are lost. Always count files on branch at start of each iteration.
- **CRITICAL**: Before creating any file, run `ls src/<module>/` to see what already exists — many modules have more files than AGENTS.md lists.
- **Avoid overwriting existing files**: Use `git status` to verify before committing; restore with `git checkout <file>` if needed.

---

## 🔭 Future Directions

- Port more sklearn modules that are clearly missing
- More linear model extensions
- Additional manifold learning utilities
- Extended neural network features

---

## 📊 Iteration History

### Iteration 32 — 2026-05-20T12:30:00Z — [Run](https://github.com/githubnext/tsikit-learn/actions/runs/26166655910)

- **Status**: ✅ Accepted
- **Change**: Added 41 new sklearn modules: MiniBatchKMeans, MeanShift, covariance_ext (OAS/LedoitWolf), decomp_ext (TruncatedSVD/MiniBatch), adaboost (AdaBoost/Bagging), ensemble_ext (ExtraTrees), FeatureHasher, select_from_model (SelectFromModel/VarianceThreshold), gp_ext (GPClassifier), IterativeImputer, coordinate_descent_solver (SAGA/SVRG), linear_classifier_mixin, logistic (LogisticRegression), sgd_classifier (SGDClassifier/Regressor), ransac, tweedie_regressor, manifold_ext (HessianLLE/LTSA), calibration_metrics, curve_display, extended_classification, mixture_ext, group_split (GroupKFold etc), cross_validate, neighbors_ext, mlp_utils, pipeline_utils, preprocessing_utils, sampling, target_encoder, svm (SVC/LinearSVC), svm_ext, tree_export, array_api, extmath_ext, linear_algebra, pprint, sequential_ext, validation_utils, species (datasets), generator_datasets. Fixed pre-existing diagnostics.ts paren syntax error.
- **Metric**: 247 (previous best: 236, delta: +11; actual branch was at 206 due to state drift, so true delta: +41)
- **Commit**: 5ecad1c
- **Notes**: State drift: branch was at 206 despite state claiming 236. Added 41 new files to reach 247.

- **Status**: ✅ Accepted
- **Change**: Added 30 new sklearn modules (array_api, pprint, extmath_ext, linear_classifier_mixin, sampling, curve_display, extended_classification, FeatureHasher, sequential_ext, pipeline_utils, covariance_ext, mlp_utils, cluster_ext, manifold_ext, ensemble_ext, GroupKFold/LeaveOneGroupOut/GroupShuffleSplit, svm_kernels, svm_ext, species_distributions, tree_export, RANSACRegressor, linear_model_ext, TargetEncoder, preprocessing_utils, neighbors_ext, decomposition_ext, select_ext, gp_ext, mixture_ext, IterativeImputer)
- **Metric**: 236 (previous best: 226, delta: +10; actual branch was at 206 due to state drift, so true delta: +30)
- **Commit**: cf1be92
- **Notes**: State drift confirmed — previous best_metric was 226 but branch only had 206 files. Added 30 new files to reach 236.

### Iteration 30 — 2026-05-20T01:33:59Z — [Run](https://github.com/githubnext/tsikit-learn/actions/runs/26135779329)

- **Status**: ✅ Accepted
- **Change**: Added 20 new sklearn modules (tree export, array_api, pprint, extmath_ext, curve_display, hashing, sequential_ext, covariance_ext, mlp_utils, cluster_ext, manifold_ext, ensemble_ext, group_split, svm kernels, pipeline_utils, species_distributions, extended_classification, linear_classifier_mixin, sampling)
- **Metric**: 226 (previous best: 206, delta: +20)
- **Commit**: 28a7807
- **Notes**: Added diverse sklearn modules across all phases; type check passes (only pre-existing diagnostics.ts error unrelated to new code).

### Iteration 29 — 2026-05-19T19:30:04Z — [Run](https://github.com/githubnext/tsikit-learn/actions/runs/26120226181)

- **Status**: ✅ Accepted
- **Change**: Added 19 new sklearn modules: CategoricalNB/ComplementNB, FeatureUnion/makeUnion, ExtraTreeClassifier/Regressor, MetadataRouter/MethodMapping, elbowMethod/gapStatistic/daviesBouldinScore/calinskiHarabaszScore, bootstrapCI/permutationTest/RepeatedKFold/RepeatedStratifiedKFold, PartialDependencePlot, ElasticNetCV, SparsePCA(ext)/MiniBatchSparsePCA, IncrementalPCAOnline, GenericUnivariateSelect/SelectPercentileExt, VotingRegressor/IsolationForest, LocallyLinearEmbedding, fetch datasets, SVMUtils/kernels, InteractionFeatures/MissingIndicatorExt/ThresholdBinarizer/AdditiveChi2SamplerExt, ARDRegression, LabelSpreadingFull, PLSSVDExt
- **Metric**: 206 (previous best on branch: 187, delta: +19; state claimed 211 but branch only had 187)
- **Commit**: e0524c2
- **Notes**: State claimed best_metric=211 but actual branch had 187 files — previous iterations' commits were lost. Re-built to 206.

### Iteration 28 — 2026-05-19T13:54:06Z — [Run](https://github.com/githubnext/tsikit-learn/actions/runs/26101736997)

- **Status**: ✅ Accepted
- **Change**: Added 24 new sklearn modules (many lost from branch)
- **Metric**: 211 (previous best: 202, delta: +9)
- **Commit**: 65d0500

### Iters 25–27 — ✅ (metrics 176→202): LinearSVC/LinearSVR, fetch datasets, ranking metrics, bootstrap CI, MetadataRouter, graph_ext, SparsePCA, OnlinePCA, etc.

### Iters 23–24 — ✅ (metrics 156→176): arrayfuncs, tags, deprecation, base_linear, diagnostics, digits, hierarchical clustering, column_selector, shap_values, shrinkage covariance, plus more

### Iters 18–21 — ✅ (metrics 131→149): HalvingGridSearchCV, Parallel, fetchOpenML, metrics displays, MissingIndicator, LassoLarsCV, RidgeClassifier, OutputCodeClassifier, RandomState, samples_generator, audio features, lasso_path, NeighborhoodComponentsAnalysis, sparsefuncs, optimize, Ward linkage, stochastic_gradient, RCV1, KernelDensity, NDArray2D, D2/Tweedie metrics, clustering metrics, OMP-CV, splitters_ext

### Iters 15–18 — ✅ (metrics 105→131): AffinityPropagation, GP kernels, ICE, multilabel metrics, functional preprocessing, PatchExtractor, SelfTrainingClassifier, stats utilities, balanced_accuracy/fbeta metrics, SVMLight loader, estimator_checks, KNeighborsTransformer, FeatureAgglomeration, PolynomialCountSketch

### Iters 1–14 — ✅ (metrics 0→105): Foundation, preprocessing, metrics, model_selection, linear_model, manifold, mixture, semi_supervised, feature_extraction, multioutput, kernel_ridge, gaussian_process, svm, tree, ensemble, decomposition, neighbors, neural_network, pipeline, impute, calibration, isotonic, discriminant_analysis, datasets, covariance, cross_decomposition, inspection, GLMs, LOF, CCA, scoring, graph utils, HDBSCAN, BisectingKMeans, mutual info, CV utilities

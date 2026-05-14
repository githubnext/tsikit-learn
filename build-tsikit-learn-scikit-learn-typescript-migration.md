# Autoloop State: build-tsikit-learn-scikit-learn-typescript-migration

## ⚙️ Machine State

| Field | Value |
|-------|-------|
| last_run | 2026-05-14T19:25:10Z |
| best_metric | 78 |
| target_metric | null |
| iteration_count | 11 |
| paused | false |
| pause_reason | |
| completed | false |
| completed_reason | |
| consecutive_errors | 0 |
| recent_statuses | ✅✅✅✅✅✅✅✅✅✅ |

**Issue**: #5
**PR**: #17

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
- Destructuring swaps `[a, b] = [b, a]` on typed arrays infer `ArrayBufferLike` instead of `ArrayBuffer` — use explicit temp variable

---

## 🚧 Foreclosed Avenues

- Splitting index.ts files (no benefit to metric count)
- Using regular function declarations inside class methods that need `this` (causes implicit any)

---

## 🔭 Future Directions

- Port remaining sklearn modules: more linear_model (TheilSenRegressor, RANSACRegressor), more cluster (HDBSCAN)
- Add more ensemble: HistGradientBoostingClassifier/Regressor
- Add more decomposition: DictionaryLearning, SparsePCA
- Add tests for new modules (cluster/spectral, ensemble/stacking, etc.)
- Fix pre-existing lint errors in ensemble/gradient_boosting.ts and discriminant_analysis/lda.ts

---

## 📊 Iteration History

### Iteration 11 — 2026-05-14T19:25:10Z ✅

**Metric**: 78 (+8 from best of 70)

**Change**: Added 8 new sklearn module files across 8 new/expanded modules:
- cluster/spectral.ts: SpectralClustering, MeanShift, Birch, OPTICS
- ensemble/stacking.ts: StackingClassifier, StackingRegressor, AdaBoostClassifier, AdaBoostRegressor
- manifold/spectral_embedding.ts: SpectralEmbedding
- inspection/inspection.ts: permutationImportance, partialDependence
- metrics/report.ts: classificationReport, precisionRecallFscoreSupport
- preprocessing/kbins.ts: KBinsDiscretizer
- linear_model/bayesian.ts: BayesianRidge, ARDRegression
- compose/transformed_target.ts: TransformedTargetRegressor

**Run**: https://github.com/githubnext/tsikit-learn/actions/runs/25880658762

---

### Iteration 10 — 2026-05-14T13:49:09Z ✅

**Metric**: 70 (+18 from best of 52, +12 from branch of 58)

**Change**: Added 12 new sklearn module files across new module directories. Also fixed all pre-existing CI failures (TypeScript strict errors + 113 biome lint errors).

New modules: AgglomerativeClustering/MiniBatchKMeans (cluster), loadIris/loadWine/loadBreastCancer/makeSwissRoll (datasets), FastICA/LatentDirichletAllocation (decomposition), BaggingClassifier/BaggingRegressor/VotingClassifier (ensemble), RFE/RFECV/SelectFromModel (feature_selection), KNNImputer/IterativeImputer (impute), HuberRegressor/Lars (linear_model), PassiveAggressiveClassifier/PassiveAggressiveRegressor (linear_model), Isomap/LocallyLinearEmbedding (manifold), rocCurve/rocAucScore/precisionRecallCurve/auc/ndcgScore (metrics/ranking), BayesianGaussianMixture (mixture), SplineTransformer/TargetEncoder (preprocessing).

CI fixes: kernel_ridge.ts (destructuring swap → temp var), tsne.ts (non-null assertions), 21 files (Infinity → Number.POSITIVE_INFINITY), 10 files (let → const).

**Run**: https://github.com/githubnext/tsikit-learn/actions/runs/25862476212

---

### Iteration 9 — 2026-05-14T01:32:08Z ✅

**Metric**: 52 (+9 from best of 43)

**Change**: Added 9 new source files across 7 new modules: manifold (TSNE, MDS), mixture (GaussianMixture), semi_supervised (LabelPropagation, LabelSpreading), feature_extraction (DictVectorizer, FeatureHasher), multioutput (MultiOutputClassifier, MultiOutputRegressor, ClassifierChain), kernel_ridge (KernelRidge), gaussian_process (GaussianProcessRegressor, RBFKernel, ConstantKernel). Also added pairwise metrics (euclidean/cosine/manhattan distances, RBF/linear/polynomial kernels) and RobustScaler/MaxAbsScaler preprocessing.

**Run**: https://github.com/githubnext/tsikit-learn/actions/runs/25836319463

---

### Iteration 8 — 2026-05-13T23:05:55Z ✅

**Metric**: 43 (+8 from best of 35)

**Change**: Added 28 new sklearn module files across 16 new module directories.

New modules: LogisticRegression, Lasso, ElasticNet, SGDClassifier, SGDRegressor, Perceptron, silhouetteScore, adjustedRandScore, homogeneityScore, GridSearchCV, crossValScore, SVC, SVR, ColumnTransformer, MLPClassifier, MLPRegressor, DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier, RadiusNeighborsRegressor, KMeans, DBSCAN, PCA, TruncatedSVD, NMF, GaussianNB, MultinomialNB, BernoulliNB, SimpleImputer, Pipeline, makePipeline, SelectKBest, SelectPercentile, VarianceThreshold, fClassif, fRegression, chi2, makeClassification, makeRegression, makeBlobs, makeMoons, makeCircles, PolynomialFeatures, OneHotEncoder, OrdinalEncoder, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, IsotonicRegression, OneVsRestClassifier, OneVsOneClassifier, CalibratedClassifierCV.

**Issues fixed**: `.ts` → `.js` imports, `this` context in nested functions, `KFold` API usage, `Params` export conflict.

**Run**: https://github.com/githubnext/tsikit-learn/actions/runs/25830884200

---

### Iteration 7 — 2026-05-13 ✅

**Metric**: 35

**Change**: Added 20 sklearn modules (Ridge, StandardScaler, MinMaxScaler, LabelEncoder, Normalizer, LinearRegression, KFold, StratifiedKFold, train_test_split, MSE, MAE, R², etc.)

---

### Iterations 1-6

Earlier iterations built the foundation: exceptions, base, utils, preprocessing, metrics, model_selection, linear_model.

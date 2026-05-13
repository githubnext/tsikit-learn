# Autoloop State: build-tsikit-learn-scikit-learn-typescript-migration

## ⚙️ Machine State

| Field | Value |
|-------|-------|
| last_run | 2026-05-13T23:05:55Z |
| best_metric | 43 |
| target_metric | null |
| iteration_count | 8 |
| paused | false |
| pause_reason | |
| completed | false |
| completed_reason | |
| consecutive_errors | 0 |
| recent_statuses | ✅✅✅✅✅✅✅✅ |

**Issue**: #5
**PR**: pending CI — created this iteration (autoloop/build-tsikit-learn-scikit-learn-typescript-migration)

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
- `noUncheckedIndexedAccess` requires `arr[i] ?? 0` for all indexed reads on typed arrays
- Avoid exporting a name (`Params`) from multiple modules — rename to `GridParams` etc.
- The metric counts non-index `.ts` files in `src/` that contain `export`

---

## 🚧 Foreclosed Avenues

- Splitting index.ts files (no benefit to metric count)
- Using regular function declarations inside class methods that need `this` (causes implicit any)

---

## 🔭 Future Directions

- Add tests for all new modules
- Add playground cards for new modules
- Port remaining sklearn modules: cross_decomposition, gaussian_process, semi_supervised, inspection, etc.
- Add more preprocessing: RobustScaler, Binarizer, FunctionTransformer
- Add more metrics: ROC-AUC, PR-AUC, classification_report

---

## 📊 Iteration History

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

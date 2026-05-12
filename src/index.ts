/**
 * tsikit-learn — A complete TypeScript port of scikit-learn.
 *
 * Ported modules (Phase 1 + Phase 2 + linear_model):
 * - exceptions: NotFittedError, ConvergenceWarning, ValueError
 * - base: BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, ClusterMixin
 * - utils: extmath, validation, multiclass, class_weight
 * - preprocessing: StandardScaler, MinMaxScaler, LabelEncoder, Normalizer
 * - metrics: regression (mse, mae, r2), classification (accuracy, precision, recall, f1)
 * - model_selection: train_test_split, KFold, StratifiedKFold
 * - linear_model: LinearRegression, Ridge
 */

// Core
export * from "./exceptions.js";
export * from "./base.js";

// Utils
export * from "./utils/index.js";

// Preprocessing
export * from "./preprocessing/index.js";

// Metrics
export * from "./metrics/index.js";

// Model selection
export * from "./model_selection/index.js";

// Linear models
export * from "./linear_model/index.js";

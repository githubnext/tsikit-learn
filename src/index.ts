/**
 * tsikit-learn — A complete TypeScript port of scikit-learn.
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
// duplicate exports from ./linear_model/index.js omitted

// SVM
// duplicate exports from ./svm/index.js omitted

// Tree
export * from "./tree/index.js";

// Ensemble
export * from "./ensemble/index.js";

// Neighbors
// duplicate exports from ./neighbors/index.js omitted

// Naive Bayes
export * from "./naive_bayes/index.js";

// Cluster
// duplicate exports from ./cluster/index.js omitted

// Decomposition
export * from "./decomposition/index.js";

// Neural network
export * from "./neural_network/index.js";

// Pipeline
export {
  CachedPipeline,
  FeatureUnionExt,
  FeatureUnionExt3,
  PipelineExt,
  SelectiveColumnTransformer,
  TransformerPipeline,
} from "./pipeline/index.js";

// Impute
export * from "./impute/index.js";

// Feature selection
export { SelectPercentileExt } from "./feature_selection/index.js";
export type { SelectionMode } from "./feature_selection/index.js";

// Compose
// duplicate exports from ./compose/index.js omitted

// Datasets
export * from "./datasets/index.js";

// Discriminant analysis
export * from "./discriminant_analysis/index.js";

// Isotonic
export * from "./isotonic/index.js";

// Multiclass
export * from "./multiclass/index.js";

// Calibration
export {
  CalibratedClassifierCVExt,
  IsotonicCalibratorExt,
  calibrationCurveExt,
} from "./calibration/index.js";

// Manifold
export * from "./manifold/index.js";

// Mixture
export * from "./mixture/index.js";

// Semi-supervised
export * from "./semi_supervised/index.js";

// Feature extraction
export * from "./feature_extraction/index.js";

// Multioutput
// duplicate exports from ./multioutput/index.js omitted

// Kernel ridge
// duplicate exports from ./kernel_ridge/index.js omitted

// Gaussian process
export * from "./gaussian_process/index.js";

// Kernel approximation
export * from "./kernel_approximation/index.js";

// Covariance
export * from "./covariance/index.js";

// Cross decomposition
export * from "./cross_decomposition/index.js";

// Inspection
// duplicate exports from ./inspection/index.js omitted

// Random projection
export * from "./random_projection/index.js";

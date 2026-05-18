/**
 * Estimator tags — sklearn 1.6+ tags API.
 * Provides metadata about an estimator's capabilities, input requirements, and output properties.
 * Analogous to sklearn.utils._tags.
 */

/** Tags describing required input properties. */
export interface InputTags {
  /** Whether the estimator can handle one-dimensional input arrays. */
  oneDArray: boolean;
  /** Whether the estimator supports 2D nd-array inputs. */
  twoDArray: boolean;
  /** Whether the estimator supports sparse matrix inputs. */
  sparse: boolean;
  /** Whether the estimator allows NaN/Inf values in input. */
  allowNan: boolean;
  /** Whether the estimator supports string-typed feature arrays. */
  strings: boolean;
  /** Positive-only input constraint (e.g. NMF, count models). */
  positiveOnly: boolean;
  /** Whether the estimator requires pairwise input (e.g. kernel matrices). */
  pairwise: boolean;
}

/** Tags describing target properties. */
export interface TargetTags {
  /** Whether the estimator is supervised (requires y). */
  required: boolean;
  /** Whether single-column targets are accepted. */
  oneDimensional: boolean;
  /** Whether 2D multi-output targets are accepted. */
  twoD: boolean;
  /** Accepted target types: "binary", "multiclass", "continuous", etc. */
  targetTypes: string[];
}

/** Tags describing classifier-specific properties. */
export interface ClassifierTags {
  /** Whether this is a meta-estimator (wraps another estimator). */
  meta: boolean;
  /** Whether the estimator supports multi-label classification. */
  multiLabel: boolean;
  /** Whether the estimator supports multi-output classification. */
  multiOutput: boolean;
  /** Whether the estimator supports predict_proba. */
  calibratable: boolean;
  /** Whether decision_function is always available. */
  hasDecisionFunction: boolean;
  /** Supported number of classes (0 = any). */
  poorScore: boolean;
}

/** Tags describing regressor-specific properties. */
export interface RegressorTags {
  /** Whether the estimator supports multi-output regression. */
  multiOutput: boolean;
  /** Whether predictions are non-negative. */
  positiveOnly: boolean;
  /** Whether the estimator may return poor R² scores (sanity check marker). */
  poorScore: boolean;
}

/** Tags describing transformer-specific properties. */
export interface TransformerTags {
  /** Whether transform is the identity (passthrough). */
  preservesDataType: boolean;
  /** Whether the transformer changes the number of samples. */
  changesSampleCount: boolean;
  /** Whether the transformer changes the number of features. */
  changesFeaturesCount: boolean;
  /** Whether this is a pairwise transformer (computes kernel/distance matrix). */
  pairwise: boolean;
}

/** Combined estimator tags object. */
export interface EstimatorTags {
  estimatorType: "classifier" | "regressor" | "transformer" | "clusterer" | "other";
  input: InputTags;
  target: TargetTags;
  classifier?: ClassifierTags;
  regressor?: RegressorTags;
  transformer?: TransformerTags;
  /** Whether the estimator requires fitting before transform/predict. */
  requiresFit: boolean;
  /** Whether the estimator is stateless (can call transform before fit). */
  noValidation: boolean;
  /** Arbitrary extra tags for custom estimators. */
  extra: Record<string, boolean | string | number>;
}

/** Default input tags (conservative: only standard 2D float arrays). */
export function defaultInputTags(overrides: Partial<InputTags> = {}): InputTags {
  return {
    oneDArray: false,
    twoDArray: true,
    sparse: false,
    allowNan: false,
    strings: false,
    positiveOnly: false,
    pairwise: false,
    ...overrides,
  };
}

/** Default target tags. */
export function defaultTargetTags(overrides: Partial<TargetTags> = {}): TargetTags {
  return {
    required: true,
    oneDimensional: true,
    twoD: false,
    targetTypes: ["binary", "multiclass", "continuous"],
    ...overrides,
  };
}

/** Builds a complete EstimatorTags object with sensible defaults. */
export function buildTags(
  estimatorType: EstimatorTags["estimatorType"],
  overrides: Partial<EstimatorTags> = {},
): EstimatorTags {
  return {
    estimatorType,
    input: defaultInputTags(),
    target: defaultTargetTags(),
    requiresFit: true,
    noValidation: false,
    extra: {},
    ...overrides,
  };
}

/** Type guard: returns true if the tags object belongs to a classifier. */
export function isClassifierTags(tags: EstimatorTags): tags is EstimatorTags & { classifier: ClassifierTags } {
  return tags.estimatorType === "classifier" && tags.classifier !== undefined;
}

/** Type guard: returns true if the tags object belongs to a regressor. */
export function isRegressorTags(tags: EstimatorTags): tags is EstimatorTags & { regressor: RegressorTags } {
  return tags.estimatorType === "regressor" && tags.regressor !== undefined;
}

/** Type guard: returns true if the tags object belongs to a transformer. */
export function isTransformerTags(tags: EstimatorTags): tags is EstimatorTags & { transformer: TransformerTags } {
  return tags.estimatorType === "transformer" && tags.transformer !== undefined;
}

/**
 * Exceptions used throughout tsikit-learn.
 * Mirrors sklearn.exceptions.
 */

/** Raised when an estimator is used before being fitted. */
export class NotFittedError extends Error {
  override readonly name = "NotFittedError";
  constructor(
    message = "This estimator is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.",
  ) {
    super(message);
  }
}

/** Warning raised when convergence is not reached. */
export class ConvergenceWarning extends Error {
  override readonly name = "ConvergenceWarning";
}

/** Raised when an invalid value is encountered. */
export class ValueError extends Error {
  override readonly name = "ValueError";
}

/** Raised when feature dimensions don't match. */
export class DataDimensionalityWarning extends Error {
  override readonly name = "DataDimensionalityWarning";
}

/** Raised when an undefined parameter is encountered. */
export class UndefinedMetricWarning extends Error {
  override readonly name = "UndefinedMetricWarning";
}

/** Raised when a change or metric is not positive. */
export class EfficiencyWarning extends Error {
  override readonly name = "EfficiencyWarning";
}

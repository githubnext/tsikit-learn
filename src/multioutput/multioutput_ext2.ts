/**
 * Additional multioutput type aliases.
 * Mirrors sklearn.multioutput extras.
 */

export type BaseRegressor = {
  fit(X: Float64Array[], y: Float64Array): BaseRegressor;
  predict(X: Float64Array[]): Float64Array;
};

export type BaseClassifier = {
  fit(X: Float64Array[], y: Int32Array): BaseClassifier;
  predict(X: Float64Array[]): Int32Array;
};

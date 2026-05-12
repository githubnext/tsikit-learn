/**
 * Base classes for all estimators.
 * Mirrors sklearn.base.
 */

import { NotFittedError } from "./exceptions.js";

export type Params = Record<string, unknown>;

/**
 * Base class for all scikit-learn estimators.
 * Provides get_params / set_params following sklearn conventions.
 */
export abstract class BaseEstimator {
  /**
   * Get parameters for this estimator.
   * Returns own enumerable string-keyed properties that are not functions.
   */
  get_params(deep = true): Params {
    const out: Params = {};
    for (const key of Object.keys(this)) {
      const val = (this as Record<string, unknown>)[key];
      if (typeof val !== "function") {
        out[key] = deep && val instanceof BaseEstimator ? val.get_params(deep) : val;
      }
    }
    return out;
  }

  /** Set the parameters of this estimator. */
  set_params(params: Params): this {
    for (const [key, val] of Object.entries(params)) {
      (this as Record<string, unknown>)[key] = val;
    }
    return this;
  }

  /** Assert the estimator is fitted. */
  protected _check_is_fitted(attributes: string[]): void {
    const missing = attributes.filter((a) => (this as Record<string, unknown>)[a] === undefined);
    if (missing.length > 0) {
      throw new NotFittedError(
        `This ${this.constructor.name} instance is not fitted yet. Call 'fit' first.`,
      );
    }
  }
}

/** Mixin class for all classifiers. */
export abstract class ClassifierMixin {
  readonly _estimator_type = "classifier" as const;

  /** Return the mean accuracy on the given test data and labels. */
  score(X: Float64Array[], y: Float64Array | Int32Array): number {
    const yPred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      if ((yPred[i] ?? 0) === (y[i] ?? 0)) correct++;
    }
    return y.length > 0 ? correct / y.length : 0;
  }

  abstract predict(X: Float64Array[]): Int32Array | Float64Array;
}

/** Mixin class for all regressors. */
export abstract class RegressorMixin {
  readonly _estimator_type = "regressor" as const;

  /** Return the coefficient of determination R² of the prediction. */
  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    const yMean = Array.from(y).reduce((a, b) => a + b, 0) / y.length;
    let ssTot = 0;
    let ssRes = 0;
    for (let i = 0; i < y.length; i++) {
      const yi = y[i] ?? 0;
      const pi = yPred[i] ?? 0;
      ssTot += (yi - yMean) ** 2;
      ssRes += (yi - pi) ** 2;
    }
    return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  }

  abstract predict(X: Float64Array[]): Float64Array;
}

/** Mixin class for all transformers. */
export abstract class TransformerMixin {
  readonly _estimator_type = "transformer" as const;

  /** Fit and transform in one step. */
  fit_transform(X: Float64Array[], y?: Float64Array | Int32Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }

  abstract fit(X: Float64Array[], y?: Float64Array | Int32Array): this;
  abstract transform(X: Float64Array[]): Float64Array[];
}

/** Mixin class for all clusterers. */
export abstract class ClusterMixin {
  readonly _estimator_type = "clusterer" as const;

  /** Perform clustering on X and return cluster labels. */
  fit_predict(X: Float64Array[], y?: Float64Array | Int32Array): Int32Array {
    return this.fit(X, y).labels_ ?? new Int32Array(X.length);
  }

  abstract fit(X: Float64Array[], y?: Float64Array | Int32Array): this;
  labels_?: Int32Array;
}

/** Clone an estimator with the same parameters. */
export function clone<T extends BaseEstimator>(estimator: T): T {
  const Cls = estimator.constructor as new () => T;
  const newEst = new Cls();
  newEst.set_params(estimator.get_params(false));
  return newEst;
}

/** Check if an estimator is fitted by looking for a trailing underscore attribute. */
export function check_is_fitted(estimator: BaseEstimator, attributes?: string[]): void {
  const attrs = attributes ?? Object.keys(estimator).filter((k) => k.endsWith("_") && !k.startsWith("_"));
  if (attrs.length === 0) {
    throw new NotFittedError(
      `This ${estimator.constructor.name} instance is not fitted yet.`,
    );
  }
  const missing = attrs.filter((a) => (estimator as unknown as Record<string, unknown>)[a] === undefined);
  if (missing.length > 0) {
    throw new NotFittedError(
      `This ${estimator.constructor.name} instance is not fitted yet. Missing attributes: ${missing.join(", ")}.`,
    );
  }
}

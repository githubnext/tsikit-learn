/**
 * Utilities for building meta-estimators.
 * Mirrors sklearn.utils.metaestimators.
 */

import { BaseEstimator } from "../base.js";

/**
 * Mixin class for all meta-estimators in scikit-learn.
 * A meta-estimator is an estimator that takes other estimators as parameters.
 */
export class MetaEstimatorMixin extends BaseEstimator {
  /** The inner estimator */
  estimator?: BaseEstimator;
}

/**
 * Base class for compositions of estimators.
 * Provides get_params/set_params that handle nested estimators by name.
 */
export abstract class _BaseComposition extends BaseEstimator {
  /** Named steps/estimators */
  protected abstract _get_named_estimators(): Array<[string, BaseEstimator]>;

  override get_params(deep = true): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    for (const [name, estimator] of this._get_named_estimators()) {
      out[name] = estimator;
      if (deep) {
        const nested = estimator.get_params(deep);
        for (const [k, v] of Object.entries(nested)) {
          out[`${name}__${k}`] = v;
        }
      }
    }
    return out;
  }

  override set_params(params: Record<string, unknown>): this {
    const nested: Record<string, Record<string, unknown>> = {};
    for (const [key, val] of Object.entries(params)) {
      const idx = key.indexOf("__");
      if (idx !== -1) {
        const name = key.slice(0, idx);
        const sub = key.slice(idx + 2);
        if (!nested[name]) nested[name] = {};
        nested[name]![sub] = val;
      } else {
        (this as Record<string, unknown>)[key] = val;
      }
    }
    for (const [name, subParams] of Object.entries(nested)) {
      const est = (this as Record<string, unknown>)[name];
      if (est instanceof BaseEstimator) {
        est.set_params(subParams);
      }
    }
    return this;
  }
}

/**
 * A decorator/helper that makes a method only available if a condition holds.
 * In TypeScript this is implemented as a wrapper function factory.
 *
 * @param check - A function that receives the estimator and returns true/false.
 * @param method - The method to wrap.
 * @returns The same method, but throws if check returns false.
 */
export function available_if<T extends BaseEstimator>(
  check: (estimator: T) => boolean,
): (
  target: T,
  propertyKey: string,
  descriptor: PropertyDescriptor,
) => PropertyDescriptor {
  return function (
    _target: T,
    _propertyKey: string,
    descriptor: PropertyDescriptor,
  ): PropertyDescriptor {
    const original = descriptor.value as (...args: unknown[]) => unknown;
    descriptor.value = function (this: T, ...args: unknown[]) {
      if (!check(this)) {
        throw new Error(
          `This method is not available because the check condition is not met.`,
        );
      }
      return original.apply(this, args);
    };
    return descriptor;
  };
}

/**
 * Wraps a method of an estimator to delegate to a named sub-estimator.
 * Mirrors sklearn.utils.metaestimators.if_delegate_has_method.
 */
export function if_delegate_has_method(
  delegate: string,
  methodName: string,
): (
  target: BaseEstimator,
  propertyKey: string,
  descriptor: PropertyDescriptor,
) => PropertyDescriptor {
  return function (
    _target: BaseEstimator,
    _propertyKey: string,
    descriptor: PropertyDescriptor,
  ): PropertyDescriptor {
    const original = descriptor.value as (...args: unknown[]) => unknown;
    descriptor.value = function (
      this: BaseEstimator & Record<string, unknown>,
      ...args: unknown[]
    ) {
      const delegateObj = this[delegate];
      if (
        !delegateObj ||
        typeof (delegateObj as Record<string, unknown>)[methodName] !==
          "function"
      ) {
        throw new Error(
          `This estimator does not have a '${methodName}' method ` +
            `because its delegate '${delegate}' does not support it.`,
        );
      }
      return original.apply(this, args);
    };
    return descriptor;
  };
}

/**
 * Check if a fitted estimator has a specific method.
 */
export function hasMethod(
  estimator: BaseEstimator,
  method: string,
): boolean {
  return typeof (estimator as unknown as Record<string, unknown>)[method] === "function";
}

/**
 * Returns the method of `estimator` if it exists, else raises an error.
 * Useful for meta-estimators that conditionally delegate methods.
 */
export function check_is_fitted_has_method<T extends BaseEstimator>(
  estimator: T,
  method: string,
): void {
  if (typeof (estimator as unknown as Record<string, unknown>)[method] !== "function") {
    throw new Error(
      `${estimator.constructor.name} does not implement '${method}'.`,
    );
  }
}

/**
 * A wrapper estimator that delegates all calls to the wrapped estimator.
 * Base class for delegating wrappers like Pipeline.
 */
export abstract class _DelegatingMixin extends MetaEstimatorMixin {
  protected _wrap_predict<T>(methodName: string, X: T): unknown {
    if (!this.estimator) {
      throw new Error("No estimator set.");
    }
    const fn = (this.estimator as unknown as Record<string, unknown>)[methodName];
    if (typeof fn !== "function") {
      throw new Error(
        `The estimator ${this.estimator.constructor.name} does not implement '${methodName}'.`,
      );
    }
    return (fn as (x: T) => unknown).call(this.estimator, X);
  }
}

/**
 * Estimator compatibility checks — lightweight TypeScript port of
 * sklearn's check_estimator utilities.
 *
 * Ports: check_estimator (structural duck-typing), parametrize_with_checks,
 *        is_classifier, is_regressor, is_transformer, is_clusterer,
 *        check_is_fitted, set_config, get_config
 */

/** Minimal estimator interface (duck-typed). */
export interface EstimatorLike {
  fit(...args: unknown[]): unknown;
  getParams?(): Record<string, unknown>;
  setParams?(params: Record<string, unknown>): void;
}

/** Check whether an object looks like a fitted estimator. */
export function checkIsFitted(estimator: object, attributes?: string[]): void {
  if (attributes) {
    const missing = attributes.filter(
      (a) =>
        !(a in estimator) ||
        (estimator as Record<string, unknown>)[a] === undefined,
    );
    if (missing.length > 0) {
      throw new Error(
        `Estimator is not fitted. Missing attributes: ${missing.join(", ")}`,
      );
    }
    return;
  }
  // Look for any fitted attribute (ending with _)
  const fitted = Object.keys(estimator).some((k) => k.endsWith("_"));
  if (!fitted) {
    throw new Error(
      `Estimator ${estimator.constructor.name} is not fitted. Call fit() first.`,
    );
  }
}

/** Returns true if the estimator has a predict method and is a classifier. */
export function isClassifier(estimator: object): boolean {
  return (
    "predict" in estimator &&
    ("classes_" in estimator ||
      estimator.constructor.name.toLowerCase().includes("classifier"))
  );
}

/** Returns true if the estimator is a regressor. */
export function isRegressor(estimator: object): boolean {
  return (
    "predict" in estimator &&
    !isClassifier(estimator) &&
    !("transform" in estimator)
  );
}

/** Returns true if the estimator has a transform method. */
export function isTransformer(estimator: object): boolean {
  return "transform" in estimator || "fitTransform" in estimator;
}

/** Returns true if the estimator is a clusterer. */
export function isClusterer(estimator: object): boolean {
  return (
    "fitPredict" in estimator ||
    "labels_" in estimator ||
    estimator.constructor.name.toLowerCase().includes("cluster") ||
    estimator.constructor.name.toLowerCase().includes("kmeans") ||
    estimator.constructor.name.toLowerCase().includes("dbscan")
  );
}

export interface CheckResult {
  passed: boolean;
  errors: string[];
}

/**
 * Structural duck-type check for a minimal estimator interface.
 * In Python sklearn this runs 100+ checks; here we verify the core API.
 */
export function checkEstimator(estimator: object): CheckResult {
  const errors: string[] = [];

  if (!("fit" in estimator)) {
    errors.push("Missing required method: fit()");
  }

  if (
    "getParams" in estimator &&
    typeof (estimator as EstimatorLike).getParams === "function"
  ) {
    try {
      const params = (estimator as EstimatorLike).getParams?.() ?? {};
      if (typeof params !== "object") {
        errors.push("getParams() must return an object");
      }
    } catch (e) {
      errors.push(`getParams() threw: ${String(e)}`);
    }
  }

  if (
    "setParams" in estimator &&
    typeof (estimator as EstimatorLike).setParams === "function"
  ) {
    const params = (estimator as EstimatorLike).getParams?.() ?? {};
    try {
      (estimator as EstimatorLike).setParams?.(params);
    } catch (e) {
      errors.push(`setParams() threw: ${String(e)}`);
    }
  }

  return { passed: errors.length === 0, errors };
}

/**
 * Returns an array of [estimator, checkFn] pairs for use with test runners.
 * Each check is a function that throws if the check fails.
 */
export function parametrizeWithChecks(
  estimators: object[],
): Array<{ estimator: object; check: (est: object) => void; name: string }> {
  const checks: Array<{
    estimator: object;
    check: (est: object) => void;
    name: string;
  }> = [];
  for (const est of estimators) {
    checks.push({
      estimator: est,
      name: `check_estimator[${est.constructor.name}]`,
      check: (e: object) => {
        const result = checkEstimator(e);
        if (!result.passed) {
          throw new Error(result.errors.join("; "));
        }
      },
    });
    checks.push({
      estimator: est,
      name: `check_is_classifier_or_regressor_or_transformer[${est.constructor.name}]`,
      check: (e: object) => {
        const ok =
          isClassifier(e) ||
          isRegressor(e) ||
          isTransformer(e) ||
          isClusterer(e);
        if (!ok) {
          throw new Error(
            `${e.constructor.name} is not recognized as a classifier, regressor, transformer, or clusterer`,
          );
        }
      },
    });
  }
  return checks;
}

/** Global configuration store. */
const _config: Record<string, unknown> = {
  assumeFinite: false,
  workingMemory: 1024,
  printChangedOnly: true,
  displayDiagram: "on",
};

/** Get the current estimator checks configuration. */
export function getChecksConfig(): Record<string, unknown> {
  return { ..._config };
}

/** Set estimator checks configuration options. */
export function setChecksConfig(options: Record<string, unknown>): void {
  for (const [k, v] of Object.entries(options)) {
    _config[k] = v;
  }
}

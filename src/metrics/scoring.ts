/**
 * Scorer utilities for model evaluation.
 * Mirrors scikit-learn's metrics._scorer.
 */

export type ScorerFn = (
  estimator: { predict: (X: Float64Array[]) => Float64Array | Int32Array },
  X: Float64Array[],
  y: Float64Array | Int32Array,
) => number;

export interface ScorerOptions {
  greaterIsBetter?: boolean;
  responseMethod?: "predict" | "predict_proba" | "decision_function";
}

/**
 * Make a scorer from a metric function.
 */
export function makeScorer(
  scoreFn: (
    yTrue: Float64Array | Int32Array,
    yPred: Float64Array | Int32Array,
  ) => number,
  options: ScorerOptions = {},
): ScorerFn {
  const { greaterIsBetter = true } = options;
  const sign = greaterIsBetter ? 1 : -1;
  return (estimator, X, y) => {
    const yPred = estimator.predict(X);
    return sign * scoreFn(y, yPred);
  };
}

const SCORERS: Record<string, ScorerFn> = {};

/**
 * Register a named scorer.
 */
export function registerScorer(name: string, scorer: ScorerFn): void {
  SCORERS[name] = scorer;
}

/**
 * Get a scorer by name.
 */
export function getScorer(name: string): ScorerFn {
  const scorer = SCORERS[name];
  if (scorer === undefined) {
    throw new Error(
      `Unknown scorer: '${name}'. Available: ${Object.keys(SCORERS).join(", ")}`,
    );
  }
  return scorer;
}

/**
 * Check that a scoring parameter is valid and return a scorer function.
 */
export function checkScoring(
  estimator: unknown,
  scoring: string | ScorerFn | null | undefined,
): ScorerFn {
  if (scoring === null || scoring === undefined) {
    // Default scorer based on estimator type
    if (
      estimator !== null &&
      typeof estimator === "object" &&
      "score" in estimator &&
      typeof (estimator as { score: unknown }).score === "function"
    ) {
      return (est, X, y) =>
        (
          est as unknown as {
            score: (X: Float64Array[], y: Float64Array | Int32Array) => number;
          }
        ).score(X, y);
    }
    throw new Error(
      "scoring must be provided when estimator has no default score method",
    );
  }
  if (typeof scoring === "string") {
    return getScorer(scoring);
  }
  return scoring;
}

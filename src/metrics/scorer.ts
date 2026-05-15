/**
 * Scoring utilities: make_scorer, check_scoring, get_scorer.
 * Mirrors sklearn.metrics._scorer.
 */

import { NotFittedError } from "../exceptions.js";

/** A scorer callable that wraps a metric function. */
export interface Scorer {
  (estimator: Estimator, X: Float64Array[], y: Float64Array | Int32Array): number;
  _sign: number;
  _scoreFn: MetricFn;
  _kwargs: Record<string, unknown>;
}

type MetricFn = (
  yTrue: Float64Array | Int32Array,
  yPred: Float64Array | Int32Array,
  ...args: unknown[]
) => number;

type Estimator = {
  predict?: (X: Float64Array[]) => Float64Array | Int32Array;
  predictProba?: (X: Float64Array[]) => Float64Array[];
  decisionFunction?: (X: Float64Array[]) => Float64Array;
  score?: (X: Float64Array[], y: Float64Array | Int32Array) => number;
};

/**
 * Create a scorer from a metric function.
 * Mirrors sklearn.metrics.make_scorer.
 */
export function makeScorer(
  scoreFn: MetricFn,
  options: {
    greaterIsBetter?: boolean;
    needsProba?: boolean;
    needsThreshold?: boolean;
    kwargs?: Record<string, unknown>;
  } = {},
): Scorer {
  const {
    greaterIsBetter = true,
    needsProba = false,
    needsThreshold = false,
    kwargs = {},
  } = options;

  const sign = greaterIsBetter ? 1 : -1;

  const scorer = (
    estimator: Estimator,
    X: Float64Array[],
    y: Float64Array | Int32Array,
  ): number => {
    let yPred: Float64Array | Int32Array;

    if (needsProba && estimator.predictProba) {
      const proba = estimator.predictProba(X);
      // Use last column for binary, or pass all probas
      yPred = new Float64Array(proba.map((row) => row[row.length - 1] ?? 0));
    } else if (needsThreshold && estimator.decisionFunction) {
      yPred = estimator.decisionFunction(X);
    } else if (estimator.predict) {
      yPred = estimator.predict(X);
    } else {
      throw new NotFittedError("Estimator");
    }

    return sign * scoreFn(y, yPred, kwargs);
  };

  (scorer as Scorer)._sign = sign;
  (scorer as Scorer)._scoreFn = scoreFn;
  (scorer as Scorer)._kwargs = kwargs;

  return scorer as Scorer;
}

/** Built-in scoring metric functions. */

/** Mean squared error (negated for scoring). */
function _mseFn(yTrue: Float64Array | Int32Array, yPred: Float64Array | Int32Array): number {
  let s = 0;
  for (let i = 0; i < yTrue.length; i++) s += ((yTrue[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
  return s / yTrue.length;
}

/** Mean absolute error. */
function _maeFn(yTrue: Float64Array | Int32Array, yPred: Float64Array | Int32Array): number {
  let s = 0;
  for (let i = 0; i < yTrue.length; i++) s += Math.abs((yTrue[i] ?? 0) - (yPred[i] ?? 0));
  return s / yTrue.length;
}

/** R² score. */
function _r2Fn(yTrue: Float64Array | Int32Array, yPred: Float64Array | Int32Array): number {
  const mean = Array.from(yTrue).reduce((a, b) => a + b, 0) / yTrue.length;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < yTrue.length; i++) {
    ssRes += ((yTrue[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
    ssTot += ((yTrue[i] ?? 0) - mean) ** 2;
  }
  return ssTot > 0 ? 1 - ssRes / ssTot : 0;
}

/** Accuracy score. */
function _accuracyFn(yTrue: Float64Array | Int32Array, yPred: Float64Array | Int32Array): number {
  let correct = 0;
  for (let i = 0; i < yTrue.length; i++) if ((yTrue[i] ?? 0) === (yPred[i] ?? 0)) correct++;
  return correct / yTrue.length;
}

/** F1 score (binary). */
function _f1Fn(yTrue: Float64Array | Int32Array, yPred: Float64Array | Int32Array): number {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const t = yTrue[i] ?? 0;
    const p = yPred[i] ?? 0;
    if (t === 1 && p === 1) tp++;
    else if (t === 0 && p === 1) fp++;
    else if (t === 1 && p === 0) fn++;
  }
  const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
  const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
  return prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0;
}

/** Registry of built-in scorers. */
const _SCORERS: Record<string, Scorer> = {
  r2: makeScorer(_r2Fn),
  neg_mean_squared_error: makeScorer(_mseFn, { greaterIsBetter: false }),
  neg_mean_absolute_error: makeScorer(_maeFn, { greaterIsBetter: false }),
  accuracy: makeScorer(_accuracyFn),
  f1: makeScorer(_f1Fn),
};

/**
 * Get a scorer by name or pass-through if already a Scorer.
 * Mirrors sklearn.metrics.check_scoring / get_scorer.
 */
export function checkScoring(
  estimator: Estimator,
  scoring?: string | Scorer | null,
): Scorer {
  if (scoring === null || scoring === undefined) {
    // Use estimator's default score method
    const defaultScorer = (
      est: Estimator,
      X: Float64Array[],
      y: Float64Array | Int32Array,
    ): number => {
      if (!est.score) throw new NotFittedError("Estimator has no score method");
      return est.score(X, y);
    };
    (defaultScorer as Scorer)._sign = 1;
    (defaultScorer as Scorer)._scoreFn = _r2Fn;
    (defaultScorer as Scorer)._kwargs = {};
    return defaultScorer as Scorer;
  }

  if (typeof scoring === "string") {
    const s = _SCORERS[scoring];
    if (!s) throw new Error(`Unknown scorer: ${scoring}. Available: ${Object.keys(_SCORERS).join(", ")}`);
    return s;
  }

  return scoring;
}

/**
 * Get a scorer by name.
 * Mirrors sklearn.metrics.get_scorer.
 */
export function getScorer(name: string): Scorer {
  const s = _SCORERS[name];
  if (!s) throw new Error(`Unknown scorer: ${name}. Available: ${Object.keys(_SCORERS).join(", ")}`);
  return s;
}

/**
 * Get available scorer names.
 * Mirrors sklearn.metrics.get_scorer_names.
 */
export function getScorerNames(): string[] {
  return Object.keys(_SCORERS);
}

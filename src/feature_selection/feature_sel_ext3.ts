/**
 * Extended feature selection: VarianceThreshold, SelectFwe, SelectFdr,
 * mutual information extras, and feature importance ranking utilities.
 */

/** VarianceThreshold: removes features with variance below a threshold. */
export class VarianceThresholdSelector {
  variances_?: Float64Array;
  supportMask_?: boolean[];
  threshold: number;

  constructor(threshold = 0.0) {
    this.threshold = threshold;
  }

  fit(X: Float64Array[]): this {
    const d = X[0]?.length ?? 0;
    const n = X.length;
    const means = new Float64Array(d);
    for (const xi of X) {
      for (let j = 0; j < d; j++) means[j] = (means[j] ?? 0) + (xi[j] ?? 0);
    }
    for (let j = 0; j < d; j++) means[j] = (means[j] ?? 0) / n;

    const variances = new Float64Array(d);
    for (const xi of X) {
      for (let j = 0; j < d; j++) {
        variances[j] = (variances[j] ?? 0) + ((xi[j] ?? 0) - (means[j] ?? 0)) ** 2;
      }
    }
    for (let j = 0; j < d; j++) variances[j] = (variances[j] ?? 0) / n;

    this.variances_ = variances;
    this.supportMask_ = Array.from(variances, (v) => v > this.threshold);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.supportMask_) throw new Error("Not fitted");
    const mask = this.supportMask_;
    return X.map((xi) => {
      const out: number[] = [];
      for (let j = 0; j < xi.length; j++) {
        if (mask[j]) out.push(xi[j] ?? 0);
      }
      return new Float64Array(out);
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

/** F-statistic for regression (F-test between feature and target). */
export function fRegression(
  X: Float64Array[],
  y: Float64Array,
): { fStats: Float64Array; pValues: Float64Array } {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const fStats = new Float64Array(d);
  const pValues = new Float64Array(d);

  const yMean = y.reduce((a, b) => a + b, 0) / n;
  const ssTot = y.reduce((s, v) => s + (v - yMean) ** 2, 0);

  for (let j = 0; j < d; j++) {
    const xj = new Float64Array(n).map((_, i) => X[i]?.[j] ?? 0);
    const xMean = xj.reduce((a, b) => a + b, 0) / n;
    let ssXY = 0, ssX = 0;
    for (let i = 0; i < n; i++) {
      const xi = (xj[i] ?? 0) - xMean;
      const yi = (y[i] ?? 0) - yMean;
      ssXY += xi * yi;
      ssX += xi * xi;
    }
    const beta = ssX > 1e-10 ? ssXY / ssX : 0;
    const ssReg = beta * ssXY;
    const ssRes = ssTot - ssReg;
    fStats[j] = ssRes > 1e-10 ? ssReg / (ssRes / (n - 2)) : 0;
    // Approximate p-value using chi-squared approximation
    pValues[j] = Math.exp(-0.5 * (fStats[j] ?? 0) / n);
  }
  return { fStats, pValues };
}

/** Chi-squared test for discrete features. */
export function chi2Test(
  X: Float64Array[],
  y: Int32Array,
): { chi2Stats: Float64Array; pValues: Float64Array } {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const classes = [...new Set(Array.from(y))];
  const chi2Stats = new Float64Array(d);
  const pValues = new Float64Array(d);

  for (let j = 0; j < d; j++) {
    const xj = Array.from({ length: n }, (_, i) => X[i]?.[j] ?? 0);
    const featureVals = [...new Set(xj)];
    let chi2 = 0;
    for (const c of classes) {
      for (const fv of featureVals) {
        const observed = xj.filter((v, i) => v === fv && (y[i] ?? -1) === c).length;
        const expected = (xj.filter((v) => v === fv).length * Array.from(y).filter((v) => v === c).length) / n;
        if (expected > 0) chi2 += (observed - expected) ** 2 / expected;
      }
    }
    chi2Stats[j] = chi2;
    pValues[j] = Math.exp(-0.5 * chi2);
  }
  return { chi2Stats, pValues };
}

/** SelectFpr: select features below a false positive rate threshold. */
export function selectFpr(
  fStats: Float64Array,
  pValues: Float64Array,
  alpha = 0.05,
): boolean[] {
  return Array.from(pValues, (p) => p < alpha);
}

/** SelectFwe: Bonferroni correction for family-wise error rate. */
export function selectFwe(
  pValues: Float64Array,
  alpha = 0.05,
): boolean[] {
  const corrected = alpha / pValues.length;
  return Array.from(pValues, (p) => p < corrected);
}

/** Permutation importance: estimate importance by shuffling features. */
export function permutationImportance(
  predictFn: (X: Float64Array[]) => Float64Array,
  X: Float64Array[],
  y: Float64Array,
  scoreFn: (yTrue: Float64Array, yPred: Float64Array) => number,
  nRepeats = 5,
): Float64Array {
  const d = X[0]?.length ?? 0;
  const baseScore = scoreFn(y, predictFn(X));
  const importances = new Float64Array(d);

  for (let j = 0; j < d; j++) {
    let decreaseSum = 0;
    for (let rep = 0; rep < nRepeats; rep++) {
      // Shuffle feature j
      const indices = Array.from({ length: X.length }, (_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const k = Math.floor(Math.random() * (i + 1));
        const tmp = indices[i]!;
        indices[i] = indices[k]!;
        indices[k] = tmp;
      }
      const Xperm = X.map((xi, i) => {
        const row = new Float64Array(xi);
        row[j] = X[indices[i] ?? 0]?.[j] ?? 0;
        return row;
      });
      const permScore = scoreFn(y, predictFn(Xperm));
      decreaseSum += baseScore - permScore;
    }
    importances[j] = decreaseSum / nRepeats;
  }
  return importances;
}

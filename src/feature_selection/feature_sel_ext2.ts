/**
 * Extended feature selection: chi2, reliefF, GenericUnivariateSelectExt, SelectFwe
 */

export function chi2(X: Float64Array[], y: Int32Array): { scores: Float64Array; pValues: Float64Array } {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  const classes = [...new Set(Array.from(y))].sort((a, b) => a - b);
  const scores = new Float64Array(p);
  const pValues = new Float64Array(p);

  for (let j = 0; j < p; j++) {
    // Compute contingency table
    const observed: Float64Array[] = classes.map(() => new Float64Array(2));
    for (let i = 0; i < n; i++) {
      const ci = classes.indexOf(y[i] ?? 0);
      const val = (X[i]![j] ?? 0) > 0 ? 1 : 0;
      observed[ci]![val] = (observed[ci]![val] ?? 0) + 1;
    }

    // Chi2 statistic
    const rowSums = observed.map((row) => (row[0] ?? 0) + (row[1] ?? 0));
    const colSums = [
      observed.reduce((acc, row) => acc + (row[0] ?? 0), 0),
      observed.reduce((acc, row) => acc + (row[1] ?? 0), 0),
    ];
    let chi = 0;
    for (let ci = 0; ci < classes.length; ci++) {
      for (let k = 0; k < 2; k++) {
        const expected = ((rowSums[ci] ?? 0) * (colSums[k] ?? 0)) / n;
        if (expected > 0) chi += ((observed[ci]![k] ?? 0) - expected) ** 2 / expected;
      }
    }
    scores[j] = chi;
    pValues[j] = chi2pValue(chi, classes.length - 1);
  }
  return { scores, pValues };
}

function chi2pValue(x: number, df: number): number {
  if (x <= 0) return 1;
  return Math.exp(-x / 2) * Math.pow(x / 2, df / 2 - 1) / gamma(df / 2);
}

function gamma(n: number): number {
  if (n === 0.5) return Math.sqrt(Math.PI);
  if (n === 1) return 1;
  if (n < 1) return gamma(n + 1) / n;
  return (n - 1) * gamma(n - 1);
}

export function reliefF(
  X: Float64Array[],
  y: Int32Array,
  nNeighbors = 10
): Float64Array {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  const weights = new Float64Array(p);

  for (let i = 0; i < n; i++) {
    const xi = X[i]!;
    const yi = y[i] ?? 0;

    // Find k nearest hits and misses
    const dists = Array.from({ length: n }, (_, j) => {
      if (j === i) return { j, d: Number.POSITIVE_INFINITY, sameClass: false };
      let d2 = 0;
      for (let k = 0; k < p; k++) d2 += ((xi[k] ?? 0) - (X[j]![k] ?? 0)) ** 2;
      return { j, d: Math.sqrt(d2), sameClass: (y[j] ?? -1) === yi };
    }).sort((a, b) => a.d - b.d);

    const hits = dists.filter((d) => d.sameClass).slice(0, nNeighbors);
    const misses = dists.filter((d) => !d.sameClass).slice(0, nNeighbors);

    for (let feat = 0; feat < p; feat++) {
      const xiF = xi[feat] ?? 0;
      for (const hit of hits) weights[feat]! -= Math.abs(xiF - (X[hit.j]![feat] ?? 0)) / (n * nNeighbors);
      for (const miss of misses) weights[feat]! += Math.abs(xiF - (X[miss.j]![feat] ?? 0)) / (n * nNeighbors);
    }
  }
  return weights;
}

export class GenericUnivariateSelectExt {
  private mode: "k_best" | "percentile" | "fwe" | "fdr" | "fpr";
  private param: number;
  private scoreFn: (X: Float64Array[], y: Int32Array) => { scores: Float64Array; pValues: Float64Array };
  scores_: Float64Array | null = null;
  pValues_: Float64Array | null = null;
  mask_: Int32Array | null = null;

  constructor(
    scoreFn: (X: Float64Array[], y: Int32Array) => { scores: Float64Array; pValues: Float64Array },
    mode: "k_best" | "percentile" | "fwe" | "fdr" | "fpr" = "percentile",
    param = 10
  ) {
    this.scoreFn = scoreFn;
    this.mode = mode;
    this.param = param;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const { scores, pValues } = this.scoreFn(X, y);
    this.scores_ = scores;
    this.pValues_ = pValues;
    const p = scores.length;
    const mask = new Int32Array(p);

    if (this.mode === "k_best") {
      const k = Math.min(Math.floor(this.param), p);
      const indices = Array.from({ length: p }, (_, i) => i).sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0));
      for (let i = 0; i < k; i++) mask[indices[i]!] = 1;
    } else if (this.mode === "percentile") {
      const threshold = this.param / 100;
      const sortedScores = Float64Array.from(scores).sort((a, b) => b - a);
      const cutoff = sortedScores[Math.floor(threshold * p)] ?? 0;
      for (let j = 0; j < p; j++) if ((scores[j] ?? 0) >= cutoff) mask[j] = 1;
    } else if (this.mode === "fwe" || this.mode === "fpr") {
      for (let j = 0; j < p; j++) if ((pValues[j] ?? 1) <= this.param) mask[j] = 1;
    } else if (this.mode === "fdr") {
      // Benjamini-Hochberg
      const sorted = Array.from({ length: p }, (_, i) => i).sort((a, b) => (pValues[a] ?? 1) - (pValues[b] ?? 1));
      for (let i = 0; i < p; i++) {
        if ((pValues[sorted[i]!] ?? 1) <= (this.param * (i + 1)) / p) mask[sorted[i]!] = 1;
      }
    }

    this.mask_ = mask;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.mask_) throw new Error("Not fitted");
    const selectedFeats = Array.from(this.mask_).map((v, i) => (v === 1 ? i : -1)).filter((i) => i >= 0);
    return X.map((row) => new Float64Array(selectedFeats.map((j) => row[j] ?? 0)));
  }

  fitTransform(X: Float64Array[], y: Int32Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}

export class SelectFwe {
  private alpha: number;
  private scoreFn: (X: Float64Array[], y: Int32Array) => { scores: Float64Array; pValues: Float64Array };
  pValues_: Float64Array | null = null;
  mask_: Int32Array | null = null;

  constructor(
    scoreFn: (X: Float64Array[], y: Int32Array) => { scores: Float64Array; pValues: Float64Array },
    alpha = 0.05
  ) {
    this.scoreFn = scoreFn;
    this.alpha = alpha;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const { pValues } = this.scoreFn(X, y);
    this.pValues_ = pValues;
    const p = pValues.length;
    const threshold = this.alpha / p; // Bonferroni correction
    this.mask_ = new Int32Array(p);
    for (let j = 0; j < p; j++) if ((pValues[j] ?? 1) <= threshold) this.mask_[j] = 1;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.mask_) throw new Error("Not fitted");
    const selectedFeats = Array.from(this.mask_).map((v, i) => (v === 1 ? i : -1)).filter((i) => i >= 0);
    return X.map((row) => new Float64Array(selectedFeats.map((j) => row[j] ?? 0)));
  }
}

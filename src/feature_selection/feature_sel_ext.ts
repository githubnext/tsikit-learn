/**
 * Feature selection base classes and correlation-based selection.
 */

export abstract class SelectorMixin {
  abstract get_support(indices?: boolean): Uint8Array | Int32Array;

  transform(X: Float64Array[]): Float64Array[] {
    const support = this.get_support();
    const cols = Array.from({ length: support.length }, (_, j) => j).filter((j) => (support[j] ?? 0) !== 0);
    return X.map((row) => Float64Array.from(cols, (j) => row[j] ?? 0));
  }

  inverseTransform(X: Float64Array[], nFeaturesIn: number): Float64Array[] {
    const support = this.get_support();
    const cols: number[] = [];
    for (let j = 0; j < support.length; j++) if ((support[j] ?? 0) !== 0) cols.push(j);
    return X.map((row) => {
      const out = new Float64Array(nFeaturesIn);
      for (let i = 0; i < cols.length; i++) out[cols[i] as number] = row[i] ?? 0;
      return out;
    });
  }
}

export class CorrelationSelector extends SelectorMixin {
  threshold: number;
  method: "pearson" | "spearman" | "kendall";
  private support_: Uint8Array | null = null;
  private correlations_: Float64Array | null = null;
  nFeaturesIn_: number = 0;
  nFeaturesOut_: number = 0;

  constructor(threshold = 0.0, method: "pearson" | "spearman" | "kendall" = "pearson") {
    super();
    this.threshold = threshold;
    this.method = method;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this.nFeaturesIn_ = p;
    this.correlations_ = new Float64Array(p);
    const yMean = y.reduce((a, b) => a + b, 0) / n;
    const yVar = y.reduce((s, v) => s + (v - yMean) ** 2, 0) / n;

    for (let j = 0; j < p; j++) {
      const xCol = Float64Array.from({ length: n }, (_, i) => X[i]?.[j] ?? 0);
      const xMean = xCol.reduce((a, b) => a + b, 0) / n;
      const xVar = xCol.reduce((s, v) => s + (v - xMean) ** 2, 0) / n;
      let cov = 0;
      for (let i = 0; i < n; i++) cov += ((xCol[i] ?? 0) - xMean) * ((y[i] ?? 0) - yMean);
      cov /= n;
      const denom = Math.sqrt(xVar * yVar);
      this.correlations_[j] = denom > 1e-10 ? cov / denom : 0;
    }

    this.support_ = new Uint8Array(p).map((_, j) => Math.abs(this.correlations_?.[j] ?? 0) >= this.threshold ? 1 : 0);
    this.nFeaturesOut_ = Array.from(this.support_).filter(Boolean).length;
    return this;
  }

  get_support(indices = false): Uint8Array | Int32Array {
    if (!this.support_) throw new Error("Not fitted");
    if (indices) return Int32Array.from(Array.from({ length: this.support_.length }, (_, j) => j).filter((j) => this.support_?.[j] !== 0));
    return this.support_;
  }

  scores_(): Float64Array {
    return this.correlations_ ?? new Float64Array(0);
  }
}

export class MutualInfoSelector extends SelectorMixin {
  kBest: number;
  discrete_features: boolean;
  private support_: Uint8Array | null = null;
  private scores__: Float64Array | null = null;
  nFeaturesIn_: number = 0;

  constructor(kBest = 10, discrete_features = false) {
    super();
    this.kBest = kBest;
    this.discrete_features = discrete_features;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this.nFeaturesIn_ = p;
    this.scores__ = new Float64Array(p);

    for (let j = 0; j < p; j++) {
      const xCol = Float64Array.from({ length: n }, (_, i) => X[i]?.[j] ?? 0);
      this.scores__[j] = _mutualInfoContinuous(xCol, y);
    }

    const sortedScores = Array.from({ length: p }, (_, j) => ({ j, s: this.scores__?.[j] ?? 0 })).sort((a, b) => b.s - a.s);
    const topK = sortedScores.slice(0, Math.min(this.kBest, p)).map((x) => x.j);
    this.support_ = new Uint8Array(p).map((_, j) => topK.includes(j) ? 1 : 0);
    return this;
  }

  get_support(indices = false): Uint8Array | Int32Array {
    if (!this.support_) throw new Error("Not fitted");
    if (indices) return Int32Array.from(Array.from({ length: this.support_.length }, (_, j) => j).filter((j) => this.support_?.[j] !== 0));
    return this.support_;
  }

  scores_(): Float64Array {
    return this.scores__ ?? new Float64Array(0);
  }
}

function _mutualInfoContinuous(x: Float64Array, y: Float64Array): number {
  // Estimate mutual information via kernel density / discretization
  const n = x.length;
  const k = 3;
  let mi = 0;
  for (let i = 0; i < Math.min(n, 200); i++) {
    const xi = x[i] ?? 0;
    const yi = y[i] ?? 0;
    const dists = Array.from({ length: n }, (_, j) => Math.abs((x[j] ?? 0) - xi) + Math.abs((y[j] ?? 0) - yi));
    dists.sort((a, b) => a - b);
    const eps = dists[k] ?? 1;
    if (eps < 1e-10) continue;
    const nX = x.filter((xj) => Math.abs(xj - xi) < eps).length;
    const nY = y.filter((yj) => Math.abs(yj - yi) < eps).length;
    mi += digamma(k) - digamma(nX) - digamma(nY) + digamma(n);
  }
  return Math.max(0, mi / Math.min(n, 200));
}

function digamma(n: number): number {
  if (n <= 0) return 0;
  let result = -0.5772;
  for (let k = 1; k < n; k++) result += 1 / k;
  return result;
}

export class MaxAbsCorrelationSelector extends SelectorMixin {
  maxFeatures: number;
  private support_: Uint8Array | null = null;
  private absCorr_: Float64Array | null = null;
  nFeaturesIn_: number = 0;

  constructor(maxFeatures = 10) {
    super();
    this.maxFeatures = maxFeatures;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this.nFeaturesIn_ = p;
    const yMean = y.reduce((a, b) => a + b, 0) / n;
    const yVar = y.reduce((s, v) => s + (v - yMean) ** 2, 0);
    this.absCorr_ = new Float64Array(p);

    for (let j = 0; j < p; j++) {
      const xMean = Array.from({ length: n }, (_, i) => X[i]?.[j] ?? 0).reduce((a, b) => a + b, 0) / n;
      let cov = 0, xVar = 0;
      for (let i = 0; i < n; i++) {
        const xi = (X[i]?.[j] ?? 0) - xMean;
        cov += xi * ((y[i] ?? 0) - yMean);
        xVar += xi ** 2;
      }
      const denom = Math.sqrt(xVar * yVar);
      this.absCorr_[j] = denom > 0 ? Math.abs(cov / denom) : 0;
    }

    const sorted = Array.from({ length: p }, (_, j) => ({ j, s: this.absCorr_?.[j] ?? 0 })).sort((a, b) => b.s - a.s);
    const selected = sorted.slice(0, Math.min(this.maxFeatures, p)).map((x) => x.j);
    this.support_ = new Uint8Array(p).map((_, j) => selected.includes(j) ? 1 : 0);
    return this;
  }

  get_support(indices = false): Uint8Array | Int32Array {
    if (!this.support_) throw new Error("Not fitted");
    if (indices) return Int32Array.from(Array.from({ length: this.support_.length }, (_, j) => j).filter((j) => this.support_?.[j] !== 0));
    return this.support_;
  }
}

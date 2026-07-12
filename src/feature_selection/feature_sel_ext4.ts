/**
 * FeatureHasher and SelectFwe feature selection extensions.
 */

export class FeatureHasher {
  nFeatures: number;
  alternateSign: boolean;
  inputType: "dict" | "pair";

  constructor(nFeatures = 1048576, alternateSign = true, inputType: "dict" | "pair" = "dict") {
    this.nFeatures = nFeatures;
    this.alternateSign = alternateSign;
    this.inputType = inputType;
  }

  private _fnv1aHash(s: string): number {
    let h = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = (h * 0x01000193) >>> 0;
    }
    return h;
  }

  transformDicts(dicts: Array<Record<string, number>>): Float64Array[] {
    return dicts.map((d) => {
      const row = new Float64Array(this.nFeatures);
      for (const [key, value] of Object.entries(d)) {
        const hash = this._fnv1aHash(key);
        const idx = hash % this.nFeatures;
        const sign = this.alternateSign && (hash >>> 31) ? -1 : 1;
        row[idx]! += sign * value;
      }
      return row;
    });
  }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map((row) => {
      const out = new Float64Array(this.nFeatures);
      for (let j = 0; j < row.length; j++) {
        const idx = j % this.nFeatures;
        out[idx]! += row[j] ?? 0;
      }
      return out;
    });
  }

  fit(_X: Float64Array[]): this {
    return this;
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class SelectFwe {
  alpha: number;
  scoreFn: (X: Float64Array[], y: Float64Array | Int32Array) => { scores: Float64Array; pValues: Float64Array };
  scores_: Float64Array | null = null;
  pValues_: Float64Array | null = null;
  private _support: Uint8Array | null = null;
  nFeaturesIn_: number = 0;

  constructor(
    scoreFn: (X: Float64Array[], y: Float64Array | Int32Array) => { scores: Float64Array; pValues: Float64Array },
    alpha = 0.05,
  ) {
    this.alpha = alpha;
    this.scoreFn = scoreFn;
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    const p = X[0]?.length ?? 0;
    this.nFeaturesIn_ = p;
    const { scores, pValues } = this.scoreFn(X, y);
    this.scores_ = scores;
    this.pValues_ = pValues;
    // Bonferroni correction (FWE)
    const alphaCorr = this.alpha / p;
    this._support = Uint8Array.from(pValues, (pv) => pv < alphaCorr ? 1 : 0);
    return this;
  }

  get_support(indices = false): Uint8Array | Int32Array {
    const support = this._support ?? new Uint8Array(this.nFeaturesIn_);
    if (indices) return Int32Array.from(Array.from({ length: support.length }, (_, i) => i).filter((i) => support[i] !== 0));
    return support;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const support = this.get_support() as Uint8Array;
    const cols = Array.from({ length: support.length }, (_, j) => j).filter((j) => support[j] !== 0);
    return X.map((row) => Float64Array.from(cols, (j) => row[j] ?? 0));
  }

  fitTransform(X: Float64Array[], y: Float64Array | Int32Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}

export class SelectFdr {
  alpha: number;
  scoreFn: (X: Float64Array[], y: Float64Array | Int32Array) => { scores: Float64Array; pValues: Float64Array };
  scores_: Float64Array | null = null;
  pValues_: Float64Array | null = null;
  private _support: Uint8Array | null = null;
  nFeaturesIn_: number = 0;

  constructor(
    scoreFn: (X: Float64Array[], y: Float64Array | Int32Array) => { scores: Float64Array; pValues: Float64Array },
    alpha = 0.05,
  ) {
    this.alpha = alpha;
    this.scoreFn = scoreFn;
  }

  fit(X: Float64Array[], y: Float64Array | Int32Array): this {
    const p = X[0]?.length ?? 0;
    this.nFeaturesIn_ = p;
    const { scores, pValues } = this.scoreFn(X, y);
    this.scores_ = scores;
    this.pValues_ = pValues;
    // Benjamini-Hochberg procedure
    const sortedPairs = Array.from(pValues).map((pv, i) => ({ pv, i })).sort((a, b) => a.pv - b.pv);
    const threshold = sortedPairs.map((e, rank) => e.pv <= (rank + 1) * this.alpha / p);
    let lastTrue = -1;
    for (let k = threshold.length - 1; k >= 0; k--) {
      if (threshold[k]) { lastTrue = k; break; }
    }
    const selected = new Set(sortedPairs.slice(0, lastTrue + 1).map((e) => e.i));
    this._support = Uint8Array.from({ length: p }, (_, i) => selected.has(i) ? 1 : 0);
    return this;
  }

  get_support(indices = false): Uint8Array | Int32Array {
    const support = this._support ?? new Uint8Array(this.nFeaturesIn_);
    if (indices) return Int32Array.from(Array.from({ length: support.length }, (_, i) => i).filter((i) => support[i] !== 0));
    return support;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const support = this.get_support() as Uint8Array;
    const cols = Array.from({ length: support.length }, (_, j) => j).filter((j) => support[j] !== 0);
    return X.map((row) => Float64Array.from(cols, (j) => row[j] ?? 0));
  }

  fitTransform(X: Float64Array[], y: Float64Array | Int32Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}

export class VarianceThresholdSelector {
  threshold: number;
  variances_: Float64Array | null = null;
  private _support: Uint8Array | null = null;
  nFeaturesIn_: number = 0;

  constructor(threshold = 0.0) {
    this.threshold = threshold;
  }

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 0;
    this.nFeaturesIn_ = p;
    const means = new Float64Array(p);
    for (const row of X) for (let j = 0; j < p; j++) means[j]! += (row[j] ?? 0) / n;
    this.variances_ = Float64Array.from({ length: p }, (_, j) => {
      return X.reduce((s, row) => s + ((row[j] ?? 0) - (means[j] ?? 0)) ** 2, 0) / Math.max(n - 1, 1);
    });
    this._support = Uint8Array.from(this.variances_, (v) => v > this.threshold ? 1 : 0);
    return this;
  }

  get_support(indices = false): Uint8Array | Int32Array {
    const support = this._support ?? new Uint8Array(this.nFeaturesIn_);
    if (indices) return Int32Array.from(Array.from({ length: support.length }, (_, i) => i).filter((i) => support[i] !== 0));
    return support;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const support = this.get_support() as Uint8Array;
    const cols = Array.from({ length: support.length }, (_, j) => j).filter((j) => support[j] !== 0);
    return X.map((row) => Float64Array.from(cols, (j) => row[j] ?? 0));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

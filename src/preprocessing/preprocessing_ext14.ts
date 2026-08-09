/**
 * Preprocessing extensions: TargetEncoder, BinaryEncoder, CyclicalEncoder, TimeSeriesScaler
 */

export class TargetEncoderExt {
  private encodings_: Map<string, Map<string | number, number>> = new Map();
  private globalMean_: number = 0;
  private fitted_ = false;

  constructor(private smoothing: number = 10.0, private minSamplesLeaf: number = 1) {}

  fit(X: (string | number)[][], y: Float64Array, categoricalFeatures?: number[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const features = categoricalFeatures ?? Array.from({ length: p }, (_, i) => i);
    this.globalMean_ = y.reduce((s, v) => s + v, 0) / n;

    for (const j of features) {
      const map = new Map<string | number, { sum: number; count: number }>();
      for (let i = 0; i < n; i++) {
        const cat = X[i]?.[j] ?? '';
        const entry = map.get(cat) ?? { sum: 0, count: 0 };
        entry.sum += y[i] ?? 0;
        entry.count++;
        map.set(cat, entry);
      }
      const encoded = new Map<string | number, number>();
      for (const [cat, { sum, count }] of map) {
        const catMean = sum / count;
        const smoothed = (count * catMean + this.smoothing * this.globalMean_) / (count + this.smoothing);
        encoded.set(cat, smoothed);
      }
      this.encodings_.set(String(j), encoded);
    }
    this.fitted_ = true;
    return this;
  }

  transform(X: (string | number)[][], categoricalFeatures?: number[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    const p = X[0]?.length ?? 0;
    const features = new Set(categoricalFeatures ?? Array.from({ length: p }, (_, i) => i));
    return X.map(row => {
      const result = new Float64Array(p);
      for (let j = 0; j < p; j++) {
        if (features.has(j)) {
          const enc = this.encodings_.get(String(j));
          result[j] = enc?.get(row[j] ?? '') ?? this.globalMean_;
        } else {
          result[j] = Number(row[j] ?? 0);
        }
      }
      return result;
    });
  }

  fitTransform(X: (string | number)[][], y: Float64Array, categoricalFeatures?: number[]): Float64Array[] {
    return this.fit(X, y, categoricalFeatures).transform(X, categoricalFeatures);
  }
}

export class BinaryEncoderExt {
  private categories_: Map<number, Map<string | number, number>> = new Map();
  private nBits_: Map<number, number> = new Map();
  private fitted_ = false;

  fit(X: (string | number)[][], categoricalFeatures?: number[]): this {
    const p = X[0]?.length ?? 0;
    const features = categoricalFeatures ?? Array.from({ length: p }, (_, i) => i);

    for (const j of features) {
      const vals = new Set<string | number>();
      for (const row of X) vals.add(row[j] ?? '');
      const sortedVals = [...vals].sort();
      const catMap = new Map<string | number, number>();
      sortedVals.forEach((v, idx) => catMap.set(v, idx));
      this.categories_.set(j, catMap);
      this.nBits_.set(j, Math.ceil(Math.log2(sortedVals.length + 1)));
    }
    this.fitted_ = true;
    return this;
  }

  transform(X: (string | number)[][], categoricalFeatures?: number[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    const p = X[0]?.length ?? 0;
    const features = categoricalFeatures ?? Array.from({ length: p }, (_, i) => i);
    const featureSet = new Set(features);

    return X.map(row => {
      const parts: number[] = [];
      for (let j = 0; j < p; j++) {
        if (featureSet.has(j)) {
          const catMap = this.categories_.get(j);
          const nBits = this.nBits_.get(j) ?? 1;
          const code = catMap?.get(row[j] ?? '') ?? 0;
          for (let b = 0; b < nBits; b++) parts.push((code >> b) & 1);
        } else {
          parts.push(Number(row[j] ?? 0));
        }
      }
      return new Float64Array(parts);
    });
  }
}

export class CyclicalEncoderExt {
  constructor(
    private periods: Record<number, number>
  ) {}

  transform(X: Float64Array[]): Float64Array[] {
    return X.map(row => {
      const parts: number[] = [];
      for (let j = 0; j < row.length; j++) {
        const period = this.periods[j];
        if (period !== undefined) {
          const angle = (2 * Math.PI * (row[j] ?? 0)) / period;
          parts.push(Math.sin(angle), Math.cos(angle));
        } else {
          parts.push(row[j] ?? 0);
        }
      }
      return new Float64Array(parts);
    });
  }
}

export class TimeSeriesScalerExt {
  private scalers_: Array<{ mean: number; std: number }> = [];
  private fitted_ = false;

  constructor(private windowSize: number = 1) {}

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this.scalers_ = [];

    for (let j = 0; j < p; j++) {
      const vals = X.map(row => row[j] ?? 0);
      const mean = vals.reduce((s, v) => s + v, 0) / n;
      const std = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / n) || 1;
      this.scalers_.push({ mean, std });
    }
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return X.map(row => new Float64Array(row.map((v, j) => (v - (this.scalers_[j]?.mean ?? 0)) / (this.scalers_[j]?.std ?? 1))));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return X.map(row => new Float64Array(row.map((v, j) => v * (this.scalers_[j]?.std ?? 1) + (this.scalers_[j]?.mean ?? 0))));
  }
}

export class HashingEncoderExt {
  constructor(private nComponents: number = 8) {}

  transform(X: (string | number)[][]): Float64Array[] {
    return X.map(row => {
      const result = new Float64Array(this.nComponents);
      for (let j = 0; j < row.length; j++) {
        const key = `${j}=${row[j]}`;
        const hash = this._hash(key);
        const idx = Math.abs(hash) % this.nComponents;
        result[idx] = (result[idx] ?? 0) + (hash > 0 ? 1 : -1);
      }
      return result;
    });
  }

  private _hash(s: string): number {
    let h = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = (h * 0x01000193) >>> 0;
    }
    return h > 0x7fffffff ? h - 0x100000000 : h;
  }
}

export class PolynomialFeaturesExt {
  private outputFeatureNames_: string[] = [];
  private fitted_ = false;
  private degree_: number;

  constructor(private degree: number = 2, private interactionOnly: boolean = false, private includeBias: boolean = true) {
    this.degree_ = degree;
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    const p = X[0]?.length ?? 0;
    this.outputFeatureNames_ = this._computeFeatureNames(p);
    this.fitted_ = true;
    return this.transform(X);
  }

  transform(X: Float64Array[]): Float64Array[] {
    const p = X[0]?.length ?? 0;
    const combinations = this._getCombinations(p);
    return X.map(row => {
      const features: number[] = [];
      if (this.includeBias) features.push(1);
      for (const combo of combinations) {
        let val = 1;
        for (const idx of combo) val *= row[idx] ?? 0;
        features.push(val);
      }
      return new Float64Array(features);
    });
  }

  private _getCombinations(p: number): number[][] {
    const result: number[][] = [];
    const generate = (start: number, current: number[], degree: number) => {
      if (degree === 0) {
        if (!this.interactionOnly || new Set(current).size === current.length) result.push([...current]);
        return;
      }
      for (let i = start; i < p; i++) {
        current.push(i);
        generate(i, current, degree - 1);
        current.pop();
      }
    };
    for (let d = 1; d <= this.degree_; d++) generate(0, [], d);
    return result;
  }

  private _computeFeatureNames(p: number): string[] {
    const names: string[] = [];
    if (this.includeBias) names.push('1');
    const combos = this._getCombinations(p);
    for (const combo of combos) names.push(combo.map(i => `x${i}`).join('*'));
    return names;
  }

  get outputFeatureNames(): string[] { return this.outputFeatureNames_; }
  get nOutputFeatures(): number { return this.outputFeatureNames_.length; }
}

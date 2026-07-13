/**
 * Preprocessing encoder extensions: TargetEncoder (extended), HashingEncoder, WOEEncoder.
 */

export class TargetEncoderExt {
  private encodings_: Map<number, Map<number, number>> = new Map();
  private globalMean_ = 0;
  private smoothing: number;

  constructor(
    private readonly smoothingParam = 10.0,
    private readonly cvFolds = 5
  ) {
    this.smoothing = smoothingParam;
  }

  fit(X: Int32Array[], y: Float64Array): this {
    const n = X.length;
    const nF = X[0]?.length ?? 0;
    this.globalMean_ = y.reduce((a, b) => a + b, 0) / Math.max(n, 1);
    this.encodings_ = new Map();
    for (let f = 0; f < nF; f++) {
      const catMap = new Map<number, { sum: number; count: number }>();
      for (let i = 0; i < n; i++) {
        const cat = X[i]?.[f] ?? 0;
        const stats = catMap.get(cat) ?? { sum: 0, count: 0 };
        stats.sum += y[i] ?? 0;
        stats.count++;
        catMap.set(cat, stats);
      }
      const encoding = new Map<number, number>();
      for (const [cat, stats] of catMap) {
        const catMean = stats.sum / stats.count;
        const weight = stats.count / (stats.count + this.smoothing);
        encoding.set(cat, weight * catMean + (1 - weight) * this.globalMean_);
      }
      this.encodings_.set(f, encoding);
    }
    return this;
  }

  transform(X: Int32Array[]): Float64Array[] {
    const nF = X[0]?.length ?? 0;
    return X.map((x) => {
      const result = new Float64Array(nF);
      for (let f = 0; f < nF; f++) {
        const cat = x[f] ?? 0;
        result[f] = this.encodings_.get(f)?.get(cat) ?? this.globalMean_;
      }
      return result;
    });
  }

  fitTransform(X: Int32Array[], y: Float64Array): Float64Array[] { return this.fit(X, y).transform(X); }
}

export class WOEEncoder {
  private encodings_: Map<number, Map<number, number>> = new Map();
  private nFeatures_ = 0;

  fit(X: Int32Array[], y: Int32Array): this {
    const n = X.length;
    this.nFeatures_ = X[0]?.length ?? 0;
    const totalPos = y.reduce((s, v) => s + (v === 1 ? 1 : 0), 0);
    const totalNeg = n - totalPos;
    for (let f = 0; f < this.nFeatures_; f++) {
      const catStats = new Map<number, { pos: number; neg: number }>();
      for (let i = 0; i < n; i++) {
        const cat = X[i]?.[f] ?? 0;
        const s = catStats.get(cat) ?? { pos: 0, neg: 0 };
        if ((y[i] ?? 0) === 1) s.pos++; else s.neg++;
        catStats.set(cat, s);
      }
      const encoding = new Map<number, number>();
      for (const [cat, stats] of catStats) {
        const pPos = stats.pos / Math.max(totalPos, 1);
        const pNeg = stats.neg / Math.max(totalNeg, 1);
        const woe = Math.log(Math.max(pPos, 1e-10) / Math.max(pNeg, 1e-10));
        encoding.set(cat, woe);
      }
      this.encodings_.set(f, encoding);
    }
    return this;
  }

  transform(X: Int32Array[]): Float64Array[] {
    return X.map((x) => new Float64Array(x.map((cat, f) => this.encodings_.get(f)?.get(cat) ?? 0)));
  }
}

export class BinaryEncoder {
  private nBits_: number[] = [];
  private categoryMaps_: Map<number, number>[] = [];

  fit(X: Int32Array[]): this {
    const nF = X[0]?.length ?? 0;
    this.nBits_ = [];
    this.categoryMaps_ = [];
    for (let f = 0; f < nF; f++) {
      const cats = new Set<number>();
      for (const x of X) cats.add(x[f] ?? 0);
      const sortedCats = [...cats].sort((a, b) => a - b);
      const catMap = new Map(sortedCats.map((c, i) => [c, i]));
      this.categoryMaps_.push(catMap);
      this.nBits_.push(Math.max(1, Math.ceil(Math.log2(cats.size + 1))));
    }
    return this;
  }

  transform(X: Int32Array[]): Float64Array[] {
    const nF = X[0]?.length ?? 0;
    const totalBits = this.nBits_.reduce((a, b) => a + b, 0);
    return X.map((x) => {
      const result = new Float64Array(totalBits);
      let offset = 0;
      for (let f = 0; f < nF; f++) {
        const cat = x[f] ?? 0;
        const idx = this.categoryMaps_[f]?.get(cat) ?? 0;
        const bits = this.nBits_[f] ?? 1;
        for (let b = 0; b < bits; b++) result[offset + b] = (idx >> b) & 1;
        offset += bits;
      }
      return result;
    });
  }

  fitTransform(X: Int32Array[]): Float64Array[] { return this.fit(X).transform(X); }
}

export class CyclicalEncoder {
  constructor(
    private readonly period: number,
    private readonly featureIndex = 0
  ) {}

  transform(X: Float64Array[]): Float64Array[] {
    return X.map((x) => {
      const v = x[this.featureIndex] ?? 0;
      return new Float64Array([
        Math.sin(2 * Math.PI * v / this.period),
        Math.cos(2 * Math.PI * v / this.period),
      ]);
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.transform(X); }
}

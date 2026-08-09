/**
 * TargetEncoder, CategoricalEncoder, and WOE (Weight of Evidence) encoder.
 */

export class TargetEncoder {
  private encodings_!: Map<number, number>[];
  private globalMean_ = 0;
  private smoothing: number;
  private fitted_ = false;

  constructor(smoothing = 10.0, private cv = 5) {
    this.smoothing = smoothing;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length, p = X[0]?.length ?? 0;
    this.globalMean_ = y.reduce((s, v) => s + v, 0) / (n || 1);
    this.encodings_ = Array.from({ length: p }, (_, j) => {
      const catMeans = new Map<number, { sum: number; count: number }>();
      for (let i = 0; i < n; i++) {
        const cat = Math.round(X[i]![j] ?? 0);
        const cur = catMeans.get(cat) ?? { sum: 0, count: 0 };
        cur.sum += y[i] ?? 0;
        cur.count += 1;
        catMeans.set(cat, cur);
      }
      const enc = new Map<number, number>();
      catMeans.forEach((v, cat) => {
        const lambda = v.count / (v.count + this.smoothing);
        enc.set(cat, lambda * (v.sum / v.count) + (1 - lambda) * this.globalMean_);
      });
      return enc;
    });
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return X.map(row => new Float64Array(row.map((v, j) => {
      const cat = Math.round(v);
      return this.encodings_[j]?.get(cat) ?? this.globalMean_;
    })));
  }

  fitTransform(X: Float64Array[], y: Float64Array): Float64Array[] { return this.fit(X, y).transform(X); }
  get encodings(): Map<number, number>[] { return this.encodings_; }
}

export class WOEEncoder {
  private woeValues_!: Map<number, number>[];
  private ivValues_!: number[];
  private fitted_ = false;

  constructor(private smoothing = 0.5) {}

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length, p = X[0]?.length ?? 0;
    const totalPos = Array.from(y).filter(v => v === 1).length;
    const totalNeg = n - totalPos;
    this.woeValues_ = Array.from({ length: p }, (_, j) => {
      const cats = new Map<number, { pos: number; neg: number }>();
      for (let i = 0; i < n; i++) {
        const cat = Math.round(X[i]![j] ?? 0);
        const cur = cats.get(cat) ?? { pos: 0, neg: 0 };
        if (y[i] === 1) cur.pos++; else cur.neg++;
        cats.set(cat, cur);
      }
      const woe = new Map<number, number>();
      cats.forEach((v, cat) => {
        const distPos = (v.pos + this.smoothing) / (totalPos + this.smoothing);
        const distNeg = (v.neg + this.smoothing) / (totalNeg + this.smoothing);
        woe.set(cat, Math.log(distPos / distNeg));
      });
      return woe;
    });
    this.ivValues_ = Array.from({ length: p }, (_, j) => {
      let iv = 0;
      const cats = new Map<number, { pos: number; neg: number }>();
      for (let i = 0; i < n; i++) {
        const cat = Math.round(X[i]![j] ?? 0);
        const cur = cats.get(cat) ?? { pos: 0, neg: 0 };
        if (y[i] === 1) cur.pos++; else cur.neg++;
        cats.set(cat, cur);
      }
      cats.forEach(v => {
        const dp = (v.pos + this.smoothing) / (totalPos + this.smoothing);
        const dn = (v.neg + this.smoothing) / (totalNeg + this.smoothing);
        iv += (dp - dn) * Math.log(dp / dn);
      });
      return iv;
    });
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return X.map(row => new Float64Array(row.map((v, j) => {
      const cat = Math.round(v);
      return this.woeValues_[j]?.get(cat) ?? 0;
    })));
  }

  fitTransform(X: Float64Array[], y: Int32Array): Float64Array[] { return this.fit(X, y).transform(X); }
  get informationValues(): number[] { return this.ivValues_; }
}

export class HashingEncoder {
  private nComponents: number;
  private fitted_ = false;

  constructor(nComponents = 8) {
    this.nComponents = nComponents;
  }

  fit(_X: Float64Array[]): this { this.fitted_ = true; return this; }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return X.map(row => {
      const out = new Float64Array(this.nComponents);
      for (let j = 0; j < row.length; j++) {
        // MurmurHash-inspired hash
        let h = Math.round(row[j] ?? 0) * 2654435761;
        h = h ^ (h >>> 16);
        const idx = Math.abs(h) % this.nComponents;
        const sign = h < 0 ? -1 : 1;
        out[idx] = (out[idx] ?? 0) + sign;
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.fit(X).transform(X); }
}

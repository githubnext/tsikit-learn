/**
 * Extended preprocessing: SplineTransformer, PiecewiseLinearTransformer, PolynomialCountSketch
 */

export class SplineTransformerExt {
  private nKnots: number;
  private degree: number;
  private knots_: Float64Array | null = null;
  nSplines_: number = 0;

  constructor(nKnots = 5, degree = 3) {
    this.nKnots = nKnots;
    this.degree = degree;
  }

  fit(X: Float64Array): this {
    const sorted = Float64Array.from(X).sort();
    const n = sorted.length;
    this.knots_ = new Float64Array(this.nKnots);
    for (let i = 0; i < this.nKnots; i++) {
      const idx = Math.floor((i / (this.nKnots - 1)) * (n - 1));
      this.knots_[i] = sorted[idx] ?? 0;
    }
    this.nSplines_ = this.nKnots + this.degree - 1;
    return this;
  }

  private bSpline(x: number, i: number, k: number): number {
    if (k === 0) {
      const ti = this.knots_![i] ?? 0;
      const ti1 = this.knots_![i + 1] ?? 0;
      return x >= ti && x < ti1 ? 1 : 0;
    }
    const ti = this.knots_![i] ?? 0;
    const tik = this.knots_![i + k] ?? 0;
    const ti1 = this.knots_![i + 1] ?? 0;
    const tik1 = this.knots_![i + k + 1] ?? 0;
    const d1 = tik - ti;
    const d2 = tik1 - ti1;
    const c1 = d1 === 0 ? 0 : ((x - ti) / d1) * this.bSpline(x, i, k - 1);
    const c2 = d2 === 0 ? 0 : ((tik1 - x) / d2) * this.bSpline(x, i + 1, k - 1);
    return c1 + c2;
  }

  transform(X: Float64Array): Float64Array[] {
    if (!this.knots_) throw new Error("Not fitted");
    return Array.from(X).map((x) => {
      const row = new Float64Array(this.nSplines_);
      for (let i = 0; i < this.nSplines_; i++) {
        row[i] = this.bSpline(x, i, this.degree);
      }
      return row;
    });
  }

  fitTransform(X: Float64Array): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class PiecewiseLinearTransformer {
  private nBreakpoints: number;
  private breakpoints_: Float64Array | null = null;

  constructor(nBreakpoints = 5) {
    this.nBreakpoints = nBreakpoints;
  }

  fit(X: Float64Array): this {
    const sorted = Float64Array.from(X).sort();
    const n = sorted.length;
    this.breakpoints_ = new Float64Array(this.nBreakpoints);
    for (let i = 0; i < this.nBreakpoints; i++) {
      const idx = Math.floor(((i + 1) / (this.nBreakpoints + 1)) * n);
      this.breakpoints_[i] = sorted[Math.min(idx, n - 1)] ?? 0;
    }
    return this;
  }

  transform(X: Float64Array): Float64Array[] {
    if (!this.breakpoints_) throw new Error("Not fitted");
    const bp = this.breakpoints_;
    return Array.from(X).map((x) => {
      const row = new Float64Array(this.nBreakpoints + 1);
      row[0] = x;
      for (let i = 0; i < this.nBreakpoints; i++) {
        row[i + 1] = Math.max(0, x - (bp[i] ?? 0));
      }
      return row;
    });
  }

  fitTransform(X: Float64Array): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class PolynomialCountSketch {
  private degree: number;
  private nComponents: number;
  private randomWeights_: Float64Array[] | null = null;
  private randomBits_: Int32Array[] | null = null;

  constructor(degree = 2, nComponents = 100) {
    this.degree = degree;
    this.nComponents = nComponents;
  }

  fit(nFeatures: number): this {
    this.randomWeights_ = Array.from({ length: this.degree }, () => {
      const w = new Float64Array(nFeatures * this.nComponents);
      for (let i = 0; i < w.length; i++) {
        w[i] = Math.random() < 0.5 ? -1 : 1;
      }
      return w;
    });
    this.randomBits_ = Array.from({ length: this.degree }, () => {
      const b = new Int32Array(nFeatures);
      for (let i = 0; i < b.length; i++) {
        b[i] = Math.floor(Math.random() * this.nComponents);
      }
      return b;
    });
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.randomWeights_ || !this.randomBits_) throw new Error("Not fitted");
    return X.map((row) => {
      const sketch = new Float64Array(this.nComponents);
      const weights = this.randomWeights_![0]!;
      const bits = this.randomBits_![0]!;
      for (let i = 0; i < row.length; i++) {
        const w = weights[i] ?? 1;
        const b = bits[i] ?? 0;
        sketch[b]! += w * (row[i] ?? 0);
      }
      return sketch;
    });
  }
}

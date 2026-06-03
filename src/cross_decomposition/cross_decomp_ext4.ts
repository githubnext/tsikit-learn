/**
 * Extended cross-decomposition methods.
 * Port of sklearn.cross_decomposition extensions.
 */

import { NotFittedError } from "../exceptions.js";

/**
 * Canonical Correlation Analysis (CCA) extended with regularization.
 */
export class RegularizedCCA {
  private nComponents: number;
  private alpha: number; // regularization
  private xWeights_: Float64Array[] = [];
  private yWeights_: Float64Array[] = [];
  private fitted = false;

  constructor(options: { nComponents?: number; alpha?: number } = {}) {
    this.nComponents = options.nComponents ?? 2;
    this.alpha = options.alpha ?? 0.1;
  }

  private center(X: Float64Array[]): { centered: Float64Array[]; mean: Float64Array } {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const mean = new Float64Array(p);
    for (const row of X) for (let j = 0; j < p; j++) mean[j] = (mean[j] ?? 0) + (row[j] ?? 0) / n;
    const centered = X.map(row => Float64Array.from({ length: p }, (_, j) => (row[j] ?? 0) - (mean[j] ?? 0)));
    return { centered, mean };
  }

  private covariance(X: Float64Array[], Y: Float64Array[]): Float64Array[] {
    const n = X.length;
    const px = X[0]?.length ?? 0;
    const py = Y[0]?.length ?? 0;
    const C = Array.from({ length: px }, () => new Float64Array(py));
    for (let k = 0; k < n; k++) {
      for (let i = 0; i < px; i++) {
        for (let j = 0; j < py; j++) {
          C[i]![j] = (C[i]?.[j] ?? 0) + (X[k]?.[i] ?? 0) * (Y[k]?.[j] ?? 0) / n;
        }
      }
    }
    return C;
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const { centered: Xc } = this.center(X);
    const { centered: Yc } = this.center(Y);
    const px = Xc[0]?.length ?? 0;
    const py = Yc[0]?.length ?? 0;
    const n = Xc.length;

    const Cxx = this.covariance(Xc, Xc);
    const Cyy = this.covariance(Yc, Yc);
    const Cxy = this.covariance(Xc, Yc);

    // Add regularization
    for (let i = 0; i < px; i++) Cxx[i]![i] = (Cxx[i]?.[i] ?? 0) + this.alpha;
    for (let i = 0; i < py; i++) Cyy[i]![i] = (Cyy[i]?.[i] ?? 0) + this.alpha;

    // Power iteration for leading canonical directions
    this.xWeights_ = [];
    this.yWeights_ = [];

    for (let k = 0; k < Math.min(this.nComponents, Math.min(px, py)); k++) {
      // Initialize random directions
      let wx = new Float64Array(px);
      let wy = new Float64Array(py);
      for (let i = 0; i < px; i++) wx[i] = Math.random() - 0.5;
      for (let i = 0; i < py; i++) wy[i] = Math.random() - 0.5;

      // Normalize
      const normX = Math.sqrt(wx.reduce((s, v) => s + v * v, 0));
      const normY = Math.sqrt(wy.reduce((s, v) => s + v * v, 0));
      for (let i = 0; i < px; i++) wx[i] = (wx[i] ?? 0) / normX;
      for (let i = 0; i < py; i++) wy[i] = (wy[i] ?? 0) / normY;

      // Power iteration
      for (let iter = 0; iter < 100; iter++) {
        // wx <- Cxx^{-1} Cxy wy (simplified: just Cxy wy)
        const newWx = new Float64Array(px);
        for (let i = 0; i < px; i++) {
          for (let j = 0; j < py; j++) newWx[i] = (newWx[i] ?? 0) + (Cxy[i]?.[j] ?? 0) * (wy[j] ?? 0);
        }
        const nwx = Math.sqrt(newWx.reduce((s, v) => s + v * v, 0));
        for (let i = 0; i < px; i++) newWx[i] = (newWx[i] ?? 0) / Math.max(nwx, 1e-10);

        const newWy = new Float64Array(py);
        for (let j = 0; j < py; j++) {
          for (let i = 0; i < px; i++) newWy[j] = (newWy[j] ?? 0) + (Cxy[i]?.[j] ?? 0) * (newWx[i] ?? 0);
        }
        const nwy = Math.sqrt(newWy.reduce((s, v) => s + v * v, 0));
        for (let j = 0; j < py; j++) newWy[j] = (newWy[j] ?? 0) / Math.max(nwy, 1e-10);

        const diff = newWx.reduce((s, v, i) => s + Math.abs(v - (wx[i] ?? 0)), 0);
        wx = newWx; wy = newWy;
        if (diff < 1e-8) break;
      }

      this.xWeights_.push(wx);
      this.yWeights_.push(wy);

      // Deflate
      for (let i = 0; i < n; i++) {
        let xProj = 0; let yProj = 0;
        for (let j = 0; j < px; j++) xProj += (Xc[i]?.[j] ?? 0) * (wx[j] ?? 0);
        for (let j = 0; j < py; j++) yProj += (Yc[i]?.[j] ?? 0) * (wy[j] ?? 0);
        for (let j = 0; j < px; j++) Xc[i]![j] = (Xc[i]?.[j] ?? 0) - xProj * (wx[j] ?? 0);
        for (let j = 0; j < py; j++) Yc[i]![j] = (Yc[i]?.[j] ?? 0) - yProj * (wy[j] ?? 0);
      }
    }

    this.fitted = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("RegularizedCCA not fitted");
    return X.map(row => Float64Array.from(this.xWeights_, wx => {
      let proj = 0;
      for (let j = 0; j < row.length; j++) proj += (row[j] ?? 0) * (wx[j] ?? 0);
      return proj;
    }));
  }
}

/**
 * PLS-DA (Discriminant Analysis) — PLS regression applied to one-hot encoded class labels.
 */
export class PLSDiscriminantAnalysis {
  private nComponents: number;
  private xWeights_: Float64Array[] = [];
  private yWeights_: Float64Array[] = [];
  private xMean_: Float64Array = new Float64Array(0);
  private yMean_: Float64Array = new Float64Array(0);
  private nClasses = 0;
  private fitted = false;

  constructor(options: { nComponents?: number } = {}) {
    this.nComponents = options.nComponents ?? 2;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const classes = new Set(Array.from(y));
    this.nClasses = classes.size;
    const classMap = new Map(Array.from(classes).sort((a, b) => a - b).map((c, i) => [c, i]));

    // One-hot encode Y
    const Y: Float64Array[] = Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(this.nClasses);
      const c = classMap.get(y[i] ?? 0) ?? 0;
      row[c] = 1;
      return row;
    });

    // Center X and Y
    const px = X[0]?.length ?? 0;
    this.xMean_ = new Float64Array(px);
    this.yMean_ = new Float64Array(this.nClasses);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < px; j++) this.xMean_[j] = (this.xMean_[j] ?? 0) + (X[i]?.[j] ?? 0) / n;
      for (let j = 0; j < this.nClasses; j++) this.yMean_[j] = (this.yMean_[j] ?? 0) + (Y[i]?.[j] ?? 0) / n;
    }

    const Xc = X.map(row => Float64Array.from({ length: px }, (_, j) => (row[j] ?? 0) - (this.xMean_[j] ?? 0)));
    const Yc = Y.map(row => Float64Array.from({ length: this.nClasses }, (_, j) => (row[j] ?? 0) - (this.yMean_[j] ?? 0)));

    this.xWeights_ = []; this.yWeights_ = [];

    for (let k = 0; k < Math.min(this.nComponents, px); k++) {
      // NIPALS step
      let u = Yc[0] ? Float64Array.from(Yc[0]) : new Float64Array(this.nClasses).fill(1 / this.nClasses);

      let w = new Float64Array(px);
      let c = new Float64Array(this.nClasses);

      for (let iter = 0; iter < 100; iter++) {
        // w = X^T u / ||X^T u||
        for (let j = 0; j < px; j++) {
          w[j] = 0;
          for (let i = 0; i < n; i++) w[j] = (w[j] ?? 0) + (Xc[i]?.[j] ?? 0) * (u[i] ?? 0);
        }
        const wNorm = Math.sqrt(w.reduce((s, v) => s + v * v, 0));
        for (let j = 0; j < px; j++) w[j] = (w[j] ?? 0) / Math.max(wNorm, 1e-10);

        // t = X w
        const t = Float64Array.from({ length: n }, (_, i) => {
          let s = 0; for (let j = 0; j < px; j++) s += (Xc[i]?.[j] ?? 0) * (w[j] ?? 0); return s;
        });

        // c = Y^T t / ||Y^T t||
        for (let j = 0; j < this.nClasses; j++) {
          c[j] = 0;
          for (let i = 0; i < n; i++) c[j] = (c[j] ?? 0) + (Yc[i]?.[j] ?? 0) * (t[i] ?? 0);
        }
        const cNorm = Math.sqrt(c.reduce((s, v) => s + v * v, 0));
        for (let j = 0; j < this.nClasses; j++) c[j] = (c[j] ?? 0) / Math.max(cNorm, 1e-10);

        // u = Y c
        const newU = Float64Array.from({ length: n }, (_, i) => {
          let s = 0; for (let j = 0; j < this.nClasses; j++) s += (Yc[i]?.[j] ?? 0) * (c[j] ?? 0); return s;
        });

        const diff = newU.reduce((s, v, i) => s + Math.abs(v - (u[i] ?? 0)), 0);
        u = newU;
        if (diff < 1e-8) break;
      }

      this.xWeights_.push(w);
      this.yWeights_.push(c);

      // Deflate
      const t = Float64Array.from({ length: n }, (_, i) => {
        let s = 0; for (let j = 0; j < px; j++) s += (Xc[i]?.[j] ?? 0) * (w[j] ?? 0); return s;
      });
      const tNorm2 = t.reduce((s, v) => s + v * v, 0);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < px; j++) Xc[i]![j] = (Xc[i]?.[j] ?? 0) - (t[i] ?? 0) * (w[j] ?? 0);
        for (let j = 0; j < this.nClasses; j++) Yc[i]![j] = (Yc[i]?.[j] ?? 0) - (t[i] ?? 0) * (c[j] ?? 0);
      }
    }

    this.fitted = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("PLSDiscriminantAnalysis not fitted");
    const px = this.xMean_.length;
    return X.map(row => {
      const xc = Float64Array.from({ length: px }, (_, j) => (row[j] ?? 0) - (this.xMean_[j] ?? 0));
      return Float64Array.from(this.xWeights_, w => {
        let s = 0; for (let j = 0; j < px; j++) s += (xc[j] ?? 0) * (w[j] ?? 0); return s;
      });
    });
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted) throw new NotFittedError("PLSDiscriminantAnalysis not fitted");
    const scores = this.transform(X);
    return Int32Array.from(scores, row => {
      let best = 0; let bestS = Number.NEGATIVE_INFINITY;
      for (let c = 0; c < row.length; c++) {
        if ((row[c] ?? 0) > bestS) { bestS = row[c] ?? 0; best = c; }
      }
      return best;
    });
  }
}

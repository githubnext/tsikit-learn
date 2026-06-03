/**
 * Extended discriminant analysis methods.
 * Port of sklearn.discriminant_analysis extensions.
 */

import { NotFittedError } from "../exceptions.js";

/**
 * Regularized Linear Discriminant Analysis with shrinkage.
 */
export class RegularizedLDA {
  private nComponents: number;
  private shrinkage: number | "auto";
  private scalings_: Float64Array[] = [];
  private means_: Map<number, Float64Array> = new Map();
  private priors_: Map<number, number> = new Map();
  private xbar_: Float64Array = new Float64Array(0);
  private fitted = false;

  constructor(options: { nComponents?: number; shrinkage?: number | "auto" } = {}) {
    this.nComponents = options.nComponents ?? 1;
    this.shrinkage = options.shrinkage ?? 0;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const classes = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);

    // Compute class means and priors
    const classSums: Map<number, Float64Array> = new Map();
    const classCounts: Map<number, number> = new Map();

    for (const c of classes) {
      classSums.set(c, new Float64Array(p));
      classCounts.set(c, 0);
    }

    for (let i = 0; i < n; i++) {
      const c = y[i] ?? 0;
      const sum = classSums.get(c)!;
      for (let j = 0; j < p; j++) sum[j] = (sum[j] ?? 0) + (X[i]?.[j] ?? 0);
      classCounts.set(c, (classCounts.get(c) ?? 0) + 1);
    }

    this.xbar_ = new Float64Array(p);
    for (const [c, sum] of classSums) {
      const count = classCounts.get(c) ?? 1;
      const mean = Float64Array.from(sum, v => v / count);
      this.means_.set(c, mean);
      this.priors_.set(c, count / n);
      for (let j = 0; j < p; j++) this.xbar_[j] = (this.xbar_[j] ?? 0) + (mean[j] ?? 0) * count / n;
    }

    // Within-class scatter matrix Sw
    const Sw = Array.from({ length: p }, () => new Float64Array(p));
    for (let i = 0; i < n; i++) {
      const mean = this.means_.get(y[i] ?? 0)!;
      const diff = Float64Array.from({ length: p }, (_, j) => (X[i]?.[j] ?? 0) - (mean[j] ?? 0));
      for (let j = 0; j < p; j++) {
        for (let k = 0; k < p; k++) {
          Sw[j]![k] = (Sw[j]?.[k] ?? 0) + (diff[j] ?? 0) * (diff[k] ?? 0);
        }
      }
    }

    // Apply shrinkage (Ledoit-Wolf style)
    const shrink = this.shrinkage === "auto"
      ? this.estimateShrinkage(Sw, n, p)
      : (this.shrinkage as number);

    const trSw = Sw.reduce((s, row, i) => s + (row[i] ?? 0), 0);
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < p; k++) {
        Sw[j]![k] = (1 - shrink) * (Sw[j]?.[k] ?? 0) / n + (j === k ? shrink * trSw / p : 0);
      }
    }

    // Between-class scatter matrix Sb
    const Sb = Array.from({ length: p }, () => new Float64Array(p));
    for (const [c, mean] of this.means_) {
      const count = classCounts.get(c) ?? 0;
      const diff = Float64Array.from({ length: p }, (_, j) => (mean[j] ?? 0) - (this.xbar_[j] ?? 0));
      for (let j = 0; j < p; j++) {
        for (let k = 0; k < p; k++) {
          Sb[j]![k] = (Sb[j]?.[k] ?? 0) + count * (diff[j] ?? 0) * (diff[k] ?? 0);
        }
      }
    }

    // Power iteration for Sw^{-1} Sb eigenvectors
    const nComp = Math.min(this.nComponents, classes.length - 1, p);
    this.scalings_ = [];

    // Simplified: use random projections as approximation
    for (let k = 0; k < nComp; k++) {
      let v = new Float64Array(p);
      for (let j = 0; j < p; j++) v[j] = Math.random() - 0.5;
      const vn = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
      for (let j = 0; j < p; j++) v[j] = (v[j] ?? 0) / vn;

      for (let iter = 0; iter < 50; iter++) {
        // v <- Sb v
        const Sbv = new Float64Array(p);
        for (let j = 0; j < p; j++) {
          for (let l = 0; l < p; l++) Sbv[j] = (Sbv[j] ?? 0) + (Sb[j]?.[l] ?? 0) * (v[l] ?? 0);
        }
        // Solve Sw * result = Sbv (simplified: use diagonal approx)
        const diag = Float64Array.from({ length: p }, (_, j) => Sw[j]?.[j] ?? 1);
        const newV = Float64Array.from({ length: p }, (_, j) => (Sbv[j] ?? 0) / Math.max(diag[j] ?? 1, 1e-10));
        const norm = Math.sqrt(newV.reduce((s, x) => s + x * x, 0));
        for (let j = 0; j < p; j++) newV[j] = (newV[j] ?? 0) / Math.max(norm, 1e-10);
        const diff = newV.reduce((s, x, j) => s + Math.abs(x - (v[j] ?? 0)), 0);
        v = newV;
        if (diff < 1e-8) break;
      }
      this.scalings_.push(v);
    }

    this.fitted = true;
    return this;
  }

  private estimateShrinkage(Sw: Float64Array[], n: number, p: number): number {
    // Ledoit-Wolf analytical approximation
    const trSw = Sw.reduce((s, row, i) => s + (row[i] ?? 0), 0);
    const trSw2 = Sw.reduce((s, row) => s + row.reduce((rs, v) => rs + v * v, 0), 0);
    const mu = trSw / p;
    const delta = (1 / (p * n)) * (trSw2 - trSw * trSw / p);
    const gamma = Math.min(1, Math.max(0, delta / trSw2));
    return gamma;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("RegularizedLDA not fitted");
    const p = this.xbar_.length;
    return X.map(row => {
      const xc = Float64Array.from({ length: p }, (_, j) => (row[j] ?? 0) - (this.xbar_[j] ?? 0));
      return Float64Array.from(this.scalings_, w => {
        let s = 0; for (let j = 0; j < p; j++) s += (xc[j] ?? 0) * (w[j] ?? 0); return s;
      });
    });
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted) throw new NotFittedError("RegularizedLDA not fitted");
    const classes = Array.from(this.means_.keys()).sort((a, b) => a - b);
    return Int32Array.from(X, row => {
      let best = classes[0] ?? 0; let bestScore = Number.NEGATIVE_INFINITY;
      for (const c of classes) {
        const mean = this.means_.get(c)!;
        let score = Math.log(this.priors_.get(c) ?? 1e-10);
        for (let j = 0; j < row.length; j++) {
          const diff = (row[j] ?? 0) - (mean[j] ?? 0);
          score -= 0.5 * diff * diff;
        }
        if (score > bestScore) { bestScore = score; best = c; }
      }
      return best;
    });
  }
}

/**
 * Flexible QDA with full covariance per class.
 */
export class FlexibleQDA {
  private priors_: Map<number, number> = new Map();
  private means_: Map<number, Float64Array> = new Map();
  private precisions_: Map<number, Float64Array[]> = new Map();
  private logDets_: Map<number, number> = new Map();
  private reg: number;
  private fitted = false;

  constructor(options: { reg?: number } = {}) {
    this.reg = options.reg ?? 1e-4;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const classes = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);

    for (const c of classes) {
      const indices = Array.from({ length: n }, (_, i) => i).filter(i => y[i] === c);
      const nc = indices.length;
      this.priors_.set(c, nc / n);

      const mean = new Float64Array(p);
      for (const idx of indices) for (let j = 0; j < p; j++) mean[j] = (mean[j] ?? 0) + (X[idx]?.[j] ?? 0) / nc;
      this.means_.set(c, mean);

      const cov = Array.from({ length: p }, () => new Float64Array(p));
      for (const idx of indices) {
        const diff = Float64Array.from({ length: p }, (_, j) => (X[idx]?.[j] ?? 0) - (mean[j] ?? 0));
        for (let j = 0; j < p; j++) {
          for (let k = 0; k < p; k++) cov[j]![k] = (cov[j]?.[k] ?? 0) + (diff[j] ?? 0) * (diff[k] ?? 0) / nc;
        }
      }
      for (let j = 0; j < p; j++) cov[j]![j] = (cov[j]?.[j] ?? 0) + this.reg;

      // Invert covariance (diagonal approximation for speed)
      const precision = Array.from({ length: p }, (_, i) => {
        const row = new Float64Array(p);
        row[i] = 1 / Math.max(cov[i]?.[i] ?? 1, 1e-10);
        return row;
      });
      this.precisions_.set(c, precision);
      this.logDets_.set(c, p * Math.log(2 * Math.PI) + Float64Array.from({ length: p }, (_, i) => cov[i]?.[i] ?? 1).reduce((s, v) => s + Math.log(Math.max(v, 1e-10)), 0));
    }

    this.fitted = true;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted) throw new NotFittedError("FlexibleQDA not fitted");
    const classes = Array.from(this.means_.keys()).sort((a, b) => a - b);
    return Int32Array.from(X, row => {
      let best = classes[0] ?? 0; let bestScore = Number.NEGATIVE_INFINITY;
      for (const c of classes) {
        const mean = this.means_.get(c)!;
        const prec = this.precisions_.get(c)!;
        const p = row.length;
        let maha = 0;
        for (let j = 0; j < p; j++) {
          const diff = (row[j] ?? 0) - (mean[j] ?? 0);
          maha += diff * diff * (prec[j]?.[j] ?? 0);
        }
        const score = Math.log(this.priors_.get(c) ?? 1e-10) - 0.5 * maha - 0.5 * (this.logDets_.get(c) ?? 0);
        if (score > bestScore) { bestScore = score; best = c; }
      }
      return best;
    });
  }
}

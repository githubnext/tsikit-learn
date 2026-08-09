/**
 * Additional Gaussian Process kernels.
 */

import type { GPKernel } from "./gp.js";

/** Matérn kernel with configurable nu parameter. */
export class MaternKernel implements GPKernel {
  lengthScale: number;
  nu: number;

  constructor(lengthScale = 1.0, nu = 1.5) {
    this.lengthScale = lengthScale;
    this.nu = nu;
  }

  compute(X1: Float64Array[], X2: Float64Array[]): Float64Array[] {
    const n = X1.length;
    const m = X2.length;
    const K: Float64Array[] = Array.from(
      { length: n },
      () => new Float64Array(m),
    );
    for (let i = 0; i < n; i++) {
      const xi = X1[i] ?? new Float64Array(0);
      for (let j = 0; j < m; j++) {
        const xj = X2[j] ?? new Float64Array(0);
        let dSq = 0;
        for (let k = 0; k < xi.length; k++)
          dSq += ((xi[k] ?? 0) - (xj[k] ?? 0)) ** 2;
        const d = Math.sqrt(dSq) / this.lengthScale;
        (K[i] as Float64Array)[j] = this._matern(d);
      }
    }
    return K;
  }

  private _matern(d: number): number {
    if (this.nu === 0.5) return Math.exp(-d);
    if (this.nu === 1.5) {
      const s = Math.SQRT2 * Math.sqrt(3) * d;
      return (1 + s) * Math.exp(-s);
    }
    if (this.nu === 2.5) {
      const s = Math.sqrt(5) * d;
      return (1 + s + (s * s) / 3) * Math.exp(-s);
    }
    // Fallback: approximate as RBF
    return Math.exp(-0.5 * d * d);
  }

  diag(X: Float64Array[]): Float64Array {
    return new Float64Array(X.length).fill(1);
  }
}

/** Linear (dot product) kernel: k(x, y) = sigma_0^2 + x · y */
export class DotProductKernel implements GPKernel {
  sigma0: number;

  constructor(sigma0 = 0.0) {
    this.sigma0 = sigma0;
  }

  compute(X1: Float64Array[], X2: Float64Array[]): Float64Array[] {
    const n = X1.length;
    const m = X2.length;
    const K: Float64Array[] = Array.from(
      { length: n },
      () => new Float64Array(m),
    );
    for (let i = 0; i < n; i++) {
      const xi = X1[i] ?? new Float64Array(0);
      for (let j = 0; j < m; j++) {
        const xj = X2[j] ?? new Float64Array(0);
        let dot = this.sigma0 ** 2;
        for (let k = 0; k < xi.length; k++) dot += (xi[k] ?? 0) * (xj[k] ?? 0);
        (K[i] as Float64Array)[j] = dot;
      }
    }
    return K;
  }

  diag(X: Float64Array[]): Float64Array {
    return Float64Array.from(X, (xi) => {
      let dot = this.sigma0 ** 2;
      for (let k = 0; k < xi.length; k++) dot += (xi[k] ?? 0) ** 2;
      return dot;
    });
  }
}

/** Rational quadratic kernel: k(x,y) = (1 + d^2/(2*alpha*l^2))^(-alpha) */
export class RationalQuadraticKernel implements GPKernel {
  lengthScale: number;
  alpha: number;

  constructor(lengthScale = 1.0, alpha = 1.0) {
    this.lengthScale = lengthScale;
    this.alpha = alpha;
  }

  compute(X1: Float64Array[], X2: Float64Array[]): Float64Array[] {
    const n = X1.length;
    const m = X2.length;
    const K: Float64Array[] = Array.from(
      { length: n },
      () => new Float64Array(m),
    );
    for (let i = 0; i < n; i++) {
      const xi = X1[i] ?? new Float64Array(0);
      for (let j = 0; j < m; j++) {
        const xj = X2[j] ?? new Float64Array(0);
        let dSq = 0;
        for (let k = 0; k < xi.length; k++)
          dSq += ((xi[k] ?? 0) - (xj[k] ?? 0)) ** 2;
        (K[i] as Float64Array)[j] =
          (1 + dSq / (2 * this.alpha * this.lengthScale ** 2)) ** -this.alpha;
      }
    }
    return K;
  }

  diag(X: Float64Array[]): Float64Array {
    return new Float64Array(X.length).fill(1);
  }
}

/** White noise kernel: k(x,y) = noise_level^2 * delta(x,y) */
export class WhiteKernel implements GPKernel {
  noiseLevel: number;

  constructor(noiseLevel = 1.0) {
    this.noiseLevel = noiseLevel;
  }

  compute(X1: Float64Array[], X2: Float64Array[]): Float64Array[] {
    const n = X1.length;
    const m = X2.length;
    const K: Float64Array[] = Array.from(
      { length: n },
      () => new Float64Array(m),
    );
    const noiseSq = this.noiseLevel ** 2;
    for (let i = 0; i < n; i++) {
      const xi = X1[i] ?? new Float64Array(0);
      for (let j = 0; j < m; j++) {
        const xj = X2[j] ?? new Float64Array(0);
        let same = xi.length === xj.length;
        if (same) {
          for (let k = 0; k < xi.length; k++) {
            if ((xi[k] ?? 0) !== (xj[k] ?? 0)) {
              same = false;
              break;
            }
          }
        }
        (K[i] as Float64Array)[j] = same ? noiseSq : 0;
      }
    }
    return K;
  }

  diag(X: Float64Array[]): Float64Array {
    return new Float64Array(X.length).fill(this.noiseLevel ** 2);
  }
}

/** Exp-Sine-Squared (periodic) kernel: k(x,y) = exp(-2*sin^2(pi*d/p)/l^2) */
export class ExpSineSquaredKernel implements GPKernel {
  lengthScale: number;
  periodicity: number;

  constructor(lengthScale = 1.0, periodicity = 1.0) {
    this.lengthScale = lengthScale;
    this.periodicity = periodicity;
  }

  compute(X1: Float64Array[], X2: Float64Array[]): Float64Array[] {
    const n = X1.length;
    const m = X2.length;
    const K: Float64Array[] = Array.from(
      { length: n },
      () => new Float64Array(m),
    );
    for (let i = 0; i < n; i++) {
      const xi = X1[i] ?? new Float64Array(0);
      for (let j = 0; j < m; j++) {
        const xj = X2[j] ?? new Float64Array(0);
        let dSq = 0;
        for (let k = 0; k < xi.length; k++)
          dSq += ((xi[k] ?? 0) - (xj[k] ?? 0)) ** 2;
        const d = Math.sqrt(dSq);
        const s = Math.sin((Math.PI * d) / this.periodicity);
        (K[i] as Float64Array)[j] = Math.exp(
          (-2 * s * s) / this.lengthScale ** 2,
        );
      }
    }
    return K;
  }

  diag(X: Float64Array[]): Float64Array {
    return new Float64Array(X.length).fill(1);
  }
}

/** Sum of two kernels: k(x,y) = k1(x,y) + k2(x,y) */
export class SumKernel implements GPKernel {
  k1: GPKernel;
  k2: GPKernel;

  constructor(k1: GPKernel, k2: GPKernel) {
    this.k1 = k1;
    this.k2 = k2;
  }

  compute(X1: Float64Array[], X2: Float64Array[]): Float64Array[] {
    const K1 = this.k1.compute(X1, X2);
    const K2 = this.k2.compute(X1, X2);
    return K1.map((row, i) => {
      const r2 = K2[i] ?? new Float64Array(row.length);
      const out = new Float64Array(row.length);
      for (let j = 0; j < row.length; j++)
        out[j] = (row[j] ?? 0) + (r2[j] ?? 0);
      return out;
    });
  }

  diag(X: Float64Array[]): Float64Array {
    const d1 = this.k1.diag(X);
    const d2 = this.k2.diag(X);
    const out = new Float64Array(d1.length);
    for (let i = 0; i < d1.length; i++) out[i] = (d1[i] ?? 0) + (d2[i] ?? 0);
    return out;
  }
}

/** Product of two kernels: k(x,y) = k1(x,y) * k2(x,y) */
export class ProductKernel implements GPKernel {
  k1: GPKernel;
  k2: GPKernel;

  constructor(k1: GPKernel, k2: GPKernel) {
    this.k1 = k1;
    this.k2 = k2;
  }

  compute(X1: Float64Array[], X2: Float64Array[]): Float64Array[] {
    const K1 = this.k1.compute(X1, X2);
    const K2 = this.k2.compute(X1, X2);
    return K1.map((row, i) => {
      const r2 = K2[i] ?? new Float64Array(row.length);
      const out = new Float64Array(row.length);
      for (let j = 0; j < row.length; j++)
        out[j] = (row[j] ?? 0) * (r2[j] ?? 0);
      return out;
    });
  }

  diag(X: Float64Array[]): Float64Array {
    const d1 = this.k1.diag(X);
    const d2 = this.k2.diag(X);
    const out = new Float64Array(d1.length);
    for (let i = 0; i < d1.length; i++) out[i] = (d1[i] ?? 0) * (d2[i] ?? 0);
    return out;
  }
}

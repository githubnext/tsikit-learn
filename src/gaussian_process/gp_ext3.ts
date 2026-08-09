/**
 * Additional Gaussian Process kernels: Matern, RationalQuadratic extensions.
 * Mirrors sklearn.gaussian_process.kernels extras.
 */

export interface Kernel {
  call(X1: Float64Array[], X2?: Float64Array[]): Float64Array[];
  diag(X: Float64Array[]): Float64Array;
  theta: Float64Array;
  nParams: number;
}

export class MaternKernel implements Kernel {
  lengthScale: number;
  nu: number;

  constructor(options: { lengthScale?: number; nu?: number } = {}) {
    this.lengthScale = options.lengthScale ?? 1.0;
    this.nu = options.nu ?? 1.5;
  }

  get theta(): Float64Array {
    return new Float64Array([Math.log(this.lengthScale)]);
  }

  get nParams(): number {
    return 1;
  }

  private _dist(a: Float64Array, b: Float64Array): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
    return Math.sqrt(s);
  }

  private _matern(d: number): number {
    const r = d / this.lengthScale;
    if (this.nu === 0.5) {
      return Math.exp(-r);
    } else if (this.nu === 1.5) {
      return (1 + Math.SQRT2 * r) * Math.exp(-Math.SQRT2 * r);
    } else if (this.nu === 2.5) {
      return (1 + Math.sqrt(5) * r + 5 * r ** 2 / 3) * Math.exp(-Math.sqrt(5) * r);
    } else {
      // Approx with infinite smoothness (RBF limit)
      return Math.exp(-(r ** 2) / 2);
    }
  }

  call(X1: Float64Array[], X2?: Float64Array[]): Float64Array[] {
    const Y = X2 ?? X1;
    return X1.map((x1) => {
      const row = new Float64Array(Y.length);
      for (let j = 0; j < Y.length; j++) {
        row[j] = this._matern(this._dist(x1, Y[j] ?? new Float64Array(0)));
      }
      return row;
    });
  }

  diag(X: Float64Array[]): Float64Array {
    return new Float64Array(X.length).fill(1.0);
  }
}

export class RationalQuadraticKernel implements Kernel {
  lengthScale: number;
  alpha: number;

  constructor(options: { lengthScale?: number; alpha?: number } = {}) {
    this.lengthScale = options.lengthScale ?? 1.0;
    this.alpha = options.alpha ?? 1.0;
  }

  get theta(): Float64Array {
    return new Float64Array([Math.log(this.lengthScale), Math.log(this.alpha)]);
  }

  get nParams(): number {
    return 2;
  }

  private _dist2(a: Float64Array, b: Float64Array): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
    return s;
  }

  call(X1: Float64Array[], X2?: Float64Array[]): Float64Array[] {
    const Y = X2 ?? X1;
    return X1.map((x1) => {
      const row = new Float64Array(Y.length);
      for (let j = 0; j < Y.length; j++) {
        const d2 = this._dist2(x1, Y[j] ?? new Float64Array(0));
        row[j] = Math.pow(1 + d2 / (2 * this.alpha * this.lengthScale ** 2), -this.alpha);
      }
      return row;
    });
  }

  diag(X: Float64Array[]): Float64Array {
    return new Float64Array(X.length).fill(1.0);
  }
}

export class ExpSineSquaredKernel implements Kernel {
  lengthScale: number;
  periodicity: number;

  constructor(options: { lengthScale?: number; periodicity?: number } = {}) {
    this.lengthScale = options.lengthScale ?? 1.0;
    this.periodicity = options.periodicity ?? 1.0;
  }

  get theta(): Float64Array {
    return new Float64Array([Math.log(this.lengthScale), Math.log(this.periodicity)]);
  }

  get nParams(): number {
    return 2;
  }

  private _dist(a: Float64Array, b: Float64Array): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
    return Math.sqrt(s);
  }

  call(X1: Float64Array[], X2?: Float64Array[]): Float64Array[] {
    const Y = X2 ?? X1;
    return X1.map((x1) => {
      const row = new Float64Array(Y.length);
      for (let j = 0; j < Y.length; j++) {
        const d = this._dist(x1, Y[j] ?? new Float64Array(0));
        const sinVal = Math.sin(Math.PI * d / this.periodicity);
        row[j] = Math.exp(-2 * sinVal ** 2 / this.lengthScale ** 2);
      }
      return row;
    });
  }

  diag(X: Float64Array[]): Float64Array {
    return new Float64Array(X.length).fill(1.0);
  }
}

export class KernelSum implements Kernel {
  k1: Kernel;
  k2: Kernel;

  constructor(k1: Kernel, k2: Kernel) {
    this.k1 = k1;
    this.k2 = k2;
  }

  get theta(): Float64Array {
    const t1 = this.k1.theta;
    const t2 = this.k2.theta;
    const out = new Float64Array(t1.length + t2.length);
    out.set(t1);
    out.set(t2, t1.length);
    return out;
  }

  get nParams(): number {
    return this.k1.nParams + this.k2.nParams;
  }

  call(X1: Float64Array[], X2?: Float64Array[]): Float64Array[] {
    const K1 = this.k1.call(X1, X2);
    const K2 = this.k2.call(X1, X2);
    return K1.map((row, i) => {
      const out = new Float64Array(row.length);
      for (let j = 0; j < row.length; j++) out[j] = (row[j] ?? 0) + (K2[i]?.[j] ?? 0);
      return out;
    });
  }

  diag(X: Float64Array[]): Float64Array {
    const d1 = this.k1.diag(X);
    const d2 = this.k2.diag(X);
    return d1.map((v, i) => v + (d2[i] ?? 0));
  }
}

export class KernelProduct implements Kernel {
  k1: Kernel;
  k2: Kernel;

  constructor(k1: Kernel, k2: Kernel) {
    this.k1 = k1;
    this.k2 = k2;
  }

  get theta(): Float64Array {
    const t1 = this.k1.theta;
    const t2 = this.k2.theta;
    const out = new Float64Array(t1.length + t2.length);
    out.set(t1);
    out.set(t2, t1.length);
    return out;
  }

  get nParams(): number {
    return this.k1.nParams + this.k2.nParams;
  }

  call(X1: Float64Array[], X2?: Float64Array[]): Float64Array[] {
    const K1 = this.k1.call(X1, X2);
    const K2 = this.k2.call(X1, X2);
    return K1.map((row, i) => {
      const out = new Float64Array(row.length);
      for (let j = 0; j < row.length; j++) out[j] = (row[j] ?? 0) * (K2[i]?.[j] ?? 0);
      return out;
    });
  }

  diag(X: Float64Array[]): Float64Array {
    const d1 = this.k1.diag(X);
    const d2 = this.k2.diag(X);
    return d1.map((v, i) => v * (d2[i] ?? 0));
  }
}

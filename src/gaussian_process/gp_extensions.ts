/**
 * Gaussian Process extensions.
 * Mirrors scikit-learn's gaussian_process kernel combinations and GP utilities.
 */

export interface Kernel {
  __call__(X: Float64Array[], Y?: Float64Array[]): Float64Array[];
  diag(X: Float64Array[]): Float64Array;
  isStationary(): boolean;
  getParams(): Record<string, number>;
}

/** Squared Exponential (RBF) kernel */
export class RBFKernel implements Kernel {
  constructor(readonly lengthScale = 1.0) {}

  __call__(X: Float64Array[], Y?: Float64Array[]): Float64Array[] {
    const Yp = Y ?? X;
    return X.map((xi) =>
      Float64Array.from(Yp, (yj) => {
        let d = 0;
        for (let k = 0; k < xi.length; k++) d += ((xi[k] ?? 0) - (yj[k] ?? 0)) ** 2;
        return Math.exp(-0.5 * d / (this.lengthScale ** 2));
      }),
    );
  }

  diag(X: Float64Array[]): Float64Array {
    return new Float64Array(X.length).fill(1);
  }

  isStationary(): boolean { return true; }
  getParams(): Record<string, number> { return { length_scale: this.lengthScale }; }
}

/** Matern kernel with nu parameter */
export class MaternKernel implements Kernel {
  constructor(
    readonly lengthScale = 1.0,
    readonly nu: 0.5 | 1.5 | 2.5 = 1.5,
  ) {}

  __call__(X: Float64Array[], Y?: Float64Array[]): Float64Array[] {
    const Yp = Y ?? X;
    const ls = this.lengthScale;
    return X.map((xi) =>
      Float64Array.from(Yp, (yj) => {
        let d = 0;
        for (let k = 0; k < xi.length; k++) d += ((xi[k] ?? 0) - (yj[k] ?? 0)) ** 2;
        d = Math.sqrt(d);
        if (this.nu === 0.5) return Math.exp(-d / ls);
        if (this.nu === 1.5) {
          const t = Math.sqrt(3) * d / ls;
          return (1 + t) * Math.exp(-t);
        }
        // nu = 2.5
        const t = Math.sqrt(5) * d / ls;
        return (1 + t + t * t / 3) * Math.exp(-t);
      }),
    );
  }

  diag(X: Float64Array[]): Float64Array {
    return new Float64Array(X.length).fill(1);
  }

  isStationary(): boolean { return true; }
  getParams(): Record<string, number> {
    return { length_scale: this.lengthScale, nu: this.nu };
  }
}

/** Periodic kernel (Exp-Sine-Squared) */
export class ExpSineSquaredKernel implements Kernel {
  constructor(
    readonly lengthScale = 1.0,
    readonly periodicity = 1.0,
  ) {}

  __call__(X: Float64Array[], Y?: Float64Array[]): Float64Array[] {
    const Yp = Y ?? X;
    return X.map((xi) =>
      Float64Array.from(Yp, (yj) => {
        let d = 0;
        for (let k = 0; k < xi.length; k++) d += ((xi[k] ?? 0) - (yj[k] ?? 0)) ** 2;
        d = Math.sqrt(d);
        const sinVal = Math.sin(Math.PI * d / this.periodicity);
        return Math.exp(-2 * sinVal * sinVal / (this.lengthScale ** 2));
      }),
    );
  }

  diag(X: Float64Array[]): Float64Array {
    return new Float64Array(X.length).fill(1);
  }

  isStationary(): boolean { return true; }
  getParams(): Record<string, number> {
    return { length_scale: this.lengthScale, periodicity: this.periodicity };
  }
}

/** Sum of two kernels */
export class SumKernel implements Kernel {
  constructor(readonly k1: Kernel, readonly k2: Kernel) {}

  __call__(X: Float64Array[], Y?: Float64Array[]): Float64Array[] {
    const K1 = this.k1.__call__(X, Y);
    const K2 = this.k2.__call__(X, Y);
    return K1.map((row, i) => Float64Array.from(row, (v, j) => v + (K2[i]?.[j] ?? 0)));
  }

  diag(X: Float64Array[]): Float64Array {
    const d1 = this.k1.diag(X);
    const d2 = this.k2.diag(X);
    return Float64Array.from(d1, (v, i) => v + (d2[i] ?? 0));
  }

  isStationary(): boolean {
    return this.k1.isStationary() && this.k2.isStationary();
  }

  getParams(): Record<string, number> {
    return { ...this.k1.getParams(), ...this.k2.getParams() };
  }
}

/** Product of two kernels */
export class ProductKernel implements Kernel {
  constructor(readonly k1: Kernel, readonly k2: Kernel) {}

  __call__(X: Float64Array[], Y?: Float64Array[]): Float64Array[] {
    const K1 = this.k1.__call__(X, Y);
    const K2 = this.k2.__call__(X, Y);
    return K1.map((row, i) => Float64Array.from(row, (v, j) => v * (K2[i]?.[j] ?? 0)));
  }

  diag(X: Float64Array[]): Float64Array {
    const d1 = this.k1.diag(X);
    const d2 = this.k2.diag(X);
    return Float64Array.from(d1, (v, i) => v * (d2[i] ?? 0));
  }

  isStationary(): boolean {
    return this.k1.isStationary() && this.k2.isStationary();
  }

  getParams(): Record<string, number> {
    return { ...this.k1.getParams(), ...this.k2.getParams() };
  }
}

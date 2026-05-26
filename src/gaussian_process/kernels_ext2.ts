/**
 * Extended GP kernels: WhiteKernel, ConstantKernel, SumKernel, ProductKernel, RationalQuadratic, ExpSineSquared, DotProduct
 */

export interface Kernel {
  call(X1: Float64Array[], X2?: Float64Array[]): Float64Array[];
  diag(X: Float64Array[]): Float64Array;
  clone(): Kernel;
}

function squaredDist(a: Float64Array, b: Float64Array): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  return d;
}

export class WhiteKernel implements Kernel {
  noiseLevel: number;
  constructor(noiseLevel = 1.0) { this.noiseLevel = noiseLevel; }

  call(X1: Float64Array[], X2?: Float64Array[]): Float64Array[] {
    const n1 = X1.length;
    const n2 = X2?.length ?? n1;
    return Array.from({ length: n1 }, (_, i) => {
      const row = new Float64Array(n2);
      if (!X2) row[i] = this.noiseLevel;
      return row;
    });
  }

  diag(X: Float64Array[]): Float64Array {
    return new Float64Array(X.length).fill(this.noiseLevel);
  }

  clone(): WhiteKernel { return new WhiteKernel(this.noiseLevel); }
}

export class ConstantKernel implements Kernel {
  constantValue: number;
  constructor(constantValue = 1.0) { this.constantValue = constantValue; }

  call(X1: Float64Array[], X2?: Float64Array[]): Float64Array[] {
    const n1 = X1.length;
    const n2 = X2?.length ?? n1;
    return Array.from({ length: n1 }, () => new Float64Array(n2).fill(this.constantValue));
  }

  diag(X: Float64Array[]): Float64Array {
    return new Float64Array(X.length).fill(this.constantValue);
  }

  clone(): ConstantKernel { return new ConstantKernel(this.constantValue); }
}

export class SumKernel implements Kernel {
  k1: Kernel;
  k2: Kernel;
  constructor(k1: Kernel, k2: Kernel) { this.k1 = k1; this.k2 = k2; }

  call(X1: Float64Array[], X2?: Float64Array[]): Float64Array[] {
    const K1 = this.k1.call(X1, X2);
    const K2 = this.k2.call(X1, X2);
    return K1.map((row, i) => row.map((v, j) => v + (K2[i]![j] ?? 0)) as unknown as Float64Array);
  }

  diag(X: Float64Array[]): Float64Array {
    const d1 = this.k1.diag(X), d2 = this.k2.diag(X);
    return d1.map((v, i) => v + (d2[i] ?? 0)) as unknown as Float64Array;
  }

  clone(): SumKernel { return new SumKernel(this.k1.clone(), this.k2.clone()); }
}

export class ProductKernel implements Kernel {
  k1: Kernel;
  k2: Kernel;
  constructor(k1: Kernel, k2: Kernel) { this.k1 = k1; this.k2 = k2; }

  call(X1: Float64Array[], X2?: Float64Array[]): Float64Array[] {
    const K1 = this.k1.call(X1, X2);
    const K2 = this.k2.call(X1, X2);
    return K1.map((row, i) => row.map((v, j) => v * (K2[i]![j] ?? 0)) as unknown as Float64Array);
  }

  diag(X: Float64Array[]): Float64Array {
    const d1 = this.k1.diag(X), d2 = this.k2.diag(X);
    return d1.map((v, i) => v * (d2[i] ?? 0)) as unknown as Float64Array;
  }

  clone(): ProductKernel { return new ProductKernel(this.k1.clone(), this.k2.clone()); }
}

export class RationalQuadraticKernel implements Kernel {
  lengthScale: number;
  alpha: number;
  constructor(lengthScale = 1.0, alpha = 1.0) { this.lengthScale = lengthScale; this.alpha = alpha; }

  call(X1: Float64Array[], X2?: Float64Array[]): Float64Array[] {
    const Xb = X2 ?? X1;
    return X1.map((xi) =>
      new Float64Array(Xb.map((xj) =>
        (1 + squaredDist(xi, xj) / (2 * this.alpha * this.lengthScale ** 2)) ** (-this.alpha)
      ))
    );
  }

  diag(X: Float64Array[]): Float64Array { return new Float64Array(X.length).fill(1); }
  clone(): RationalQuadraticKernel { return new RationalQuadraticKernel(this.lengthScale, this.alpha); }
}

export class ExpSineSquaredKernel implements Kernel {
  lengthScale: number;
  periodicity: number;
  constructor(lengthScale = 1.0, periodicity = 1.0) { this.lengthScale = lengthScale; this.periodicity = periodicity; }

  call(X1: Float64Array[], X2?: Float64Array[]): Float64Array[] {
    const Xb = X2 ?? X1;
    return X1.map((xi) =>
      new Float64Array(Xb.map((xj) => {
        const dist = Math.sqrt(squaredDist(xi, xj));
        return Math.exp(-2 * Math.sin(Math.PI * dist / this.periodicity) ** 2 / this.lengthScale ** 2);
      }))
    );
  }

  diag(X: Float64Array[]): Float64Array { return new Float64Array(X.length).fill(1); }
  clone(): ExpSineSquaredKernel { return new ExpSineSquaredKernel(this.lengthScale, this.periodicity); }
}

export class DotProductKernel implements Kernel {
  sigma0: number;
  constructor(sigma0 = 1.0) { this.sigma0 = sigma0; }

  call(X1: Float64Array[], X2?: Float64Array[]): Float64Array[] {
    const Xb = X2 ?? X1;
    return X1.map((xi) =>
      new Float64Array(Xb.map((xj) => {
        let dot = this.sigma0 ** 2;
        for (let k = 0; k < xi.length; k++) dot += (xi[k] ?? 0) * (xj[k] ?? 0);
        return dot;
      }))
    );
  }

  diag(X: Float64Array[]): Float64Array {
    return new Float64Array(X.map((xi) => {
      let d = this.sigma0 ** 2;
      for (const v of xi) d += v * v;
      return d;
    }));
  }

  clone(): DotProductKernel { return new DotProductKernel(this.sigma0); }
}

export class MaternKernelExt implements Kernel {
  lengthScale: number;
  nu: number;
  constructor(lengthScale = 1.0, nu = 1.5) { this.lengthScale = lengthScale; this.nu = nu; }

  call(X1: Float64Array[], X2?: Float64Array[]): Float64Array[] {
    const Xb = X2 ?? X1;
    const ls = this.lengthScale;
    return X1.map((xi) =>
      new Float64Array(Xb.map((xj) => {
        const d = Math.sqrt(squaredDist(xi, xj)) / ls;
        if (this.nu === 0.5) return Math.exp(-d);
        if (this.nu === 1.5) return (1 + Math.sqrt(3) * d) * Math.exp(-Math.sqrt(3) * d);
        if (this.nu === 2.5) return (1 + Math.sqrt(5) * d + 5 * d * d / 3) * Math.exp(-Math.sqrt(5) * d);
        return Math.exp(-d * d / 2); // approx for other nu
      }))
    );
  }

  diag(X: Float64Array[]): Float64Array { return new Float64Array(X.length).fill(1); }
  clone(): MaternKernelExt { return new MaternKernelExt(this.lengthScale, this.nu); }
}

/**
 * GP utility functions and SparseGaussianProcess — gaussian process extensions.
 */

export function rbfKernel(X1: Float64Array[], X2: Float64Array[], lengthScale = 1.0, variance = 1.0): Float64Array[] {
  const n1 = X1.length, n2 = X2.length;
  const K: Float64Array[] = Array.from({ length: n1 }, () => new Float64Array(n2));
  for (let i = 0; i < n1; i++) {
    for (let j = 0; j < n2; j++) {
      let dist2 = 0;
      const p = X1[i]?.length ?? 0;
      for (let d = 0; d < p; d++) {
        const diff = ((X1[i]?.[d] ?? 0) - (X2[j]?.[d] ?? 0)) / lengthScale;
        dist2 += diff * diff;
      }
      (K[i] as Float64Array)[j] = variance * Math.exp(-0.5 * dist2);
    }
  }
  return K;
}

export function maternKernel(X1: Float64Array[], X2: Float64Array[], lengthScale = 1.0, nu = 1.5): Float64Array[] {
  const n1 = X1.length, n2 = X2.length;
  const K: Float64Array[] = Array.from({ length: n1 }, () => new Float64Array(n2));
  for (let i = 0; i < n1; i++) {
    for (let j = 0; j < n2; j++) {
      let dist2 = 0;
      const p = X1[i]?.length ?? 0;
      for (let d = 0; d < p; d++) {
        const diff = ((X1[i]?.[d] ?? 0) - (X2[j]?.[d] ?? 0)) / lengthScale;
        dist2 += diff * diff;
      }
      const r = Math.sqrt(dist2);
      let k: number;
      if (nu === 0.5) {
        k = Math.exp(-r);
      } else if (nu === 1.5) {
        const t = Math.sqrt(3) * r;
        k = (1 + t) * Math.exp(-t);
      } else if (nu === 2.5) {
        const t = Math.sqrt(5) * r;
        k = (1 + t + t * t / 3) * Math.exp(-t);
      } else {
        k = Math.exp(-r);
      }
      (K[i] as Float64Array)[j] = k;
    }
  }
  return K;
}

export function rationalQuadraticKernel(X1: Float64Array[], X2: Float64Array[], lengthScale = 1.0, alpha = 1.0): Float64Array[] {
  const n1 = X1.length, n2 = X2.length;
  const K: Float64Array[] = Array.from({ length: n1 }, () => new Float64Array(n2));
  for (let i = 0; i < n1; i++) {
    for (let j = 0; j < n2; j++) {
      let dist2 = 0;
      const p = X1[i]?.length ?? 0;
      for (let d = 0; d < p; d++) {
        const diff = (X1[i]?.[d] ?? 0) - (X2[j]?.[d] ?? 0);
        dist2 += diff * diff;
      }
      (K[i] as Float64Array)[j] = (1 + dist2 / (2 * alpha * lengthScale ** 2)) ** (-alpha);
    }
  }
  return K;
}

export function addKernels(K1: Float64Array[], K2: Float64Array[]): Float64Array[] {
  return K1.map((row, i) => row.map((v, j) => v + ((K2[i] as Float64Array)[j] ?? 0)));
}

export function multiplyKernels(K1: Float64Array[], K2: Float64Array[]): Float64Array[] {
  return K1.map((row, i) => row.map((v, j) => v * ((K2[i] as Float64Array)[j] ?? 0)));
}

export class GPUtilities {
  static choleskyDecompose(K: Float64Array[]): Float64Array[] {
    const n = K.length;
    const L: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = K[i]?.[j] ?? 0;
        for (let k = 0; k < j; k++) s -= ((L[i] as Float64Array)[k] ?? 0) * ((L[j] as Float64Array)[k] ?? 0);
        if (i === j) {
          (L[i] as Float64Array)[j] = Math.sqrt(Math.max(s, 1e-12));
        } else {
          (L[i] as Float64Array)[j] = s / ((L[j] as Float64Array)[j] ?? 1e-12);
        }
      }
    }
    return L;
  }

  static solveLower(L: Float64Array[], b: Float64Array): Float64Array {
    const n = b.length;
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = b[i] ?? 0;
      for (let j = 0; j < i; j++) s -= ((L[i] as Float64Array)[j] ?? 0) * (x[j] ?? 0);
      x[i] = s / ((L[i] as Float64Array)[i] ?? 1e-12);
    }
    return x;
  }

  static solveUpper(L: Float64Array[], b: Float64Array): Float64Array {
    const n = b.length;
    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let s = b[i] ?? 0;
      for (let j = i + 1; j < n; j++) s -= ((L[j] as Float64Array)[i] ?? 0) * (x[j] ?? 0);
      x[i] = s / ((L[i] as Float64Array)[i] ?? 1e-12);
    }
    return x;
  }

  static logDeterminant(L: Float64Array[]): number {
    return 2 * L.reduce((s, row, i) => s + Math.log(Math.max((row as Float64Array)[i] ?? 1e-12, 1e-12)), 0);
  }
}

export class SparseGaussianProcess {
  nInducingPoints: number;
  lengthScale: number;
  noiseVar: number;
  private inducingPoints: Float64Array[] | null = null;
  private alpha_: Float64Array | null = null;
  private L_: Float64Array[] | null = null;
  private trainX_: Float64Array[] | null = null;

  constructor(nInducingPoints = 20, lengthScale = 1.0, noiseVar = 0.1) {
    this.nInducingPoints = nInducingPoints;
    this.lengthScale = lengthScale;
    this.noiseVar = noiseVar;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const m = Math.min(this.nInducingPoints, n);
    // Select inducing points via random subsampling
    const indices = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const t = indices[i]; indices[i] = indices[j] as number; indices[j] = t as number;
    }
    this.inducingPoints = indices.slice(0, m).map((i) => new Float64Array(X[i] as Float64Array));
    this.trainX_ = X;

    // FITC approximation: compute Kuu, Kuf
    const Kuu = rbfKernel(this.inducingPoints, this.inducingPoints, this.lengthScale);
    const Kuf = rbfKernel(this.inducingPoints, X, this.lengthScale);

    // Diagonal of Kff
    const kff_diag = X.map(() => 1.0);
    const Qff_diag = new Float64Array(n);
    const L = GPUtilities.choleskyDecompose(Kuu.map((row, i) => row.map((v, j) => v + (i === j ? 1e-6 : 0))));

    for (let i = 0; i < n; i++) {
      const kuf_col = Float64Array.from({ length: m }, (_, k) => (Kuf[k] as Float64Array)[i] ?? 0);
      const v = GPUtilities.solveLower(L, kuf_col);
      Qff_diag[i] = v.reduce((s, vi) => s + vi * vi, 0);
    }

    // Lambda = diag(Kff - Qff + noise)
    const lambda = Float64Array.from({ length: n }, (_, i) => Math.max(kff_diag[i] ?? 1 - (Qff_diag[i] ?? 0), this.noiseVar) + this.noiseVar);

    // Posterior update
    const B: Float64Array[] = Array.from({ length: m }, () => new Float64Array(m));
    const c = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      const li = Kuu[i] as Float64Array;
      void li;
      for (let j = 0; j < m; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += ((Kuf[i] as Float64Array)[k] ?? 0) * ((Kuf[j] as Float64Array)[k] ?? 0) / (lambda[k] ?? 1);
        (B[i] as Float64Array)[j] = s + ((Kuu[i] as Float64Array)[j] ?? 0);
      }
      for (let k = 0; k < n; k++) c[i]! += ((Kuf[i] as Float64Array)[k] ?? 0) * (y[k] ?? 0) / (lambda[k] ?? 1);
    }
    this.L_ = GPUtilities.choleskyDecompose(B);
    const Lc = GPUtilities.solveLower(this.L_, c);
    this.alpha_ = GPUtilities.solveUpper(this.L_, Lc);
    return this;
  }

  predict(Xstar: Float64Array[]): { mean: Float64Array; variance: Float64Array } {
    if (!this.inducingPoints || !this.alpha_ || !this.L_) throw new Error("Not fitted");
    const Kus = rbfKernel(this.inducingPoints, Xstar, this.lengthScale);
    const n = Xstar.length;
    const mean = new Float64Array(n);
    const variance = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const kus_col = Float64Array.from({ length: this.inducingPoints.length }, (_, k) => (Kus[k] as Float64Array)[i] ?? 0);
      mean[i] = kus_col.reduce((s, v, k) => s + v * (this.alpha_?.[k] ?? 0), 0);
      const v = GPUtilities.solveLower(this.L_ as Float64Array[], kus_col);
      variance[i] = Math.max(0, 1.0 - v.reduce((s, vi) => s + vi * vi, 0));
    }
    return { mean, variance };
  }
}

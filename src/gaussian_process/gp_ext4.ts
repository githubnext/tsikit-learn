/**
 * SparseGaussianProcess with additional kernels and GP utilities.
 */

export class GaussianProcessKernels {
  static constantKernel(X1: Float64Array[], X2: Float64Array[], constantValue = 1.0): Float64Array[] {
    return X1.map(() => Float64Array.from(X2, () => constantValue));
  }

  static dotProductKernel(X1: Float64Array[], X2: Float64Array[], sigma0 = 0.0): Float64Array[] {
    return X1.map((x1) => Float64Array.from(X2, (x2) => {
      const dot = x1.reduce((s, v, d) => s + v * (x2[d] ?? 0), 0);
      return dot + sigma0 ** 2;
    }));
  }

  static periodicKernel(X1: Float64Array[], X2: Float64Array[], period = 1.0, lengthScale = 1.0): Float64Array[] {
    return X1.map((x1) => Float64Array.from(X2, (x2) => {
      const dist = Math.sqrt(x1.reduce((s, v, d) => s + (v - (x2[d] ?? 0)) ** 2, 0));
      const sin = Math.sin(Math.PI * dist / period);
      return Math.exp(-2 * sin * sin / lengthScale ** 2);
    }));
  }

  static whiteKernel(X1: Float64Array[], noiseLevel = 1.0): Float64Array[] {
    return X1.map((_, i) => Float64Array.from({ length: X1.length }, (__, j) => i === j ? noiseLevel : 0));
  }
}

export class GPPrediction {
  mean: Float64Array;
  stdDev: Float64Array;
  covMatrix: Float64Array[] | null;

  constructor(mean: Float64Array, variance: Float64Array, covMatrix: Float64Array[] | null = null) {
    this.mean = mean;
    this.stdDev = variance.map((v) => Math.sqrt(Math.max(v, 0)));
    this.covMatrix = covMatrix;
  }

  confidenceInterval(alpha = 0.95): { lower: Float64Array; upper: Float64Array } {
    const z = alpha === 0.95 ? 1.96 : alpha === 0.99 ? 2.576 : 1.645;
    return {
      lower: Float64Array.from(this.mean, (m, i) => m - z * (this.stdDev[i] ?? 0)),
      upper: Float64Array.from(this.mean, (m, i) => m + z * (this.stdDev[i] ?? 0)),
    };
  }

  quantile(q: number): Float64Array {
    // Approximate using normal distribution
    const z = Math.sqrt(2) * (q > 0.5 ? 1 : -1) * Math.abs(Math.log(2 * Math.abs(q - 0.5)) * -1);
    return Float64Array.from(this.mean, (m, i) => m + z * (this.stdDev[i] ?? 0));
  }
}

export class GaussianProcessRegressor {
  kernel: "rbf" | "matern" | "dot" | "periodic";
  alpha: number;
  nRestarts: number;
  lengthScale: number;
  noiseLevel: number;
  private _XTrain: Float64Array[] | null = null;
  private _yTrain: Float64Array | null = null;
  private _L: Float64Array[] | null = null;
  private _alpha_: Float64Array | null = null;
  private _logLikelihood: number = 0;
  nFeaturesIn_: number = 0;

  constructor(
    kernel: "rbf" | "matern" | "dot" | "periodic" = "rbf",
    alpha = 1e-10,
    nRestarts = 0,
    lengthScale = 1.0,
    noiseLevel = 1e-10,
  ) {
    this.kernel = kernel;
    this.alpha = alpha;
    this.nRestarts = nRestarts;
    this.lengthScale = lengthScale;
    this.noiseLevel = noiseLevel;
  }

  private _computeKernel(X1: Float64Array[], X2: Float64Array[]): Float64Array[] {
    if (this.kernel === "rbf") {
      return X1.map((x1) => Float64Array.from(X2, (x2) => {
        const d2 = x1.reduce((s, v, d) => s + (v - (x2[d] ?? 0)) ** 2, 0);
        return Math.exp(-0.5 * d2 / this.lengthScale ** 2);
      }));
    } else if (this.kernel === "matern") {
      return X1.map((x1) => Float64Array.from(X2, (x2) => {
        const d = Math.sqrt(x1.reduce((s, v, d) => s + (v - (x2[d] ?? 0)) ** 2, 0));
        const t = Math.sqrt(5) * d / this.lengthScale;
        return (1 + t + t * t / 3) * Math.exp(-t);
      }));
    } else if (this.kernel === "periodic") {
      return X1.map((x1) => Float64Array.from(X2, (x2) => {
        const d = Math.sqrt(x1.reduce((s, v, d) => s + (v - (x2[d] ?? 0)) ** 2, 0));
        const sinV = Math.sin(Math.PI * d / this.lengthScale);
        return Math.exp(-2 * sinV * sinV);
      }));
    } else {
      return X1.map((x1) => Float64Array.from(X2, (x2) => x1.reduce((s, v, d) => s + v * (x2[d] ?? 0), 0)));
    }
  }

  private _cholesky(K: Float64Array[]): Float64Array[] {
    const n = K.length;
    const L: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = K[i]?.[j] ?? 0;
        for (let k = 0; k < j; k++) s -= ((L[i] as Float64Array)[k] ?? 0) * ((L[j] as Float64Array)[k] ?? 0);
        if (i === j) {
          (L[i] as Float64Array)[j] = Math.sqrt(Math.max(s, 1e-12));
        } else {
          (L[i] as Float64Array)[j] = s / Math.max((L[j] as Float64Array)[j] ?? 1e-12, 1e-12);
        }
      }
    }
    return L;
  }

  private _solveLower(L: Float64Array[], b: Float64Array): Float64Array {
    const n = b.length, x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = b[i] ?? 0;
      for (let j = 0; j < i; j++) s -= ((L[i] as Float64Array)[j] ?? 0) * (x[j] ?? 0);
      x[i] = s / Math.max((L[i] as Float64Array)[i] ?? 1e-12, 1e-12);
    }
    return x;
  }

  private _solveUpper(L: Float64Array[], b: Float64Array): Float64Array {
    const n = b.length, x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let s = b[i] ?? 0;
      for (let j = i + 1; j < n; j++) s -= ((L[j] as Float64Array)[i] ?? 0) * (x[j] ?? 0);
      x[i] = s / Math.max((L[i] as Float64Array)[i] ?? 1e-12, 1e-12);
    }
    return x;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    this.nFeaturesIn_ = X[0]?.length ?? 0;
    this._XTrain = X;
    this._yTrain = y;

    const K = this._computeKernel(X, X);
    for (let i = 0; i < n; i++) (K[i] as Float64Array)[i]! += this.alpha + this.noiseLevel;

    this._L = this._cholesky(K);
    const Ly = this._solveLower(this._L, y);
    this._alpha_ = this._solveUpper(this._L, Ly);

    const logDet = this._L.reduce((s, row, i) => s + Math.log(Math.max((row as Float64Array)[i] ?? 1e-12, 1e-12)), 0) * 2;
    const yKy = y.reduce((s, yi, i) => s + yi * (this._alpha_![i] ?? 0), 0);
    this._logLikelihood = -0.5 * (yKy + logDet + n * Math.log(2 * Math.PI));
    return this;
  }

  predict(X: Float64Array[], returnStd = false): GPPrediction {
    if (!this._XTrain || !this._alpha_ || !this._L) throw new Error("Not fitted");
    const Kstar = this._computeKernel(X, this._XTrain);
    const mean = Float64Array.from(Kstar, (row) => row.reduce((s, v, j) => s + v * (this._alpha_![j] ?? 0), 0));
    let variance: Float64Array;
    if (returnStd) {
      variance = Float64Array.from(Kstar, (row) => {
        const v = this._solveLower(this._L as Float64Array[], row);
        const kxx = 1.0;
        return Math.max(0, kxx - v.reduce((s, vi) => s + vi * vi, 0));
      });
    } else {
      variance = new Float64Array(X.length);
    }
    return new GPPrediction(mean, variance);
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    const yMean = y.reduce((s, v) => s + v, 0) / y.length;
    const ss_res = pred.mean.reduce((s, v, i) => s + (v - (y[i] ?? 0)) ** 2, 0);
    const ss_tot = y.reduce((s, v) => s + (v - yMean) ** 2, 0);
    return 1 - ss_res / Math.max(ss_tot, 1e-12);
  }

  get logMarginalLikelihood(): number {
    return this._logLikelihood;
  }
}

/**
 * FastICA, TruncatedSVD, and NMF decomposition extensions.
 */

function kurtosis(x: Float64Array): number {
  const n = x.length;
  const mean = x.reduce((s, v) => s + v, 0) / n;
  const std = Math.sqrt(x.reduce((s, v) => s + (v - mean) ** 2, 0) / n);
  if (std < 1e-12) return 0;
  const m4 = x.reduce((s, v) => s + ((v - mean) / std) ** 4, 0) / n;
  return m4 - 3;
}

export class FastICA {
  nComponents: number;
  algorithm: "parallel" | "deflation";
  funName: "logcosh" | "exp" | "cube";
  maxIter: number;
  tol: number;
  randomState: number;
  components_: Float64Array[] | null = null;
  mixingMatrix_: Float64Array[] | null = null;
  nIter_: number = 0;
  private _whiteningMatrix: Float64Array[] | null = null;
  nFeaturesIn_: number = 0;

  constructor(
    nComponents = 2,
    algorithm: "parallel" | "deflation" = "parallel",
    funName: "logcosh" | "exp" | "cube" = "logcosh",
    maxIter = 200,
    tol = 1e-4,
    randomState = 42,
  ) {
    this.nComponents = nComponents;
    this.algorithm = algorithm;
    this.funName = funName;
    this.maxIter = maxIter;
    this.tol = tol;
    this.randomState = randomState;
  }

  private _g(u: Float64Array): { g: Float64Array; gPrime: Float64Array } {
    if (this.funName === "logcosh") {
      return {
        g: u.map((v) => Math.tanh(v)),
        gPrime: u.map((v) => 1 - Math.tanh(v) ** 2),
      };
    } else if (this.funName === "exp") {
      return {
        g: u.map((v) => v * Math.exp(-v * v / 2)),
        gPrime: u.map((v) => (1 - v * v) * Math.exp(-v * v / 2)),
      };
    } else {
      return {
        g: u.map((v) => v * v * v),
        gPrime: u.map((v) => 3 * v * v),
      };
    }
  }

  private _center(X: Float64Array[]): { Xc: Float64Array[]; mean: Float64Array } {
    const n = X.length, p = X[0]?.length ?? 0;
    const mean = new Float64Array(p);
    for (const row of X) for (let j = 0; j < p; j++) mean[j]! += (row[j] ?? 0) / n;
    const Xc = X.map((row) => row.map((v, j) => v - (mean[j] ?? 0)));
    return { Xc, mean };
  }

  private _whiten(X: Float64Array[]): { Xw: Float64Array[]; W: Float64Array[] } {
    const n = X.length, p = X[0]?.length ?? 0;
    const cov: Float64Array[] = Array.from({ length: p }, () => new Float64Array(p));
    for (const row of X) {
      for (let j = 0; j < p; j++) {
        for (let k = 0; k < p; k++) (cov[j] as Float64Array)[k]! += (row[j] ?? 0) * (row[k] ?? 0) / n;
      }
    }
    // Diagonal whitening
    const std = Float64Array.from({ length: p }, (_, j) => Math.sqrt(Math.max((cov[j] as Float64Array)[j] ?? 1e-12, 1e-12)));
    const W = Array.from({ length: p }, (_, j) => {
      const row = new Float64Array(p);
      row[j] = 1 / (std[j] ?? 1);
      return row;
    });
    const Xw = X.map((row) => row.map((v, j) => v / (std[j] ?? 1)));
    this._whiteningMatrix = W;
    return { Xw, W };
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this.nFeaturesIn_ = p;
    const k = Math.min(this.nComponents, p);

    const { Xc } = this._center(X);
    const { Xw } = this._whiten(Xc);
    const Xt = Array.from({ length: p }, (_, j) => Float64Array.from({ length: n }, (_, i) => (Xw[i] as Float64Array)[j] ?? 0));

    // Initialize W randomly
    let W: Float64Array[] = Array.from({ length: k }, () => {
      const row = Float64Array.from({ length: p }, () => Math.random() - 0.5);
      const norm = Math.sqrt(row.reduce((s, v) => s + v * v, 0));
      return row.map((v) => v / Math.max(norm, 1e-12));
    });

    for (let iter = 0; iter < this.maxIter; iter++) {
      const Wprev = W.map((row) => new Float64Array(row));

      for (let c = 0; c < k; c++) {
        // Compute w^T * X
        const u = Float64Array.from({ length: n }, (_, i) => (W[c] as Float64Array).reduce((s, wj, j) => s + wj * ((Xw[i] as Float64Array)[j] ?? 0), 0));
        const { g, gPrime } = this._g(u);
        const E_gPrime = gPrime.reduce((s, v) => s + v, 0) / n;

        // Update w
        const newW = new Float64Array(p);
        for (let j = 0; j < p; j++) {
          let sum1 = 0;
          for (let i = 0; i < n; i++) sum1 += (g[i] ?? 0) * ((Xt[j] as Float64Array)[i] ?? 0);
          newW[j] = sum1 / n - E_gPrime * ((W[c] as Float64Array)[j] ?? 0);
        }

        // Decorrelate (symmetric deflation)
        for (let c2 = 0; c2 < c; c2++) {
          const dot = (W[c2] as Float64Array).reduce((s, v, j) => s + v * (newW[j] ?? 0), 0);
          for (let j = 0; j < p; j++) newW[j]! -= dot * ((W[c2] as Float64Array)[j] ?? 0);
        }
        const norm = Math.sqrt(newW.reduce((s, v) => s + v * v, 0));
        W[c] = norm > 0 ? newW.map((v) => v / norm) : newW;
      }

      // Check convergence
      const change = W.reduce((s, w, c) => {
        const dot = w.reduce((ss, wj, j) => ss + wj * ((Wprev[c] as Float64Array)[j] ?? 0), 0);
        return Math.max(s, Math.abs(1 - Math.abs(dot)));
      }, 0);
      this.nIter_ = iter + 1;
      if (change < this.tol) break;
    }

    this.components_ = W;
    this.mixingMatrix_ = W.map((row) => new Float64Array(row));
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_) throw new Error("Not fitted");
    const k = this.components_.length;
    return X.map((x) => Float64Array.from({ length: k }, (_, c) => (this.components_![c] as Float64Array).reduce((s, w, j) => s + w * (x[j] ?? 0), 0)));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class TruncatedSVDDecomp {
  nComponents: number;
  nIter: number;
  randomState: number;
  components_: Float64Array[] | null = null;
  singularValues_: Float64Array | null = null;
  explainedVariance_: Float64Array | null = null;
  explainedVarianceRatio_: Float64Array | null = null;
  nFeaturesIn_: number = 0;

  constructor(nComponents = 2, nIter = 5, randomState = 42) {
    this.nComponents = nComponents;
    this.nIter = nIter;
    this.randomState = randomState;
  }

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 0;
    this.nFeaturesIn_ = p;
    const k = Math.min(this.nComponents, Math.min(n, p));

    const components: Float64Array[] = [];
    const singularValues: number[] = [];
    const Xcopy = X.map((row) => new Float64Array(row));

    for (let rank = 0; rank < k; rank++) {
      let v = Float64Array.from({ length: p }, () => Math.random() - 0.5);
      const normV = Math.sqrt(v.reduce((s, vi) => s + vi * vi, 0));
      v = v.map((vi) => vi / Math.max(normV, 1e-12));

      for (let iter = 0; iter < this.nIter * 2; iter++) {
        const u = Float64Array.from({ length: n }, (_, i) => (Xcopy[i] as Float64Array).reduce((s, xij, j) => s + xij * (v[j] ?? 0), 0));
        const sigma = Math.sqrt(u.reduce((s, ui) => s + ui * ui, 0));
        const uNorm = sigma > 0 ? u.map((ui) => ui / sigma) : u;
        let newV = new Float64Array(p);
        for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) newV[j]! += (uNorm[i] ?? 0) * ((Xcopy[i] as Float64Array)[j] ?? 0);
        const normNewV = Math.sqrt(newV.reduce((s, vi) => s + vi * vi, 0));
        newV = newV.map((vi) => vi / Math.max(normNewV, 1e-12));
        v = newV;
      }

      const u = Float64Array.from({ length: n }, (_, i) => (Xcopy[i] as Float64Array).reduce((s, xij, j) => s + xij * (v[j] ?? 0), 0));
      const sigma = Math.sqrt(u.reduce((s, ui) => s + ui * ui, 0));
      components.push(new Float64Array(v));
      singularValues.push(sigma);

      for (let i = 0; i < n; i++) {
        const ui = sigma > 0 ? (u[i] ?? 0) / sigma : 0;
        for (let j = 0; j < p; j++) (Xcopy[i] as Float64Array)[j]! -= sigma * ui * (v[j] ?? 0);
      }
    }

    this.components_ = components;
    this.singularValues_ = Float64Array.from(singularValues);
    const totalVar = X.reduce((s, row) => s + row.reduce((ss, v) => ss + v * v, 0), 0);
    this.explainedVariance_ = Float64Array.from(singularValues, (s) => s * s / (n - 1));
    this.explainedVarianceRatio_ = Float64Array.from(singularValues, (s) => s * s / Math.max(totalVar, 1e-12));
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_) throw new Error("Not fitted");
    const k = this.components_.length;
    return X.map((x) => Float64Array.from({ length: k }, (_, c) => (this.components_![c] as Float64Array).reduce((s, v, j) => s + v * (x[j] ?? 0), 0)));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

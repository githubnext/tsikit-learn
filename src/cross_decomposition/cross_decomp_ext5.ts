/**
 * Canonical Correlation Analysis extension — regularized and kernel CCA.
 */

function matMul(A: Float64Array[], B: Float64Array[]): Float64Array[] {
  const n = A.length, k = B.length, m = B[0]?.length ?? 0;
  return Array.from({ length: n }, (_, i) =>
    Float64Array.from({ length: m }, (_, j) =>
      Array.from({ length: k }, (__, l) => ((A[i] as Float64Array)[l] ?? 0) * ((B[l] as Float64Array)[j] ?? 0)).reduce((s, v) => s + v, 0)
    )
  );
}

function covMatrix(X: Float64Array[]): Float64Array[] {
  const n = X.length, p = X[0]?.length ?? 0;
  const mean = new Float64Array(p);
  for (const row of X) for (let j = 0; j < p; j++) mean[j] += (row[j] ?? 0) / n;
  const cov: Float64Array[] = Array.from({ length: p }, () => new Float64Array(p));
  for (const row of X) {
    for (let j = 0; j < p; j++) {
      const dj = (row[j] ?? 0) - (mean[j] ?? 0);
      for (let k = 0; k < p; k++) (cov[j] as Float64Array)[k] += dj * ((row[k] ?? 0) - (mean[k] ?? 0)) / (n - 1);
    }
  }
  return cov;
}

export class RegularizedCCA {
  nComponents: number;
  regX: number;
  regY: number;
  maxIter: number;
  tol: number;
  xWeights_: Float64Array[] | null = null;
  yWeights_: Float64Array[] | null = null;
  cancorr_: Float64Array | null = null;
  nFeaturesXIn_: number = 0;
  nFeaturesYIn_: number = 0;

  constructor(nComponents = 2, regX = 0.1, regY = 0.1, maxIter = 500, tol = 1e-6) {
    this.nComponents = nComponents;
    this.regX = regX;
    this.regY = regY;
    this.maxIter = maxIter;
    this.tol = tol;
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 0, q = Y[0]?.length ?? 0;
    this.nFeaturesXIn_ = p;
    this.nFeaturesYIn_ = q;

    const Sxx = covMatrix(X);
    const Syy = covMatrix(Y);

    // Cross-covariance Sxy
    const meanX = new Float64Array(p), meanY = new Float64Array(q);
    for (const row of X) for (let j = 0; j < p; j++) meanX[j] += (row[j] ?? 0) / n;
    for (const row of Y) for (let j = 0; j < q; j++) meanY[j] += (row[j] ?? 0) / n;
    const Sxy: Float64Array[] = Array.from({ length: p }, () => new Float64Array(q));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < p; j++) {
        const dxj = (X[i]?.[j] ?? 0) - (meanX[j] ?? 0);
        for (let k = 0; k < q; k++) {
          (Sxy[j] as Float64Array)[k] += dxj * ((Y[i]?.[k] ?? 0) - (meanY[k] ?? 0)) / (n - 1);
        }
      }
    }

    // Regularize
    for (let j = 0; j < p; j++) (Sxx[j] as Float64Array)[j] += this.regX;
    for (let k = 0; k < q; k++) (Syy[k] as Float64Array)[k] += this.regY;

    // Iterative deflation for CCA (power method approach)
    const xWeights: Float64Array[] = [], yWeights: Float64Array[] = [], cancorr: number[] = [];

    const SxyDefl = Sxy.map((row) => new Float64Array(row));

    for (let comp = 0; comp < Math.min(this.nComponents, Math.min(p, q)); comp++) {
      // Initialize
      let wx = Float64Array.from({ length: p }, (_, i) => i === comp % p ? 1 : 0);
      let wy = new Float64Array(q);

      for (let iter = 0; iter < this.maxIter; iter++) {
        // wy = Sxy^T * wx / ||Sxy^T * wx||Syy
        const newWy = new Float64Array(q);
        for (let k = 0; k < q; k++) for (let j = 0; j < p; j++) newWy[k] += (SxyDefl[j] as Float64Array)[k] * (wx[j] ?? 0);
        // Normalize under Syy: solve Syy * wy = newWy
        const syyNorm = Math.sqrt(newWy.reduce((s, v, k) => {
          let syyv = 0;
          for (let l = 0; l < q; l++) syyv += ((Syy[k] as Float64Array)[l] ?? 0) * (newWy[l] ?? 0);
          return s + v * syyv;
        }, 0));
        wy = syyNorm > 0 ? newWy.map((v) => v / syyNorm) : newWy;

        // wx = Sxy * wy / ||Sxy * wy||Sxx
        const newWx = new Float64Array(p);
        for (let j = 0; j < p; j++) for (let k = 0; k < q; k++) newWx[j] += ((SxyDefl[j] as Float64Array)[k] ?? 0) * (wy[k] ?? 0);
        const sxxNorm = Math.sqrt(newWx.reduce((s, v, j) => {
          let sxxv = 0;
          for (let l = 0; l < p; l++) sxxv += ((Sxx[j] as Float64Array)[l] ?? 0) * (newWx[l] ?? 0);
          return s + v * sxxv;
        }, 0));
        const newWxNorm = sxxNorm > 0 ? newWx.map((v) => v / sxxNorm) : newWx;

        const change = newWxNorm.reduce((s, v, i) => s + (v - (wx[i] ?? 0)) ** 2, 0);
        wx = newWxNorm;
        if (change < this.tol) break;
      }

      // Compute canonical correlation
      const rho = wx.reduce((s, wxi, j) => {
        return s + wxi * (Sxy[j] as Float64Array).reduce((ss, v, k) => ss + v * (wy[k] ?? 0), 0);
      }, 0);

      xWeights.push(wx);
      yWeights.push(wy);
      cancorr.push(rho);

      // Deflate
      for (let j = 0; j < p; j++) for (let k = 0; k < q; k++) {
        (SxyDefl[j] as Float64Array)[k] -= rho * (wx[j] ?? 0) * (wy[k] ?? 0);
      }
    }

    this.xWeights_ = xWeights;
    this.yWeights_ = yWeights;
    this.cancorr_ = Float64Array.from(cancorr);
    return this;
  }

  transformX(X: Float64Array[]): Float64Array[] {
    if (!this.xWeights_) throw new Error("Not fitted");
    const k = this.xWeights_.length;
    return X.map((x) => Float64Array.from({ length: k }, (_, c) => (this.xWeights_![c] as Float64Array).reduce((s, w, j) => s + w * (x[j] ?? 0), 0)));
  }

  transformY(Y: Float64Array[]): Float64Array[] {
    if (!this.yWeights_) throw new Error("Not fitted");
    const k = this.yWeights_.length;
    return Y.map((y) => Float64Array.from({ length: k }, (_, c) => (this.yWeights_![c] as Float64Array).reduce((s, w, j) => s + w * (y[j] ?? 0), 0)));
  }

  fitTransform(X: Float64Array[], Y: Float64Array[]): [Float64Array[], Float64Array[]] {
    this.fit(X, Y);
    return [this.transformX(X), this.transformY(Y)];
  }
}

export class KernelCCA {
  nComponents: number;
  kernel: "rbf" | "poly" | "linear";
  gamma: number;
  degree: number;
  regParam: number;
  xWeights_: Float64Array[] | null = null;
  yWeights_: Float64Array[] | null = null;
  private _XTrain: Float64Array[] | null = null;
  private _YTrain: Float64Array[] | null = null;

  constructor(nComponents = 2, kernel: "rbf" | "poly" | "linear" = "rbf", gamma = 1.0, degree = 3, regParam = 0.1) {
    this.nComponents = nComponents;
    this.kernel = kernel;
    this.gamma = gamma;
    this.degree = degree;
    this.regParam = regParam;
  }

  private _kernelFunc(a: Float64Array, b: Float64Array): number {
    if (this.kernel === "linear") return a.reduce((s, v, d) => s + v * (b[d] ?? 0), 0);
    if (this.kernel === "poly") return (this.gamma * a.reduce((s, v, d) => s + v * (b[d] ?? 0), 0) + 1) ** this.degree;
    const dist2 = a.reduce((s, v, d) => s + (v - (b[d] ?? 0)) ** 2, 0);
    return Math.exp(-this.gamma * dist2);
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const n = X.length;
    this._XTrain = X;
    this._YTrain = Y;

    const Kx = Array.from({ length: n }, (_, i) => Float64Array.from({ length: n }, (_, j) => this._kernelFunc(X[i] as Float64Array, X[j] as Float64Array)));
    const Ky = Array.from({ length: n }, (_, i) => Float64Array.from({ length: n }, (_, j) => this._kernelFunc(Y[i] as Float64Array, Y[j] as Float64Array)));

    // Regularize
    for (let i = 0; i < n; i++) { (Kx[i] as Float64Array)[i] += this.regParam; (Ky[i] as Float64Array)[i] += this.regParam; }

    // CCA in kernel space: approximate via simple SVD on Kx * Ky
    const KxKy = matMul(Kx, Ky);

    const xW: Float64Array[] = [], yW: Float64Array[] = [];
    let curr = KxKy.map((row) => new Float64Array(row));
    for (let c = 0; c < Math.min(this.nComponents, n); c++) {
      let v = Float64Array.from({ length: n }, (_, i) => i === c ? 1 : 0);
      for (let iter = 0; iter < 30; iter++) {
        let newV = new Float64Array(n);
        for (let i = 0; i < n; i++) newV = newV.map((_, k) => newV[k] + (curr[i]?.[k] ?? 0) * (v[i] ?? 0));
        const norm = Math.sqrt(newV.reduce((s, vi) => s + vi * vi, 0));
        v = norm > 0 ? newV.map((vi) => vi / norm) : newV;
      }
      xW.push(new Float64Array(v));
      yW.push(new Float64Array(v));
      // Deflate
      const sigma = curr.map((row) => row.reduce((s, vi, j) => s + vi * (v[j] ?? 0), 0));
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) (curr[i] as Float64Array)[j] -= (sigma[i] ?? 0) * (v[j] ?? 0);
    }
    this.xWeights_ = xW;
    this.yWeights_ = yW;
    return this;
  }
}

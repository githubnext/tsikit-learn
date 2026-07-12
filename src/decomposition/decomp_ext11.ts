/**
 * Decomposition extensions: Independent Subspace Analysis, Robust PCA, Sparse NMF
 */

export class RobustPCAExt {
  private L_: Float64Array[] = []; // low-rank component
  private S_: Float64Array[] = []; // sparse component
  private fitted_ = false;

  constructor(private lambda: number = 0.1, private maxIter: number = 100, private tol: number = 1e-7) {}

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 0;
    let L = X.map(row => row.slice());
    let S = Array.from({ length: n }, () => new Float64Array(p));
    const mu = (n * p) / (4 * this._l1Norm(X));
    const thresh = this.lambda / mu;

    for (let iter = 0; iter < this.maxIter; iter++) {
      const oldL = L.map(r => r.slice());
      // Update L: SVT of (X - S)
      const M = X.map((row, i) => new Float64Array(row.map((v, j) => v - (S[i]?.[j] ?? 0))));
      L = this._svt(M, 1 / mu) as Float64Array<ArrayBuffer>[];
      // Update S: soft threshold of (X - L)
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < p; j++) {
          const v = (X[i]?.[j] ?? 0) - (L[i]?.[j] ?? 0);
          S[i]![j] = Math.sign(v) * Math.max(Math.abs(v) - thresh, 0);
        }
      }
      // Check convergence
      let diff = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) diff += ((L[i]?.[j] ?? 0) - (oldL[i]?.[j] ?? 0)) ** 2;
      if (Math.sqrt(diff) < this.tol) break;
    }
    this.L_ = L; this.S_ = S;
    this.fitted_ = true;
    return this;
  }

  private _l1Norm(X: Float64Array[]): number {
    return X.reduce((s, row) => s + row.reduce((ss, v) => ss + Math.abs(v), 0), 0);
  }

  private _svt(M: Float64Array[], threshold: number): Float64Array[] {
    // Approximate SVT via power iteration
    const n = M.length, p = M[0]?.length ?? 0;
    const rank = Math.min(n, p, 10);
    let rng = 42;
    const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return (rng / 0xffffffff) * 2 - 1; };

    const result = M.map(row => new Float64Array(p));
    let Mk = M.map(row => row.slice());

    for (let s = 0; s < rank; s++) {
      let u = new Float64Array(n).map(() => rand());
      let v = new Float64Array(p);
      for (let iter = 0; iter < 10; iter++) {
        for (let j = 0; j < p; j++) { v[j] = 0; for (let i = 0; i < n; i++) v[j] = (v[j] ?? 0) + (Mk[i]?.[j] ?? 0) * (u[i] ?? 0); }
        const vn = Math.sqrt(v.reduce((ss, x) => ss + x * x, 0)) || 1;
        v = v.map(x => x / vn);
        for (let i = 0; i < n; i++) { u[i] = 0; for (let j = 0; j < p; j++) u[i] = (u[i] ?? 0) + (Mk[i]?.[j] ?? 0) * (v[j] ?? 0); }
        const un = Math.sqrt(u.reduce((ss, x) => ss + x * x, 0)) || 1;
        u = u.map(x => x / un);
      }
      let sigma = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) sigma += (u[i] ?? 0) * (Mk[i]?.[j] ?? 0) * (v[j] ?? 0);
      const sigmaT = Math.max(sigma - threshold, 0);
      if (sigmaT <= 0) break;
      for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) result[i]![j] = (result[i]?.[j] ?? 0) + sigmaT * (u[i] ?? 0) * (v[j] ?? 0);
      for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) Mk[i]![j] = (Mk[i]?.[j] ?? 0) - sigma * (u[i] ?? 0) * (v[j] ?? 0);
    }
    return result;
  }

  get lowRankComponent(): Float64Array[] { return this.L_; }
  get sparseComponent(): Float64Array[] { return this.S_; }
}

export class SparseNMFExt {
  private W_: Float64Array[] = []; // n x k
  private H_: Float64Array[] = []; // k x p
  private fitted_ = false;

  constructor(
    private nComponents: number = 10,
    private l1RatioW: number = 0.1,
    private l1RatioH: number = 0.1,
    private maxIter: number = 200,
    private tol: number = 1e-4,
    private randomState: number = 42
  ) {}

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 0;
    const k = this.nComponents;
    let rng = this.randomState;
    const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 0xffffffff; };

    // Initialize W, H with random non-negative values
    let W = Array.from({ length: n }, () => new Float64Array(k).map(() => rand()));
    let H = Array.from({ length: k }, () => new Float64Array(p).map(() => rand()));

    for (let iter = 0; iter < this.maxIter; iter++) {
      const oldW = W.map(r => r.slice());
      // Update H: H = H * (W^T X) / (W^T W H + lambda_H)
      const WtX = Array.from({ length: k }, (_, a) => new Float64Array(p).map((_, j) =>
        W.reduce((s, Wi, i) => s + (Wi[a] ?? 0) * (X[i]?.[j] ?? 0), 0)
      ));
      const WtW = Array.from({ length: k }, (_, a) => new Float64Array(k).map((_, b) =>
        W.reduce((s, Wi) => s + (Wi[a] ?? 0) * (Wi[b] ?? 0), 0)
      ));
      for (let a = 0; a < k; a++) {
        for (let j = 0; j < p; j++) {
          const num = WtX[a]?.[j] ?? 0;
          let denom = 0;
          for (let b = 0; b < k; b++) denom += (WtW[a]?.[b] ?? 0) * (H[b]?.[j] ?? 0);
          denom += this.l1RatioH + 1e-10;
          H[a]![j] = Math.max((H[a]?.[j] ?? 0) * num / denom, 0);
        }
      }
      // Update W: W = W * (X H^T) / (W H H^T + lambda_W)
      const XHt = Array.from({ length: n }, (_, i) => new Float64Array(k).map((_, a) =>
        (X[i]!).reduce((s, v, j) => s + v * (H[a]?.[j] ?? 0), 0)
      ));
      const HHt = Array.from({ length: k }, (_, a) => new Float64Array(k).map((_, b) =>
        (H[a]!).reduce((s, v, j) => s + v * (H[b]?.[j] ?? 0), 0)
      ));
      for (let i = 0; i < n; i++) {
        for (let a = 0; a < k; a++) {
          const num = XHt[i]?.[a] ?? 0;
          let denom = 0;
          for (let b = 0; b < k; b++) denom += (W[i]?.[b] ?? 0) * (HHt[b]?.[a] ?? 0);
          denom += this.l1RatioW + 1e-10;
          W[i]![a] = Math.max((W[i]?.[a] ?? 0) * num / denom, 0);
        }
      }
      // Check convergence
      let diff = 0;
      for (let i = 0; i < n; i++) for (let a = 0; a < k; a++) diff += ((W[i]?.[a] ?? 0) - (oldW[i]?.[a] ?? 0)) ** 2;
      if (Math.sqrt(diff) < this.tol) break;
    }
    this.W_ = W; this.H_ = H;
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    const n = X.length, k = this.nComponents, p = X[0]?.length ?? 0;
    return Array.from({ length: n }, (_, i) => {
      const w = new Float64Array(k);
      for (let iter = 0; iter < 50; iter++) {
        for (let a = 0; a < k; a++) {
          const num = (X[i]!).reduce((s, v, j) => s + v * (this.H_[a]?.[j] ?? 0), 0);
          let denom = 0;
          for (let b = 0; b < k; b++) denom += (w[b] ?? 0) * (this.H_[a]!).reduce((s, v, j) => s + v * (this.H_[b]?.[j] ?? 0), 0);
          denom += this.l1RatioW + 1e-10;
          w[a] = Math.max((w[a] ?? 0) * num / denom, 0);
        }
      }
      return w;
    });
  }

  get components(): Float64Array[] { return this.H_; }
  get W(): Float64Array[] { return this.W_; }
}

export class KernelPCASpectralExt {
  private alphas_: Float64Array[] = [];
  private lambdas_: Float64Array = new Float64Array(0);
  private XFit_: Float64Array[] = [];
  private fitted_ = false;

  constructor(
    private nComponents: number = 2,
    private kernel: 'rbf' | 'poly' | 'linear' = 'rbf',
    private gamma: number = 1.0,
    private degree: number = 3,
    private coef0: number = 1.0
  ) {}

  fit(X: Float64Array[]): this {
    const n = X.length;
    const K = this._computeKernel(X, X, n);
    this._centerKernel(K, n);
    const { vecs, vals } = this._eigenDecomp(K, n, this.nComponents);
    this.lambdas_ = new Float64Array(vals);
    this.alphas_ = vecs;
    this.XFit_ = X;
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    const n = X.length, nFit = this.XFit_.length;
    const K = this._computeKernel(X, this.XFit_, nFit);
    // Center test kernel
    const rowMeans = Array.from({ length: n }, (_, i) => K[i]!.reduce((s, v) => s + v, 0) / nFit);
    const colMeans = new Float64Array(nFit);
    for (let j = 0; j < nFit; j++) colMeans[j] = K.reduce((s, row) => s + (row[j] ?? 0), 0) / n;
    const grandMean = K.reduce((s, row) => s + row.reduce((ss, v) => ss + v, 0), 0) / (n * nFit);
    const Kc = K.map((row, i) => new Float64Array(row.map((v, j) => v - (rowMeans[i] ?? 0) - (colMeans[j] ?? 0) + grandMean)));

    return Array.from({ length: n }, (_, i) => {
      return new Float64Array(this.nComponents).map((_, c) => {
        const lambda = this.lambdas_[c] ?? 1e-10;
        let val = 0;
        for (let j = 0; j < nFit; j++) val += (Kc[i]?.[j] ?? 0) * (this.alphas_[c]?.[j] ?? 0);
        return val / Math.sqrt(Math.abs(lambda) + 1e-10);
      });
    });
  }

  private _computeKernel(X: Float64Array[], Y: Float64Array[], m: number): Float64Array[] {
    return X.map(xi => new Float64Array(Y.map(yj => {
      if (this.kernel === 'rbf') {
        const d2 = xi.reduce((s, v, j) => s + (v - (yj[j] ?? 0)) ** 2, 0);
        return Math.exp(-this.gamma * d2);
      } else if (this.kernel === 'poly') {
        const dot = xi.reduce((s, v, j) => s + v * (yj[j] ?? 0), 0);
        return (this.gamma * dot + this.coef0) ** this.degree;
      } else {
        return xi.reduce((s, v, j) => s + v * (yj[j] ?? 0), 0);
      }
    })));
  }

  private _centerKernel(K: Float64Array[], n: number): void {
    const rowMeans = K.map(row => row.reduce((s, v) => s + v, 0) / n);
    const colMeans = new Float64Array(n);
    for (let j = 0; j < n; j++) colMeans[j] = K.reduce((s, row) => s + (row[j] ?? 0), 0) / n;
    const grandMean = rowMeans.reduce((s, v) => s + v, 0) / n;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i]![j] = (K[i]?.[j] ?? 0) - (rowMeans[i] ?? 0) - (colMeans[j] ?? 0) + grandMean;
  }

  private _eigenDecomp(K: Float64Array[], n: number, k: number): { vecs: Float64Array[]; vals: number[] } {
    let rng = 42;
    const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return (rng / 0xffffffff) * 2 - 1; };
    const vecs: Float64Array[] = [], vals: number[] = [];
    let Kk = K.map(r => r.slice());
    for (let s = 0; s < k; s++) {
      let v = new Float64Array(n).map(() => rand());
      let lambda = 0;
      for (let iter = 0; iter < 50; iter++) {
        let Av = new Float64Array(n);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Av[i] = (Av[i] ?? 0) + (Kk[i]?.[j] ?? 0) * (v[j] ?? 0);
        lambda = Math.sqrt(Av.reduce((s, x) => s + x * x, 0)) || 1;
        v = Av.map(x => x / lambda);
      }
      vecs.push(v); vals.push(lambda);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Kk[i]![j] = (Kk[i]?.[j] ?? 0) - lambda * (v[i] ?? 0) * (v[j] ?? 0);
    }
    return { vecs, vals };
  }
}

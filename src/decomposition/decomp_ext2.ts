/**
 * Extended decomposition: MiniBatchNMF, MiniBatchDictionaryLearning, LatentSemanticAnalysis
 */

export class MiniBatchNMF {
  private nComponents: number;
  private batchSize: number;
  private maxIter: number;
  private tol: number;
  W_: Float64Array[] | null = null;
  H_: Float64Array[] | null = null;
  nIter_: number = 0;

  constructor(nComponents = 10, batchSize = 200, maxIter = 200, tol = 1e-4) {
    this.nComponents = nComponents;
    this.batchSize = batchSize;
    this.maxIter = maxIter;
    this.tol = tol;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const k = this.nComponents;

    // Initialize H (dictionary)
    this.H_ = Array.from({ length: k }, () => {
      const row = new Float64Array(p);
      for (let j = 0; j < p; j++) row[j] = Math.random() + 0.1;
      return row;
    });

    // Mini-batch updates
    for (let iter = 0; iter < this.maxIter; iter++) {
      const batchStart = (iter * this.batchSize) % n;
      const batchEnd = Math.min(batchStart + this.batchSize, n);
      const batch = X.slice(batchStart, batchEnd);

      // Solve for W given H
      const W = this.solveW(batch, this.H_);

      // Update H
      const HtH: Float64Array[] = Array.from({ length: k }, () => new Float64Array(k));
      const WtX: Float64Array[] = Array.from({ length: k }, () => new Float64Array(p));
      for (let s = 0; s < batch.length; s++) {
        for (let i = 0; i < k; i++) {
          for (let j = 0; j < k; j++) HtH[i]![j]! += (W[s]![i] ?? 0) * (W[s]![j] ?? 0);
          for (let j = 0; j < p; j++) WtX[i]![j]! += (W[s]![i] ?? 0) * (batch[s]![j] ?? 0);
        }
      }
      for (let i = 0; i < k; i++) {
        for (let j = 0; j < p; j++) {
          const denom = batch.length > 0
            ? this.H_![i]!.reduce((acc, _, l) => acc + (HtH[l]![i] ?? 0) * (this.H_![l]![j] ?? 0), 0)
            : 1;
          this.H_![i]![j] = Math.max(1e-10, (this.H_![i]![j] ?? 0) * (WtX[i]![j] ?? 0) / (denom + 1e-10));
        }
      }
      this.nIter_ = iter + 1;
    }

    this.W_ = this.solveW(X, this.H_);
    return this;
  }

  private solveW(X: Float64Array[], H: Float64Array[]): Float64Array[] {
    const k = H.length;
    const p = H[0]?.length ?? 0;
    return X.map((row) => {
      const w = new Float64Array(k).fill(1);
      for (let iter = 0; iter < 20; iter++) {
        for (let i = 0; i < k; i++) {
          let num = 0, denom = 1e-10;
          for (let j = 0; j < p; j++) {
            let hx = 0;
            for (let l = 0; l < k; l++) hx += (w[l] ?? 0) * (H[l]![j] ?? 0);
            num += (H[i]![j] ?? 0) * (row[j] ?? 0);
            denom += (H[i]![j] ?? 0) * hx;
          }
          w[i] = Math.max(1e-10, (w[i] ?? 0) * num / denom);
        }
      }
      return w;
    });
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.H_) throw new Error("Not fitted");
    return this.solveW(X, this.H_);
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class MiniBatchDictionaryLearning {
  private nComponents: number;
  private batchSize: number;
  private maxIter: number;
  private alpha: number;
  components_: Float64Array[] | null = null;
  nIter_: number = 0;

  constructor(nComponents = 10, batchSize = 200, maxIter = 1000, alpha = 1.0) {
    this.nComponents = nComponents;
    this.batchSize = batchSize;
    this.maxIter = maxIter;
    this.alpha = alpha;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const k = this.nComponents;

    // Initialize dictionary with random rows from X
    this.components_ = Array.from({ length: k }, (_, i) => {
      const row = X[i % n] ?? new Float64Array(p);
      const norm = Math.sqrt(row.reduce((acc, v) => acc + v * v, 0)) || 1;
      return new Float64Array(row).map((v) => v / norm) as unknown as Float64Array;
    });

    const A: Float64Array[] = Array.from({ length: k }, () => new Float64Array(k));
    const B: Float64Array[] = Array.from({ length: k }, () => new Float64Array(p));

    for (let iter = 0; iter < this.maxIter; iter++) {
      const batchIdx = Math.floor(Math.random() * Math.max(1, n - this.batchSize));
      const batch = X.slice(batchIdx, batchIdx + this.batchSize);

      // Sparse coding (LASSO-like via soft thresholding)
      const codes = batch.map((x) => this.sparseCode(x, this.components_!));

      // Update A and B
      for (let s = 0; s < codes.length; s++) {
        for (let i = 0; i < k; i++) {
          for (let j = 0; j < k; j++) A[i]![j] = (A[i]![j] ?? 0) * 0.99 + (codes[s]![i] ?? 0) * (codes[s]![j] ?? 0);
          for (let j = 0; j < p; j++) B[i]![j] = (B[i]![j] ?? 0) * 0.99 + (codes[s]![i] ?? 0) * (batch[s]![j] ?? 0);
        }
      }

      // Update dictionary
      for (let i = 0; i < k; i++) {
        const aii = A[i]![i] ?? 1;
        for (let j = 0; j < p; j++) {
          let u = (B[i]![j] ?? 0);
          for (let l = 0; l < k; l++) if (l !== i) u -= (A[i]![l] ?? 0) * (this.components_![l]![j] ?? 0);
          this.components_![i]![j] = aii > 1e-10 ? u / aii : 0;
        }
        // Normalize
        const norm = Math.sqrt(this.components_![i]!.reduce((acc, v) => acc + v * v, 0)) || 1;
        for (let j = 0; j < p; j++) this.components_![i]![j] = (this.components_![i]![j] ?? 0) / norm;
      }
      this.nIter_ = iter + 1;
    }
    return this;
  }

  private sparseCode(x: Float64Array, D: Float64Array[]): Float64Array {
    const k = D.length;
    const codes = new Float64Array(k);
    for (let i = 0; i < k; i++) {
      let dot = 0;
      for (let j = 0; j < x.length; j++) dot += (x[j] ?? 0) * (D[i]![j] ?? 0);
      codes[i] = Math.sign(dot) * Math.max(0, Math.abs(dot) - this.alpha);
    }
    return codes;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_) throw new Error("Not fitted");
    return X.map((x) => this.sparseCode(x, this.components_!));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class LatentSemanticAnalysis {
  private nComponents: number;
  private nIter: number;
  components_: Float64Array[] | null = null;
  singularValues_: Float64Array | null = null;
  explainedVariance_: Float64Array | null = null;

  constructor(nComponents = 2, nIter = 5) {
    this.nComponents = nComponents;
    this.nIter = nIter;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const k = Math.min(this.nComponents, n, p);

    // Randomized SVD
    const { V, sigma } = this.randomizedSVD(X, k);
    this.components_ = V;
    this.singularValues_ = sigma;
    const totalVar = sigma.reduce((acc, v) => acc + v * v, 0);
    this.explainedVariance_ = new Float64Array(k);
    for (let i = 0; i < k; i++) this.explainedVariance_[i] = (sigma[i] ?? 0) ** 2 / (totalVar || 1);
    return this;
  }

  private randomizedSVD(X: Float64Array[], k: number): { V: Float64Array[]; sigma: Float64Array } {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    // Power iteration
    const Omega: Float64Array[] = Array.from({ length: p }, () => {
      const row = new Float64Array(k);
      for (let j = 0; j < k; j++) { const u1 = Math.random(), u2 = Math.random(); row[j] = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2); }
      return row;
    });
    let Y = this.matMul(X, Omega, n, p, k);
    for (let iter = 0; iter < this.nIter; iter++) {
      Y = this.matMul(X, this.matMulT(X, Y, n, p, k), n, p, k);
    }
    // QR
    const Q = this.gramSchmidt(Y, n, k);
    // B = Q^T X
    const B = this.matMulT(Q, X.map((row, i) => { const r = new Float64Array(n); r[i] = 1; return r; }), n, n, n);
    const sigma = new Float64Array(k);
    const V: Float64Array[] = Array.from({ length: k }, () => new Float64Array(p));
    for (let i = 0; i < k; i++) {
      let norm = 0;
      for (let j = 0; j < p; j++) { const v = X.reduce((acc, row, s) => acc + (Q[s]![i] ?? 0) * (row[j] ?? 0), 0); V[i]![j] = v; norm += v * v; }
      sigma[i] = Math.sqrt(norm);
      if (sigma[i]! > 1e-10) for (let j = 0; j < p; j++) V[i]![j] = (V[i]![j] ?? 0) / (sigma[i] ?? 1);
    }
    return { V, sigma };
  }

  private matMul(A: Float64Array[], B: Float64Array[], n: number, p: number, k: number): Float64Array[] {
    return Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(k);
      for (let j = 0; j < k; j++) for (let l = 0; l < p; l++) row[j]! += (A[i]![l] ?? 0) * (B[l]![j] ?? 0);
      return row;
    });
  }

  private matMulT(A: Float64Array[], B: Float64Array[], n: number, p: number, k: number): Float64Array[] {
    const result: Float64Array[] = Array.from({ length: p }, () => new Float64Array(k));
    for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) for (let l = 0; l < k; l++) result[j]![l]! += (A[i]![l] ?? 0) * (B[i]![j] ?? 0);
    return result;
  }

  private gramSchmidt(X: Float64Array[], n: number, k: number): Float64Array[] {
    const Q: Float64Array[] = [];
    for (let j = 0; j < k; j++) {
      let v = new Float64Array(n);
      for (let i = 0; i < n; i++) v[i] = X[i]![j] ?? 0;
      for (const q of Q) { const dot = q.reduce((acc, qi, i) => acc + qi * (v[i] ?? 0), 0); for (let i = 0; i < n; i++) v[i] = (v[i] ?? 0) - dot * (q[i] ?? 0); }
      const norm = Math.sqrt(v.reduce((acc, vi) => acc + vi * vi, 0)) || 1;
      Q.push(v.map((vi) => vi / norm) as unknown as Float64Array);
    }
    return Array.from({ length: n }, (_, i) => { const row = new Float64Array(k); for (let j = 0; j < k; j++) row[j] = Q[j]![i] ?? 0; return row; });
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_) throw new Error("Not fitted");
    return X.map((row) => {
      const out = new Float64Array(this.nComponents);
      for (let i = 0; i < this.nComponents; i++) {
        let s = 0;
        for (let j = 0; j < row.length; j++) s += (row[j] ?? 0) * (this.components_![i]![j] ?? 0);
        out[i] = s;
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

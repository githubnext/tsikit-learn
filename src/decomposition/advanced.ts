/**
 * Additional decomposition methods: IncrementalPCA, KernelPCA, FactorAnalysis.
 * Mirrors sklearn.decomposition.
 */

import { NotFittedError } from "../exceptions.js";

/** Compute column means. */
function colMeans(X: Float64Array[]): Float64Array {
  const p = (X[0] ?? new Float64Array(0)).length;
  const m = new Float64Array(p);
  for (const xi of X) for (let j = 0; j < p; j++) m[j] = (m[j] ?? 0) + (xi[j] ?? 0);
  for (let j = 0; j < p; j++) m[j] = (m[j] ?? 0) / X.length;
  return m;
}

/** Matrix multiply A (m x k) * B (k x n) */
function matMul(A: Float64Array[], B: Float64Array[]): Float64Array[] {
  const m = A.length;
  const k = (A[0] ?? new Float64Array(0)).length;
  const n = (B[0] ?? new Float64Array(0)).length;
  const C = Array.from({ length: m }, () => new Float64Array(n));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let l = 0; l < k; l++) s += (A[i]![l] ?? 0) * (B[l]![j] ?? 0);
      C[i]![j] = s;
    }
  }
  return C;
}

/** Compute X^T X. */
function gramMatrix(X: Float64Array[]): Float64Array[] {
  const p = (X[0] ?? new Float64Array(0)).length;
  const n = X.length;
  const G = Array.from({ length: p }, () => new Float64Array(p));
  for (let i = 0; i < n; i++) {
    const xi = X[i] ?? new Float64Array(p);
    for (let a = 0; a < p; a++) {
      for (let b = a; b < p; b++) {
        const val = (xi[a] ?? 0) * (xi[b] ?? 0);
        G[a]![b] = (G[a]![b] ?? 0) + val;
        if (a !== b) G[b]![a] = (G[b]![a] ?? 0) + val;
      }
    }
  }
  return G;
}

/** Power iteration for top-k eigenvectors of a symmetric matrix. */
function eigenDecomp(
  M: Float64Array[],
  k: number,
  nIter = 100,
): { vectors: Float64Array[]; values: Float64Array } {
  const p = M.length;
  const vectors: Float64Array[] = [];
  const values = new Float64Array(k);
  // Deflation approach
  const Mwork = M.map((row) => row.slice());

  for (let comp = 0; comp < k; comp++) {
    // Random init
    let v = new Float64Array(p);
    for (let j = 0; j < p; j++) v[j] = (j === comp ? 1 : 0.01 * Math.sin(j + comp));
    let eigenval = 0;
    for (let iter = 0; iter < nIter; iter++) {
      const Mv = new Float64Array(p);
      for (let i = 0; i < p; i++) {
        for (let j = 0; j < p; j++) Mv[i] = (Mv[i] ?? 0) + (Mwork[i]![j] ?? 0) * (v[j] ?? 0);
      }
      eigenval = 0;
      for (let j = 0; j < p; j++) eigenval += (v[j] ?? 0) * (Mv[j] ?? 0);
      let norm = 0;
      for (let j = 0; j < p; j++) norm += (Mv[j] ?? 0) ** 2;
      norm = Math.sqrt(norm);
      if (norm < 1e-15) break;
      const vNew = Float64Array.from(Mv, (x) => x / norm);
      const diff = Math.sqrt(vNew.reduce((s, x, i) => s + (x - (v[i] ?? 0)) ** 2, 0));
      v = vNew;
      if (diff < 1e-10) break;
    }
    vectors[comp] = v;
    values[comp] = Math.max(0, eigenval);
    // Deflate
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        Mwork[i]![j] = (Mwork[i]![j] ?? 0) - eigenval * (v[i] ?? 0) * (v[j] ?? 0);
      }
    }
  }
  return { vectors, values };
}

/**
 * Incremental principal component analysis (IPCA).
 * Processes data in batches, enabling large-scale PCA.
 * Mirrors sklearn.decomposition.IncrementalPCA.
 */
export class IncrementalPCA {
  nComponents: number | null;
  batchSize: number | null;
  whiten: boolean;

  components_: Float64Array[] | null = null;
  explainedVariance_: Float64Array | null = null;
  explainedVarianceRatio_: Float64Array | null = null;
  mean_: Float64Array | null = null;
  nSamplesSeen_: number = 0;

  constructor(
    options: {
      nComponents?: number | null;
      batchSize?: number | null;
      whiten?: boolean;
    } = {},
  ) {
    this.nComponents = options.nComponents ?? null;
    this.batchSize = options.batchSize ?? null;
    this.whiten = options.whiten ?? false;
  }

  partialFit(X: Float64Array[]): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const k = Math.min(this.nComponents ?? p, p, n);

    // Incremental mean update
    if (this.mean_ === null) {
      this.mean_ = colMeans(X);
      this.nSamplesSeen_ = n;
    } else {
      const prevN = this.nSamplesSeen_;
      const batchMean = colMeans(X);
      const totalN = prevN + n;
      const newMean = new Float64Array(p);
      for (let j = 0; j < p; j++) {
        newMean[j] = ((this.mean_[j] ?? 0) * prevN + (batchMean[j] ?? 0) * n) / totalN;
      }
      this.mean_ = newMean;
      this.nSamplesSeen_ = totalN;
    }

    // Center data
    const Xc = X.map((xi) => {
      const out = new Float64Array(p);
      for (let j = 0; j < p; j++) out[j] = (xi[j] ?? 0) - (this.mean_![j] ?? 0);
      return out;
    });

    // Compute covariance contribution and update components via SVD
    const G = gramMatrix(Xc);

    if (this.components_ !== null) {
      // Merge with existing: approximate by re-computing on augmented covariance
      const prevComp = this.components_!;
      const prevVar = this.explainedVariance_!;
      // Add previous covariance contribution
      for (let a = 0; a < k; a++) {
        const va = prevComp[a] ?? new Float64Array(p);
        const lambda = prevVar[a] ?? 0;
        for (let i = 0; i < p; i++) {
          for (let j = 0; j < p; j++) {
            G[i]![j] = (G[i]![j] ?? 0) + lambda * (va[i] ?? 0) * (va[j] ?? 0);
          }
        }
      }
    }

    const { vectors, values } = eigenDecomp(G, k);
    this.components_ = vectors;
    const totalVar = values.reduce((s, v) => s + v, 0);
    this.explainedVariance_ = values;
    this.explainedVarianceRatio_ = Float64Array.from(
      values,
      (v) => v / (totalVar || 1),
    );
    return this;
  }

  fit(X: Float64Array[]): this {
    const batchSize = this.batchSize ?? Math.max(50, X.length);
    this.mean_ = null;
    this.components_ = null;
    this.nSamplesSeen_ = 0;
    for (let i = 0; i < X.length; i += batchSize) {
      this.partialFit(X.slice(i, i + batchSize));
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.components_ === null || this.mean_ === null) throw new NotFittedError();
    const k = this.components_.length;
    const p = this.mean_.length;
    return X.map((xi) => {
      const xc = new Float64Array(p);
      for (let j = 0; j < p; j++) xc[j] = (xi[j] ?? 0) - (this.mean_![j] ?? 0);
      const out = new Float64Array(k);
      for (let i = 0; i < k; i++) {
        const comp = this.components_![i] ?? new Float64Array(p);
        let s = 0;
        for (let j = 0; j < p; j++) s += (xc[j] ?? 0) * (comp[j] ?? 0);
        if (this.whiten) {
          const std = Math.sqrt(this.explainedVariance_![i] ?? 1) || 1;
          out[i] = s / std;
        } else {
          out[i] = s;
        }
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

/**
 * Kernel PCA — kernelized non-linear PCA.
 * Mirrors sklearn.decomposition.KernelPCA.
 */
export class KernelPCA {
  nComponents: number | null;
  kernel: "rbf" | "poly" | "sigmoid" | "cosine" | "linear";
  gamma: number | null;
  degree: number;
  coef0: number;

  alphas_: Float64Array[] | null = null;
  lambdas_: Float64Array | null = null;
  xFit_: Float64Array[] | null = null;
  kFitRows_: Float64Array[] | null = null;

  constructor(
    options: {
      nComponents?: number | null;
      kernel?: "rbf" | "poly" | "sigmoid" | "cosine" | "linear";
      gamma?: number | null;
      degree?: number;
      coef0?: number;
    } = {},
  ) {
    this.nComponents = options.nComponents ?? null;
    this.kernel = options.kernel ?? "rbf";
    this.gamma = options.gamma ?? null;
    this.degree = options.degree ?? 3;
    this.coef0 = options.coef0 ?? 1;
  }

  private _kernelFunc(a: Float64Array, b: Float64Array): number {
    const p = a.length;
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let j = 0; j < p; j++) {
      dot += (a[j] ?? 0) * (b[j] ?? 0);
      normA += (a[j] ?? 0) ** 2;
      normB += (b[j] ?? 0) ** 2;
    }
    const gamma = this.gamma ?? (1 / p || 1);
    switch (this.kernel) {
      case "rbf": {
        let dist = 0;
        for (let j = 0; j < p; j++) dist += ((a[j] ?? 0) - (b[j] ?? 0)) ** 2;
        return Math.exp(-gamma * dist);
      }
      case "poly": return (gamma * dot + this.coef0) ** this.degree;
      case "sigmoid": return Math.tanh(gamma * dot + this.coef0);
      case "cosine": {
        const denom = Math.sqrt(normA) * Math.sqrt(normB);
        return denom > 1e-15 ? dot / denom : 0;
      }
      default: return dot;
    }
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const k = Math.min(this.nComponents ?? n, n);
    this.xFit_ = X;
    // Compute kernel matrix
    const K = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const val = this._kernelFunc(X[i] ?? new Float64Array(0), X[j] ?? new Float64Array(0));
        K[i]![j] = val;
        K[j]![i] = val;
      }
    }
    // Center kernel matrix
    const rowMeans = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) rowMeans[i] = (rowMeans[i] ?? 0) + (K[i]![j] ?? 0);
      rowMeans[i] = (rowMeans[i] ?? 0) / n;
    }
    let grandMean = 0;
    for (let i = 0; i < n; i++) grandMean += rowMeans[i] ?? 0;
    grandMean /= n;
    const Kc = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        Kc[i]![j] = (K[i]![j] ?? 0) - (rowMeans[i] ?? 0) - (rowMeans[j] ?? 0) + grandMean;
      }
    }
    this.kFitRows_ = Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(n);
      for (let j = 0; j < n; j++) row[j] = Kc[i]![j] ?? 0;
      return row;
    });

    // Eigen decomposition of Kc
    const { vectors, values } = eigenDecomp(Kc, k);
    this.lambdas_ = values;
    // alpha_i = eigvec_i / sqrt(eigenval_i)
    this.alphas_ = vectors.map((v, i) => {
      const lam = values[i] ?? 1e-15;
      const scale = Math.sqrt(Math.abs(lam) || 1e-15);
      return Float64Array.from(v, (x) => x / scale);
    });
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.alphas_ === null || this.xFit_ === null || this.kFitRows_ === null) {
      throw new NotFittedError();
    }
    const nTrain = this.xFit_.length;
    const k = this.alphas_.length;
    return X.map((xi) => {
      const kv = new Float64Array(nTrain);
      for (let j = 0; j < nTrain; j++) {
        kv[j] = this._kernelFunc(xi, this.xFit_![j] ?? new Float64Array(0));
      }
      const out = new Float64Array(k);
      for (let i = 0; i < k; i++) {
        const alpha = this.alphas_![i] ?? new Float64Array(nTrain);
        let s = 0;
        for (let j = 0; j < nTrain; j++) s += (kv[j] ?? 0) * (alpha[j] ?? 0);
        out[i] = s;
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

/**
 * Factor Analysis via EM algorithm.
 * Mirrors sklearn.decomposition.FactorAnalysis.
 */
export class FactorAnalysis {
  nComponents: number;
  maxIter: number;
  tol: number;
  svdMethod: "randomized" | "lapack";

  components_: Float64Array[] | null = null;
  noiseVariance_: Float64Array | null = null;
  mean_: Float64Array | null = null;
  nIter_: number = 0;

  constructor(
    options: {
      nComponents?: number;
      maxIter?: number;
      tol?: number;
      svdMethod?: "randomized" | "lapack";
    } = {},
  ) {
    this.nComponents = options.nComponents ?? 1;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-2;
    this.svdMethod = options.svdMethod ?? "randomized";
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const k = Math.min(this.nComponents, p);

    this.mean_ = colMeans(X);
    const Xc = X.map((xi) => {
      const out = new Float64Array(p);
      for (let j = 0; j < p; j++) out[j] = (xi[j] ?? 0) - (this.mean_![j] ?? 0);
      return out;
    });

    // Initialize W (p x k) and psi (noise variances, p)
    const W = Array.from({ length: p }, (_, i) =>
      Float64Array.from({ length: k }, (_, j) => (i === j ? 1 : 0.1 * Math.sin(i + j))),
    );
    const psi = new Float64Array(p).fill(1);

    // EM algorithm
    for (let iter = 0; iter < this.maxIter; iter++) {
      // E-step: compute posterior mean of factors
      // M = W^T Psi^-1 W + I (k x k)
      const M = Array.from({ length: k }, () => new Float64Array(k));
      for (let a = 0; a < k; a++) {
        M[a]![a] = 1;
        for (let b = 0; b < k; b++) {
          for (let j = 0; j < p; j++) {
            M[a]![b] = (M[a]![b] ?? 0) + (W[j]![a] ?? 0) * (W[j]![b] ?? 0) / ((psi[j] ?? 1) || 1);
          }
        }
      }

      // Invert M (k x k) via simple Gauss-Jordan
      const Minv = this._invertKK(M, k);

      // Compute E[z|x] = Minv W^T Psi^-1 x
      // WtPsiInv = W^T Psi^-1 (k x p)
      const WtPsiInv = Array.from({ length: k }, (_, a) =>
        Float64Array.from({ length: p }, (_, j) => (W[j]![a] ?? 0) / ((psi[j] ?? 1) || 1)),
      );

      // Ez (n x k): Ez[i] = Minv WtPsiInv Xc[i]
      const Ez = Array.from({ length: n }, (_, i) => {
        const xi = Xc[i] ?? new Float64Array(p);
        const out = new Float64Array(k);
        for (let a = 0; a < k; a++) {
          let s = 0;
          for (let j = 0; j < p; j++) s += (WtPsiInv[a]![j] ?? 0) * (xi[j] ?? 0);
          for (let b = 0; b < k; b++) out[a] = (out[a] ?? 0) + (Minv[a]![b] ?? 0) * s;
        }
        return out;
      });

      // E[zz^T] = Minv + Ez Ez^T (per sample, but summed)
      const Ezz = Array.from({ length: k }, () => new Float64Array(k));
      for (let a = 0; a < k; a++) {
        for (let b = 0; b < k; b++) {
          Ezz[a]![b] = n * (Minv[a]![b] ?? 0);
          for (let i = 0; i < n; i++) {
            Ezz[a]![b] = (Ezz[a]![b] ?? 0) + (Ez[i]![a] ?? 0) * (Ez[i]![b] ?? 0);
          }
        }
      }

      // M-step: update W
      // W_new (p x k) = (sum_i x_i E[z|x_i]^T) Ezz^-1
      const XEz = Array.from({ length: p }, () => new Float64Array(k));
      for (let i = 0; i < n; i++) {
        const xi = Xc[i] ?? new Float64Array(p);
        for (let j = 0; j < p; j++) {
          for (let a = 0; a < k; a++) {
            XEz[j]![a] = (XEz[j]![a] ?? 0) + (xi[j] ?? 0) * (Ez[i]![a] ?? 0);
          }
        }
      }
      const EzzInv = this._invertKK(Ezz, k);
      const WnewArr = matMul(XEz, EzzInv);

      // Update psi
      const psiNew = new Float64Array(p);
      for (let j = 0; j < p; j++) {
        let s = 0;
        for (let i = 0; i < n; i++) {
          const xi = Xc[i] ?? new Float64Array(p);
          s += (xi[j] ?? 0) ** 2;
          for (let a = 0; a < k; a++) {
            s -= (WnewArr[j]![a] ?? 0) * (Ez[i]![a] ?? 0) * (xi[j] ?? 0);
          }
        }
        psiNew[j] = Math.max(1e-6, s / n);
      }

      // Check convergence
      let maxDiff = 0;
      for (let j = 0; j < p; j++) {
        for (let a = 0; a < k; a++) {
          maxDiff = Math.max(maxDiff, Math.abs((WnewArr[j]![a] ?? 0) - (W[j]![a] ?? 0)));
        }
      }

      for (let j = 0; j < p; j++) {
        for (let a = 0; a < k; a++) W[j]![a] = WnewArr[j]![a] ?? 0;
        psi[j] = psiNew[j] ?? 1e-6;
      }

      this.nIter_ = iter + 1;
      if (maxDiff < this.tol) break;
    }

    // components_ = W^T (k x p)
    this.components_ = Array.from({ length: k }, (_, a) =>
      Float64Array.from({ length: p }, (_, j) => W[j]![a] ?? 0),
    );
    this.noiseVariance_ = psi;
    return this;
  }

  private _invertKK(M: Float64Array[], k: number): Float64Array[] {
    const aug = Array.from({ length: k }, (_, i) => {
      const row = new Float64Array(2 * k);
      for (let j = 0; j < k; j++) row[j] = M[i]![j] ?? 0;
      row[k + i] = 1;
      return row;
    });
    for (let col = 0; col < k; col++) {
      let maxRow = col;
      for (let row = col + 1; row < k; row++) {
        if (Math.abs(aug[row]![col] ?? 0) > Math.abs(aug[maxRow]![col] ?? 0)) maxRow = row;
      }
      const tmpAdv = aug[col]!; aug[col] = aug[maxRow]!; aug[maxRow] = tmpAdv;
      const pivot = aug[col]![col] ?? 1e-12;
      if (Math.abs(pivot) < 1e-15) continue;
      for (let j = 0; j < 2 * k; j++) aug[col]![j] = (aug[col]![j] ?? 0) / pivot;
      for (let row = 0; row < k; row++) {
        if (row === col) continue;
        const factor = aug[row]![col] ?? 0;
        for (let j = 0; j < 2 * k; j++) {
          aug[row]![j] = (aug[row]![j] ?? 0) - factor * (aug[col]![j] ?? 0);
        }
      }
    }
    return aug.map((row) => Float64Array.from({ length: k }, (_, j) => row[k + j] ?? 0));
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.components_ === null || this.mean_ === null) throw new NotFittedError();
    const k = this.components_.length;
    const p = this.mean_.length;
    return X.map((xi) => {
      const xc = new Float64Array(p);
      for (let j = 0; j < p; j++) xc[j] = (xi[j] ?? 0) - (this.mean_![j] ?? 0);
      const out = new Float64Array(k);
      for (let i = 0; i < k; i++) {
        const comp = this.components_![i] ?? new Float64Array(p);
        let s = 0;
        for (let j = 0; j < p; j++) s += (xc[j] ?? 0) * (comp[j] ?? 0);
        out[i] = s;
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

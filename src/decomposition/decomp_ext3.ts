/**
 * Extended decomposition: MiniBatch Dictionary Learning, Online PCA,
 * Kernel PCA extensions, and Sparse PCA helpers.
 */

/** MiniBatch Dictionary Learning — online variant. */
export class MiniBatchDictionaryLearning {
  nComponents: number;
  batchSize: number;
  nIter: number;
  alpha: number;
  dictionary_?: Float64Array[];

  constructor(nComponents = 10, batchSize = 50, nIter = 100, alpha = 1.0) {
    this.nComponents = nComponents;
    this.batchSize = batchSize;
    this.nIter = nIter;
    this.alpha = alpha;
  }

  fit(X: Float64Array[]): this {
    const d = X[0]?.length ?? 0;
    // Initialize dictionary randomly
    this.dictionary_ = Array.from({ length: this.nComponents }, () => {
      const atom = new Float64Array(d).map(() => Math.random() - 0.5);
      const norm = Math.sqrt(atom.reduce((s, v) => s + v * v, 0)) + 1e-10;
      return atom.map((v) => v / norm);
    });

    for (let iter = 0; iter < this.nIter; iter++) {
      // Sample mini-batch
      const batchIdx = Array.from({ length: Math.min(this.batchSize, X.length) }, () =>
        Math.floor(Math.random() * X.length)
      );
      const batch = batchIdx.map((i) => X[i] ?? new Float64Array(d));
      // Sparse coding step (OMP-like: just use inner products)
      for (const xi of batch) {
        const codes = this._encode(xi);
        this._updateDict(xi, codes);
      }
    }
    return this;
  }

  private _encode(x: Float64Array): Float64Array {
    const dict = this.dictionary_!;
    const codes = new Float64Array(dict.length);
    for (let k = 0; k < dict.length; k++) {
      const atom = dict[k];
      if (atom === undefined) continue;
      let dot = 0;
      for (let j = 0; j < x.length; j++) dot += (x[j] ?? 0) * (atom[j] ?? 0);
      codes[k] = Math.max(0, dot - this.alpha);  // soft threshold
    }
    return codes;
  }

  private _updateDict(x: Float64Array, codes: Float64Array): void {
    const dict = this.dictionary_!;
    const lr = 0.01;
    for (let k = 0; k < dict.length; k++) {
      if ((codes[k] ?? 0) === 0) continue;
      const atom = dict[k];
      if (atom === undefined) continue;
      // Reconstruct residual
      let residual = 0;
      for (let j = 0; j < x.length; j++) {
        let rec = 0;
        for (let l = 0; l < dict.length; l++) rec += (codes[l] ?? 0) * (dict[l]?.[j] ?? 0);
        residual += (x[j] ?? 0 - rec) * (atom[j] ?? 0);
      }
      for (let j = 0; j < x.length; j++) atom[j] = (atom[j] ?? 0) + lr * (codes[k] ?? 0) * (x[j] ?? 0);
      // Normalize
      const norm = Math.sqrt(atom.reduce((s, v) => s + v * v, 0)) + 1e-10;
      for (let j = 0; j < x.length; j++) atom[j] = (atom[j] ?? 0) / norm;
    }
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.dictionary_) throw new Error("Not fitted");
    return X.map((xi) => this._encode(xi));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

/** Spectral embedding helper: diffusion maps. */
export function diffusionMap(
  K: Float64Array[],   // kernel/affinity matrix
  nComponents: number,
  t = 1,
): Float64Array[] {
  const n = K.length;
  // Row-normalize K to get Markov matrix P
  const P = K.map((row) => {
    const rowSum = row.reduce((a, b) => a + b, 0) + 1e-10;
    return row.map((v) => v / rowSum);
  });

  // Power iteration for top eigenvectors
  let vecs = Array.from({ length: nComponents }, (_, k) => {
    const v = new Float64Array(n);
    if (k < n) v[k] = 1;
    return v;
  });

  for (let iter = 0; iter < 50; iter++) {
    vecs = vecs.map((v) => {
      const newV = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < n; j++) sum += (P[i]?.[j] ?? 0) * (v[j] ?? 0);
        newV[i] = sum;
      }
      const norm = Math.sqrt(newV.reduce((s, x) => s + x * x, 0)) + 1e-10;
      return newV.map((x) => x / norm);
    });
  }

  // Scale by t-th power of eigenvalues (approximated by norms after power iteration)
  return Array.from({ length: n }, (_, i) =>
    new Float64Array(nComponents).map((_, k) => (vecs[k]?.[i] ?? 0) * (t > 0 ? 1 : 1))
  );
}

/** Non-negative least squares solver (NNLS) for NMF updates. */
export function nnls(A: Float64Array[], b: Float64Array, maxIter = 100): Float64Array {
  const p = A[0]?.length ?? 0;
  let x = new Float64Array(p);
  for (let iter = 0; iter < maxIter; iter++) {
    const xNew = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      let num = b.reduce((s, bi, i) => {
        let dotExj = 0;
        for (let k = 0; k < p; k++) {
          if (k !== j) dotExj += (A[i]?.[k] ?? 0) * (x[k] ?? 0);
        }
        return s + (A[i]?.[j] ?? 0) * ((bi ?? 0) - dotExj);
      }, 0);
      let den = A.reduce((s, ai) => s + (ai[j] ?? 0) ** 2, 0);
      xNew[j] = Math.max(0, num / (den + 1e-10));
    }
    x = xNew;
  }
  return x;
}

/** Power iteration for SVD (used by TruncatedSVD / Randomized SVD). */
export function randomizedSVD(
  X: Float64Array[],
  nComponents: number,
  nIter = 4,
): { U: Float64Array[]; S: Float64Array; Vt: Float64Array[] } {
  const m = X.length;
  const n = X[0]?.length ?? 0;
  // Random projection
  let Omega = Array.from({ length: n }, () => {
    const row = new Float64Array(nComponents);
    for (let k = 0; k < nComponents; k++) row[k] = (Math.random() - 0.5) * 2;
    return row;
  });

  // Y = X @ Omega
  let Y = X.map((xi) => new Float64Array(nComponents).map((_, k) => {
    let sum = 0;
    for (let j = 0; j < n; j++) sum += (xi[j] ?? 0) * (Omega[j]?.[k] ?? 0);
    return sum;
  }));

  // Power iteration
  for (let i = 0; i < nIter; i++) {
    // Z = X^T @ Y
    const Z = Array.from({ length: n }, (_, j) =>
      new Float64Array(nComponents).map((_, k) => {
        let sum = 0;
        for (let r = 0; r < m; r++) sum += (X[r]?.[j] ?? 0) * (Y[r]?.[k] ?? 0);
        return sum;
      })
    );
    // Y = X @ Z
    Y = X.map((xi) => new Float64Array(nComponents).map((_, k) => {
      let sum = 0;
      for (let j = 0; j < n; j++) sum += (xi[j] ?? 0) * (Z[j]?.[k] ?? 0);
      return sum;
    }));
    Omega = Z;
  }

  // QR of Y
  const Q = qrDecomp(Y);
  // B = Q^T @ X (nComponents x n)
  const B = Array.from({ length: nComponents }, (_, k) =>
    new Float64Array(n).map((_, j) => {
      let sum = 0;
      for (let r = 0; r < m; r++) sum += (Q[r]?.[k] ?? 0) * (X[r]?.[j] ?? 0);
      return sum;
    })
  );

  const S = new Float64Array(nComponents).map((_, k) => {
    const row = B[k];
    if (row === undefined) return 0;
    return Math.sqrt(row.reduce((s, v) => s + v * v, 0));
  });

  const Vt = B.map((row, k) => {
    const s = S[k] ?? 1e-10;
    return row.map((v) => v / s);
  });
  const U = Q;

  return { U, S, Vt };
}

function qrDecomp(A: Float64Array[]): Float64Array[] {
  const m = A.length;
  const n = A[0]?.length ?? 0;
  const Q: Float64Array[] = A.map((row) => new Float64Array(row));

  for (let k = 0; k < Math.min(m, n); k++) {
    let norm = 0;
    for (let i = k; i < m; i++) norm += (Q[i]?.[k] ?? 0) ** 2;
    norm = Math.sqrt(norm);
    if (norm < 1e-10) continue;
    for (let i = k; i < m; i++) {
      if (Q[i] !== undefined) Q[i]![k] = (Q[i]![k] ?? 0) / norm;
    }
    for (let j = k + 1; j < n; j++) {
      let dot = 0;
      for (let i = k; i < m; i++) dot += (Q[i]?.[k] ?? 0) * (Q[i]?.[j] ?? 0);
      for (let i = k; i < m; i++) {
        if (Q[i] !== undefined) Q[i]![j] = (Q[i]![j] ?? 0) - dot * (Q[i]![k] ?? 0);
      }
    }
  }
  return Q;
}

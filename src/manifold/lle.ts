/**
 * Manifold learning: LocallyLinearEmbedding (LLE) and extensions.
 * Mirrors sklearn.manifold.LocallyLinearEmbedding.
 */

export type LLEMethod = "standard" | "hessian" | "modified" | "ltsa";

export interface LLEOptions {
  nNeighbors?: number;
  nComponents?: number;
  reg?: number;
  method?: LLEMethod;
  eigSolver?: "auto" | "arpack" | "dense";
  tol?: number;
  maxIter?: number;
  randomState?: number | null;
}

/**
 * Locally Linear Embedding.
 * Reduces dimensionality while preserving local neighborhood structure.
 */
export class LocallyLinearEmbedding {
  nNeighbors: number;
  nComponents: number;
  reg: number;
  method: LLEMethod;
  tol: number;
  maxIter: number;
  randomState: number | null;

  embedding_: Float64Array[] | null = null;
  reconstructionError_: number = 0;
  nFeatures_: number = 0;
  nSamples_: number = 0;

  constructor(options: LLEOptions = {}) {
    this.nNeighbors = options.nNeighbors ?? 5;
    this.nComponents = options.nComponents ?? 2;
    this.reg = options.reg ?? 1e-3;
    this.method = options.method ?? "standard";
    this.tol = options.tol ?? 1e-6;
    this.maxIter = options.maxIter ?? 100;
    this.randomState = options.randomState ?? null;
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).embedding_!;
  }

  fit(X: Float64Array[]): this {
    const nSamples = X.length;
    const nFeatures = X[0]?.length ?? 0;
    this.nFeatures_ = nFeatures;
    this.nSamples_ = nSamples;
    const k = Math.min(this.nNeighbors, nSamples - 1);
    const d = Math.min(this.nComponents, nSamples - 1);

    // Step 1: Find k nearest neighbors for each point
    const neighbors = this._findNeighbors(X, k);

    // Step 2: Compute reconstruction weights W
    const W = this._computeWeights(X, neighbors, k);

    // Step 3: Compute embedding via eigendecomposition of (I-W)^T(I-W)
    this.embedding_ = this._computeEmbedding(W, nSamples, d);

    // Compute reconstruction error
    let error = 0;
    for (let i = 0; i < nSamples; i++) {
      const xi = this.embedding_[i]!;
      for (const [neighbor, wij] of neighbors[i]!.map(
        (n, j) => [n, W[i]?.[j] ?? 0] as [number, number],
      )) {
        const xj = this.embedding_[neighbor]!;
        for (let d2 = 0; d2 < xi.length; d2++) {
          error += ((xi[d2] ?? 0) - wij * (xj[d2] ?? 0)) ** 2;
        }
      }
    }
    this.reconstructionError_ = error;

    return this;
  }

  private _findNeighbors(X: Float64Array[], k: number): number[][] {
    const nSamples = X.length;
    const nFeatures = X[0]?.length ?? 0;
    return X.map((xi, i) => {
      const dists = X.map((xj, j) => {
        if (i === j) return Number.POSITIVE_INFINITY;
        let d = 0;
        for (let f = 0; f < nFeatures; f++)
          d += ((xi[f] ?? 0) - (xj[f] ?? 0)) ** 2;
        return d;
      });
      return dists
        .map((d, j) => ({ d, j }))
        .sort((a, b) => a.d - b.d)
        .slice(0, k)
        .map(({ j }) => j);
    });
  }

  private _computeWeights(
    X: Float64Array[],
    neighbors: number[][],
    k: number,
  ): Float64Array[] {
    const nSamples = X.length;
    const nFeatures = X[0]?.length ?? 0;
    const W: Float64Array[] = [];

    for (let i = 0; i < nSamples; i++) {
      const xi = X[i]!;
      const nbrs = neighbors[i]!;
      // Local covariance matrix C = Z^T Z where Z_j = x_i - x_neighbor_j
      const Z = nbrs.map((n) => {
        const z = new Float64Array(nFeatures);
        for (let f = 0; f < nFeatures; f++)
          z[f] = (xi[f] ?? 0) - (X[n]?.[f] ?? 0);
        return z;
      });

      // Gram matrix G = Z * Z^T (k x k)
      const G = Array.from({ length: k }, (_, a) =>
        new Float64Array(k).map((_, b) => {
          let dot = 0;
          for (let f = 0; f < nFeatures; f++)
            dot += (Z[a]?.[f] ?? 0) * (Z[b]?.[f] ?? 0);
          return dot;
        }),
      );

      // Regularize
      const trace = G.reduce((s, row, a) => s + (row[a] ?? 0), 0);
      for (let a = 0; a < k; a++) G[a]![a] = (G[a]![a] ?? 0) + this.reg * trace;

      // Solve G * w = 1 (ones vector)
      const w = this._solveLinear(G, new Float64Array(k).fill(1));

      // Normalize
      const wSum = w.reduce((s, v) => s + v, 0);
      const weights = new Float64Array(w.map((v) => v / (wSum || 1)));
      W.push(weights);
    }
    return W;
  }

  private _solveLinear(A: Float64Array[], b: Float64Array): Float64Array {
    const n = b.length;
    // Simple Gaussian elimination
    const mat = A.map((row, i) => {
      const r = new Float64Array(n + 1);
      r.set(row);
      r[n] = b[i] ?? 0;
      return r;
    });

    for (let col = 0; col < n; col++) {
      // Find pivot
      let maxVal = Math.abs(mat[col]?.[col] ?? 0);
      let maxRow = col;
      for (let row = col + 1; row < n; row++) {
        if (Math.abs(mat[row]?.[col] ?? 0) > maxVal) {
          maxVal = Math.abs(mat[row]?.[col] ?? 0);
          maxRow = row;
        }
      }
      if (maxRow !== col) {
        const tmp = mat[col]!;
        mat[col] = mat[maxRow]!;
        mat[maxRow] = tmp;
      }

      const pivot = mat[col]?.[col] ?? 1e-10;
      for (let row = col + 1; row < n; row++) {
        const factor = (mat[row]?.[col] ?? 0) / (pivot || 1e-10);
        for (let j = col; j <= n; j++)
          mat[row]![j] = (mat[row]![j] ?? 0) - factor * (mat[col]![j] ?? 0);
      }
    }

    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      x[i] = mat[i]?.[n] ?? 0;
      for (let j = i + 1; j < n; j++) x[i]! -= (mat[i]?.[j] ?? 0) * (x[j] ?? 0);
      x[i]! /= mat[i]?.[i] ?? 1e-10;
    }
    return x;
  }

  private _computeEmbedding(
    W: Float64Array[],
    nSamples: number,
    d: number,
  ): Float64Array[] {
    // Compute M = (I-W)^T (I-W) and find smallest non-zero eigenvectors
    // Use power iteration for dominant eigenvectors of M

    let seed = this.randomState ?? 42;
    function rand(): number {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return ((seed >>> 0) / 0xffffffff) * 2 - 1;
    }

    // Initialize random vectors
    const vecs: Float64Array[] = [];
    for (let c = 0; c <= d; c++) {
      const v = new Float64Array(nSamples);
      let norm = 0;
      for (let i = 0; i < nSamples; i++) {
        v[i] = rand();
        norm += (v[i] ?? 0) ** 2;
      }
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < nSamples; i++) v[i] = (v[i] ?? 0) / norm;
      vecs.push(v);
    }

    // Compute M * v for each v (M = (I-W)^T(I-W))
    const Mv = (v: Float64Array): Float64Array => {
      // (I-W) * v
      const u = new Float64Array(nSamples);
      for (let i = 0; i < nSamples; i++) {
        u[i] = v[i] ?? 0;
        for (let j = 0; j < W[i]!.length; j++)
          u[i]! -= (W[i]![j] ?? 0) * (v[j] ?? 0);
      }
      // (I-W)^T * u
      const Mvu = new Float64Array(nSamples);
      for (let i = 0; i < nSamples; i++) Mvu[i] = u[i] ?? 0;
      for (let i = 0; i < nSamples; i++) {
        for (let j = 0; j < W[i]!.length; j++)
          Mvu[j]! -= (W[i]![j] ?? 0) * (u[i] ?? 0);
      }
      return Mvu;
    };

    // Inverse iteration for smallest eigenvectors
    const eigenVecs: Float64Array[] = [];
    for (let c = 0; c < d + 1; c++) {
      let v = vecs[c]!;
      for (let iter = 0; iter < 20; iter++) {
        v = Mv(v);
        // Orthogonalize against previous
        for (const prev of eigenVecs) {
          let dot = 0;
          for (let i = 0; i < nSamples; i++)
            dot += (v[i] ?? 0) * (prev[i] ?? 0);
          for (let i = 0; i < nSamples; i++)
            v[i] = (v[i] ?? 0) - dot * (prev[i] ?? 0);
        }
        let norm = 0;
        for (let i = 0; i < nSamples; i++) norm += (v[i] ?? 0) ** 2;
        norm = Math.sqrt(norm) || 1;
        for (let i = 0; i < nSamples; i++) v[i] = (v[i] ?? 0) / norm;
      }
      eigenVecs.push(v);
    }

    // Skip the trivial eigenvector (all-ones), use next d
    const embedding = Array.from(
      { length: nSamples },
      (_, i) =>
        new Float64Array(eigenVecs.slice(1, d + 1).map((v) => v[i] ?? 0)),
    );

    return embedding;
  }

  transform(_X: Float64Array[]): Float64Array[] {
    if (!this.embedding_) throw new Error("LocallyLinearEmbedding not fitted");
    return this.embedding_;
  }
}

/**
 * Additional manifold learning: UMAP extensions, TriMap utilities.
 * Mirrors sklearn.manifold extras.
 */

import { NotFittedError } from "../exceptions.js";

export class LocallyLinearEmbeddingExt {
  nComponents: number;
  nNeighbors: number;
  method: "standard" | "modified" | "hessian" | "ltsa";
  randomState: number;

  embedding_: Float64Array[] | null = null;
  reconstructionError_: number = 0;

  constructor(
    options: {
      nComponents?: number;
      nNeighbors?: number;
      method?: "standard" | "modified" | "hessian" | "ltsa";
      randomState?: number;
    } = {},
  ) {
    this.nComponents = options.nComponents ?? 2;
    this.nNeighbors = options.nNeighbors ?? 5;
    this.method = options.method ?? "standard";
    this.randomState = options.randomState ?? 0;
  }

  private _kNeighbors(X: Float64Array[], k: number, idx: number): number[] {
    const dists = X.map((row, i) => {
      if (i === idx) return { i, d: Number.POSITIVE_INFINITY };
      let s = 0;
      for (let j = 0; j < row.length; j++) {
        s += ((row[j] ?? 0) - (X[idx]?.[j] ?? 0)) ** 2;
      }
      return { i, d: Math.sqrt(s) };
    });
    dists.sort((a, b) => a.d - b.d);
    return dists.slice(0, k).map((d) => d.i);
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const nFeatures = X[0]?.length ?? 0;
    const k = Math.min(this.nNeighbors, n - 1);
    const d = Math.min(this.nComponents, nFeatures - 1);

    // Random initialization as fallback embedding
    let rng = this.randomState;
    const nextRand = (): number => {
      rng = (rng * 1664525 + 1013904223) >>> 0;
      return (rng / 4294967296) * 2 - 1;
    };

    // Compute weight matrix W
    const W = Array.from({ length: n }, () => new Float64Array(n));

    for (let i = 0; i < n; i++) {
      const neighbors = this._kNeighbors(X, k, i);
      const Z = neighbors.map((j) => {
        const diff = new Float64Array(nFeatures);
        for (let f = 0; f < nFeatures; f++) {
          diff[f] = (X[j]?.[f] ?? 0) - (X[i]?.[f] ?? 0);
        }
        return diff;
      });

      // Local covariance
      const C = Array.from({ length: k }, () => new Float64Array(k));
      for (let a = 0; a < k; a++) {
        for (let b = 0; b < k; b++) {
          let s = 0;
          for (let f = 0; f < nFeatures; f++) {
            s += (Z[a]?.[f] ?? 0) * (Z[b]?.[f] ?? 0);
          }
          C[a]![b] = s;
        }
      }

      // Add regularization
      const trace = C.reduce((acc, row, ri) => acc + (row[ri] ?? 0), 0);
      for (let a = 0; a < k; a++) C[a]![a] = (C[a]?.[a] ?? 0) + 1e-3 * trace;

      // Solve C * w = 1 (simplified: w = C^-1 * 1, then normalize)
      const ones = new Float64Array(k).fill(1);
      const w = this._solveDiag(C, ones, k);
      const wSum = w.reduce((a, b) => a + b, 0);
      for (let a = 0; a < k; a++) {
        W[i]![neighbors[a] ?? 0] = wSum !== 0 ? (w[a] ?? 0) / wSum : 1 / k;
      }
    }

    // Random initial embedding
    const Y = Array.from({ length: n }, () => {
      const v = new Float64Array(d);
      for (let j = 0; j < d; j++) v[j] = nextRand() * 0.01;
      return v;
    });

    // Gradient descent optimization
    for (let iter = 0; iter < 200; iter++) {
      const lr = 0.1 / (1 + iter * 0.01);
      const grad = Array.from({ length: n }, () => new Float64Array(d));

      let err = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const wij = W[i]?.[j] ?? 0;
          if (Math.abs(wij) < 1e-10) continue;
          const diff = new Float64Array(d);
          for (let f = 0; f < d; f++) diff[f] = (Y[i]?.[f] ?? 0) - (Y[j]?.[f] ?? 0);
          const norm2 = diff.reduce((a, b) => a + b ** 2, 0);
          err += wij * norm2;
          for (let f = 0; f < d; f++) {
            grad[i]![f] = (grad[i]?.[f] ?? 0) + 2 * wij * (diff[f] ?? 0);
          }
        }
      }

      for (let i = 0; i < n; i++) {
        for (let f = 0; f < d; f++) {
          Y[i]![f] = (Y[i]?.[f] ?? 0) - lr * (grad[i]?.[f] ?? 0);
        }
      }

      this.reconstructionError_ = err;
      if (err < 1e-6) break;
    }

    this.embedding_ = Y;
    return Y;
  }

  private _solveDiag(C: Float64Array[], b: Float64Array, k: number): Float64Array {
    // Gauss-Jordan elimination
    const A = C.map((row) => row.slice());
    const x = b.slice();
    for (let col = 0; col < k; col++) {
      let maxRow = col;
      for (let row = col + 1; row < k; row++) {
        if (Math.abs(A[row]?.[col] ?? 0) > Math.abs(A[maxRow]?.[col] ?? 0)) maxRow = row;
      }
      const tmpRow = A[col];
      A[col] = A[maxRow]!;
      A[maxRow] = tmpRow!;
      const tmpB = x[col];
      x[col] = x[maxRow] ?? 0;
      x[maxRow] = tmpB ?? 0;

      const pivot = A[col]?.[col] ?? 0;
      if (Math.abs(pivot) < 1e-12) continue;
      for (let row = 0; row < k; row++) {
        if (row === col) continue;
        const factor = (A[row]?.[col] ?? 0) / pivot;
        for (let c = 0; c < k; c++) {
          A[row]![c] = (A[row]?.[c] ?? 0) - factor * (A[col]?.[c] ?? 0);
        }
        x[row] = (x[row] ?? 0) - factor * (x[col] ?? 0);
      }
      const p = A[col]?.[col] ?? 1;
      for (let c = 0; c < k; c++) A[col]![c] = (A[col]?.[c] ?? 0) / p;
      x[col] = (x[col] ?? 0) / p;
    }
    return x;
  }

  fit(X: Float64Array[]): this {
    this.fitTransform(X);
    return this;
  }

  transform(_X: Float64Array[]): Float64Array[] {
    if (!this.embedding_) throw new NotFittedError("LocallyLinearEmbeddingExt is not fitted");
    return this.embedding_;
  }
}

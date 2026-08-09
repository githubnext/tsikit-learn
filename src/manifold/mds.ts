/**
 * MDS (Multidimensional Scaling) and related manifold learning.
 * Mirrors sklearn.manifold.MDS.
 */

import { NotFittedError } from "../exceptions.js";

export interface MDSOptions {
  nComponents?: number;
  metric?: boolean;
  nInit?: number;
  maxIter?: number;
  verbose?: number;
  eps?: number;
  nJobs?: number | null;
  random_state?: number;
  dissimilarity?: "euclidean" | "precomputed";
}

/**
 * MDS — Multidimensional Scaling.
 * Projects high-dimensional data to lower dimensions preserving pairwise distances.
 */
export class MDS {
  nComponents: number;
  metric: boolean;
  nInit: number;
  maxIter: number;
  eps: number;
  randomState: number;
  dissimilarity: "euclidean" | "precomputed";
  nIter_: number = 0;
  stress_: number = 0;

  embedding_: Float64Array[] | null = null;

  constructor(options: MDSOptions = {}) {
    this.nComponents = options.nComponents ?? 2;
    this.metric = options.metric ?? true;
    this.nInit = options.nInit ?? 4;
    this.maxIter = options.maxIter ?? 300;
    this.eps = options.eps ?? 1e-3;
    this.randomState = options.random_state ?? 42;
    this.dissimilarity = options.dissimilarity ?? "euclidean";
  }

  private _euclideanDissim(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    return Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let k = 0; k < p; k++)
          s += ((X[i]![k] ?? 0) - (X[j]![k] ?? 0)) ** 2;
        row[j]! = Math.sqrt(s);
      }
      return row;
    });
  }

  private _smacof(
    D: Float64Array[],
    n: number,
  ): { embedding: Float64Array[]; stress: number; nIter: number } {
    const k = this.nComponents;
    let rng = this.randomState;
    const nextRng = () => {
      rng = (rng * 1664525 + 1013904223) >>> 0;
      return (rng / 4294967296) * 2 - 1;
    };

    // Initialize embedding randomly
    let X: Float64Array[] = Array.from({ length: n }, () => {
      const row = new Float64Array(k);
      for (let j = 0; j < k; j++) row[j]! = nextRng();
      return row;
    });

    let prevStress = Number.POSITIVE_INFINITY;

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Compute current distances
      const Dcurr: Float64Array[] = Array.from({ length: n }, (_, i) => {
        const row = new Float64Array(n);
        for (let j = 0; j < n; j++) {
          let s = 0;
          for (let kk = 0; kk < k; kk++)
            s += ((X[i]![kk] ?? 0) - (X[j]![kk] ?? 0)) ** 2;
          row[j]! = Math.sqrt(s);
        }
        return row;
      });

      // Compute stress
      let stress = 0;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const diff = (Dcurr[i]![j] ?? 0) - (D[i]![j] ?? 0);
          stress += diff * diff;
        }
      }

      if (Math.abs(prevStress - stress) < this.eps) {
        this.nIter_ = iter + 1;
        return { embedding: X, stress, nIter: iter + 1 };
      }
      prevStress = stress;

      // SMACOF update (B matrix)
      const Xnew: Float64Array[] = Array.from(
        { length: n },
        () => new Float64Array(k),
      );
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (i === j) continue;
          const dij = Dcurr[i]![j] ?? 0;
          const bij = dij < 1e-10 ? 0 : -(D[i]![j] ?? 0) / dij;
          for (let kk = 0; kk < k; kk++) {
            Xnew[i]![kk]! += bij * ((X[i]![kk] ?? 0) - (X[j]![kk] ?? 0));
          }
        }
        for (let kk = 0; kk < k; kk++) Xnew[i]![kk]! = (Xnew[i]![kk] ?? 0) / n;
      }
      X = Xnew;
    }

    let finalStress = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let s = 0;
        for (let kk = 0; kk < k; kk++)
          s += ((X[i]![kk] ?? 0) - (X[j]![kk] ?? 0)) ** 2;
        const dij = Math.sqrt(s);
        const diff = dij - (D[i]![j] ?? 0);
        finalStress += diff * diff;
      }
    }
    return { embedding: X, stress: finalStress, nIter: this.maxIter };
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const D =
      this.dissimilarity === "precomputed" ? X : this._euclideanDissim(X);

    let bestStress = Number.POSITIVE_INFINITY;
    let bestEmbedding: Float64Array[] = [];

    for (let init = 0; init < this.nInit; init++) {
      this.randomState += init;
      const { embedding, stress, nIter } = this._smacof(D, n);
      if (stress < bestStress) {
        bestStress = stress;
        bestEmbedding = embedding;
        this.nIter_ = nIter;
      }
    }

    this.stress_ = bestStress;
    this.embedding_ = bestEmbedding;
    return bestEmbedding;
  }

  fit(X: Float64Array[]): this {
    this.fitTransform(X);
    return this;
  }
}

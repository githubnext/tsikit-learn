/**
 * t-SNE (t-distributed Stochastic Neighbor Embedding).
 * Mirrors sklearn.manifold.TSNE.
 */

import { NotFittedError } from "../exceptions.js";

export interface TSNEOptions {
  nComponents?: number;
  perplexity?: number;
  learningRate?: number | "auto";
  nIter?: number;
  earlyExaggeration?: number;
  randomState?: number | null;
  verbose?: number;
}

export class TSNE {
  nComponents: number;
  perplexity: number;
  learningRate: number | "auto";
  nIter: number;
  earlyExaggeration: number;

  embedding_: Float64Array[] | null = null;
  klDivergence_: number | null = null;
  nIter_: number | null = null;

  constructor(options: TSNEOptions = {}) {
    this.nComponents = options.nComponents ?? 2;
    this.perplexity = options.perplexity ?? 30;
    this.learningRate = options.learningRate ?? "auto";
    this.nIter = options.nIter ?? 1000;
    this.earlyExaggeration = options.earlyExaggeration ?? 12;
  }

  private _pairwiseDistSq(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const D: Float64Array[] = Array.from(
      { length: n },
      () => new Float64Array(n),
    );
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let d = 0;
        const xi = X[i] ?? new Float64Array(0);
        const xj = X[j] ?? new Float64Array(0);
        for (let k = 0; k < xi.length; k++) {
          d += ((xi[k] ?? 0) - (xj[k] ?? 0)) ** 2;
        }
        (D[i] as Float64Array)[j] = d;
        (D[j] as Float64Array)[i] = d;
      }
    }
    return D;
  }

  private _binarySearchPerplexity(
    di: Float64Array,
    targetPerp: number,
    i: number,
  ): Float64Array {
    const n = di.length;
    const pi = new Float64Array(n);
    let beta = 1.0;
    const betaMin = Number.NEGATIVE_INFINITY;
    const betaMax = Number.POSITIVE_INFINITY;
    let betaMinL = betaMin;
    let betaMaxL = betaMax;
    const tol = 1e-5;
    const maxIter = 50;

    for (let iter = 0; iter < maxIter; iter++) {
      let sumP = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) {
          pi[j] = 0;
          continue;
        }
        pi[j] = Math.exp(-((di[j] ?? 0) * beta));
        sumP += pi[j] ?? 0;
      }
      if (sumP === 0) sumP = 1e-10;
      let H = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const p = (pi[j] ?? 0) / sumP;
        if (p > 1e-10) H -= p * Math.log2(p);
        pi[j] = p;
      }
      const hDiff = H - Math.log2(targetPerp);
      if (Math.abs(hDiff) < tol) break;
      if (hDiff > 0) {
        betaMinL = beta;
        beta =
          betaMaxL === Number.POSITIVE_INFINITY
            ? beta * 2
            : (beta + betaMaxL) / 2;
      } else {
        betaMaxL = beta;
        beta =
          betaMinL === Number.NEGATIVE_INFINITY
            ? beta / 2
            : (beta + betaMinL) / 2;
      }
      void betaMin;
      void betaMax;
    }
    return pi;
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const d = this.nComponents;
    const lr =
      this.learningRate === "auto"
        ? Math.max(n / (this.earlyExaggeration * 4), 50)
        : this.learningRate;

    // Compute pairwise distances
    const Dsq = this._pairwiseDistSq(X);

    // Compute P (symmetrized conditional probabilities)
    const P: Float64Array[] = Array.from(
      { length: n },
      () => new Float64Array(n),
    );
    for (let i = 0; i < n; i++) {
      const pi = this._binarySearchPerplexity(
        Dsq[i] as Float64Array,
        this.perplexity,
        i,
      );
      for (let j = 0; j < n; j++) {
        (P[i] as Float64Array)[j] = pi[j] ?? 0;
      }
    }
    // Symmetrize
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const val =
          ((P[i] as Float64Array)[j] ?? 0 + ((P[j] as Float64Array)[i] ?? 0)) /
          (2 * n);
        (P[i] as Float64Array)[j] = val;
        (P[j] as Float64Array)[i] = val;
      }
    }

    // Random initialization
    const Y: Float64Array[] = Array.from({ length: n }, () => {
      const yi = new Float64Array(d);
      for (let k = 0; k < d; k++) yi[k] = (Math.random() - 0.5) * 0.0001;
      return yi;
    });
    const gains: Float64Array[] = Array.from({ length: n }, () =>
      new Float64Array(d).fill(1),
    );
    const iY: Float64Array[] = Array.from(
      { length: n },
      () => new Float64Array(d),
    );

    const exag = this.earlyExaggeration;
    for (let iter = 0; iter < this.nIter; iter++) {
      const pMult = iter < 250 ? exag : 1;
      // Compute Q
      const num: Float64Array[] = Array.from(
        { length: n },
        () => new Float64Array(n),
      );
      let sumQ = 0;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          let distSq = 0;
          const yi = Y[i] as Float64Array;
          const yj = Y[j] as Float64Array;
          for (let k = 0; k < d; k++)
            distSq += ((yi[k] ?? 0) - (yj[k] ?? 0)) ** 2;
          const v = 1 / (1 + distSq);
          (num[i] as Float64Array)[j] = v;
          (num[j] as Float64Array)[i] = v;
          sumQ += 2 * v;
        }
      }
      if (sumQ === 0) sumQ = 1e-10;

      // Compute gradients
      const dY: Float64Array[] = Array.from(
        { length: n },
        () => new Float64Array(d),
      );
      let klDiv = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (i === j) continue;
          const p = (P[i] as Float64Array)[j] ?? 0;
          const q = ((num[i] as Float64Array)[j] ?? 0) / sumQ;
          const pq = pMult * p - q;
          const mult = 4 * pq * ((num[i] as Float64Array)[j] ?? 0);
          const yi = Y[i] as Float64Array;
          const yj = Y[j] as Float64Array;
          const dy = dY[i] as Float64Array;
          for (let k = 0; k < d; k++) {
            dy[k] = (dy[k] ?? 0) + mult * ((yi[k] ?? 0) - (yj[k] ?? 0));
          }
          if (p > 1e-12 && q > 1e-12) klDiv += p * Math.log(p / q);
        }
      }

      // Update
      for (let i = 0; i < n; i++) {
        const dy = dY[i] as Float64Array;
        const g = gains[i] as Float64Array;
        const iy = iY[i] as Float64Array;
        const yi = Y[i] as Float64Array;
        for (let k = 0; k < d; k++) {
          const gNew =
            Math.sign(dy[k] ?? 0) !== Math.sign(iy[k] ?? 0)
              ? (g[k] ?? 1) + 0.2
              : (g[k] ?? 1) * 0.8;
          g[k] = Math.max(gNew, 0.01);
          iy[k] = 0.8 * (iy[k] ?? 0) - lr * (g[k] ?? 1) * (dy[k] ?? 0);
          yi[k] = (yi[k] ?? 0) + (iy[k] ?? 0);
        }
      }

      if (iter === this.nIter - 1) this.klDivergence_ = klDiv;
    }

    this.embedding_ = Y;
    this.nIter_ = this.nIter;
    return Y;
  }

  fit(X: Float64Array[]): this {
    this.fitTransform(X);
    return this;
  }

  transform(_X: Float64Array[]): Float64Array[] {
    if (this.embedding_ === null)
      throw new NotFittedError("TSNE is not fitted.");
    throw new Error(
      "TSNE does not support transform on new data. Use fit_transform.",
    );
  }
}

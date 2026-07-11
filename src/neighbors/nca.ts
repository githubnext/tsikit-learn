import { BaseEstimator } from "../base.js";
/**
 * NeighborhoodComponentsAnalysis (NCA).
 * Mirrors sklearn.neighbors.NeighborhoodComponentsAnalysis.
 */
import { NotFittedError } from "../exceptions.js";

export interface NCAOptions {
  nComponents?: number;
  init?: "auto" | "pca" | "lda" | "identity" | "random";
  tol?: number;
  maxIter?: number;
  randomState?: number;
  verbose?: number;
}

/**
 * NeighborhoodComponentsAnalysis — learns a linear transformation that maximises
 * the classification accuracy of a leave-one-out k-NN classifier in the
 * transformed space.
 *
 * @example
 * const nca = new NeighborhoodComponentsAnalysis({ nComponents: 2 });
 * nca.fit(X, y);
 * const Xt = nca.transform(X);
 */
export class NeighborhoodComponentsAnalysis extends BaseEstimator {
  nComponents: number | undefined;
  init: "auto" | "pca" | "lda" | "identity" | "random";
  tol: number;
  maxIter: number;
  randomState: number;
  verbose: number;

  components_: Float64Array[] | undefined;
  n_iter_: number | undefined;
  n_features_in_: number | undefined;
  classes_: Int32Array | undefined;

  constructor(options: NCAOptions = {}) {
    super();
    this.nComponents = options.nComponents;
    this.init = options.init ?? "auto";
    this.tol = options.tol ?? 1e-5;
    this.maxIter = options.maxIter ?? 50;
    this.randomState = options.randomState ?? 0;
    this.verbose = options.verbose ?? 0;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    const nComp = this.nComponents ?? d;

    this.n_features_in_ = d;
    const uniqueClasses = new Set<number>();
    for (let i = 0; i < n; i++) uniqueClasses.add(y[i] ?? 0);
    this.classes_ = new Int32Array([...uniqueClasses].sort((a, b) => a - b));

    // Initialise transformation matrix A (nComp × d) as random or identity
    const A: Float64Array[] = [];
    const rng = this._rng(this.randomState);
    if (this.init === "identity" || nComp === d) {
      for (let i = 0; i < nComp; i++) {
        const row = new Float64Array(d);
        if (i < d) row[i] = 1.0;
        A.push(row);
      }
    } else {
      for (let i = 0; i < nComp; i++) {
        const row = new Float64Array(d);
        for (let j = 0; j < d; j++) row[j] = rng() * 0.01;
        A.push(row);
      }
    }

    // Gradient-descent optimisation with finite-difference gradient
    let iter = 0;
    const lr = 0.001;
    for (iter = 0; iter < this.maxIter; iter++) {
      const Ax = X.map((x) => this._transform(A, x));
      const { loss, grad } = this._lossGrad(Ax, y, A, X, n, nComp, d);
      // SGD step
      let maxGrad = 0;
      for (let i = 0; i < nComp; i++) {
        for (let j = 0; j < d; j++) {
          const g = grad[i]![j] ?? 0;
          A[i]![j]! -= lr * g;
          if (Math.abs(g) > maxGrad) maxGrad = Math.abs(g);
        }
      }
      if (this.verbose > 0)
        console.log(`NCA iter ${iter}, loss=${loss.toFixed(6)}`);
      if (maxGrad < this.tol) break;
    }

    this.components_ = A;
    this.n_iter_ = iter;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_) throw new NotFittedError("NCA is not fitted");
    return X.map((x) => this._transform(this.components_!, x));
  }

  fitTransform(X: Float64Array[], y: Int32Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }

  private _transform(A: Float64Array[], x: Float64Array): Float64Array {
    const out = new Float64Array(A.length);
    for (let i = 0; i < A.length; i++) {
      let s = 0;
      const row = A[i]!;
      for (let j = 0; j < row.length; j++) s += (row[j] ?? 0) * (x[j] ?? 0);
      out[i] = s;
    }
    return out;
  }

  private _lossGrad(
    Ax: Float64Array[],
    y: Int32Array,
    A: Float64Array[],
    X: Float64Array[],
    n: number,
    nComp: number,
    d: number,
  ): { loss: number; grad: Float64Array[] } {
    const grad: Float64Array[] = Array.from(
      { length: nComp },
      () => new Float64Array(d),
    );
    let loss = 0;

    for (let i = 0; i < n; i++) {
      const axi = Ax[i]!;
      const yi = y[i] ?? 0;
      // Softmax over distances in transformed space
      const dists = new Float64Array(n);
      for (let k = 0; k < n; k++) {
        if (k === i) {
          dists[k] = 0;
          continue;
        }
        let sq = 0;
        const axk = Ax[k]!;
        for (let c = 0; c < nComp; c++) {
          const diff = (axi[c] ?? 0) - (axk[c] ?? 0);
          sq += diff * diff;
        }
        dists[k] = sq;
      }
      // Compute softmax weights
      const maxD = Math.max(...dists.filter((_, k) => k !== i));
      let sumExp = 0;
      const expD = new Float64Array(n);
      for (let k = 0; k < n; k++) {
        if (k === i) continue;
        expD[k] = Math.exp(-(dists[k] ?? 0) + maxD);
        sumExp += expD[k]!;
      }
      // p_ij = exp(-d_ij) / sum_k exp(-d_ik)  for k≠i
      // p_i  = sum_{j: class(j)==class(i)} p_ij  (prob of correct class)
      let pi = 0;
      for (let k = 0; k < n; k++) {
        if (k === i) continue;
        if ((y[k] ?? 0) === yi) pi += (expD[k] ?? 0) / sumExp;
      }
      loss += 1 - pi;
      // Gradient contribution (simplified stochastic gradient)
      for (let k = 0; k < n; k++) {
        if (k === i) continue;
        const pij = (expD[k] ?? 0) / sumExp;
        const sameClass = (y[k] ?? 0) === yi ? 1 : 0;
        const coeff = 2 * pij * (pi - sameClass);
        for (let c = 0; c < nComp; c++) {
          const diff = (axi[c] ?? 0) - (Ax[k]![c] ?? 0);
          for (let j = 0; j < d; j++) {
            grad[c]![j]! += coeff * diff * ((X[i]![j] ?? 0) - (X[k]![j] ?? 0));
          }
        }
      }
    }
    return { loss: loss / n, grad };
  }

  private _rng(seed: number): () => number {
    let s = seed | 0;
    return () => {
      s = (s ^ (s << 13)) >>> 0;
      s = (s ^ (s >>> 17)) >>> 0;
      s = (s ^ (s << 5)) >>> 0;
      return (s >>> 0) / 0xffffffff;
    };
  }

  getParams(): NCAOptions {
    return {
      nComponents: this.nComponents,
      init: this.init,
      tol: this.tol,
      maxIter: this.maxIter,
      randomState: this.randomState,
      verbose: this.verbose,
    };
  }
}

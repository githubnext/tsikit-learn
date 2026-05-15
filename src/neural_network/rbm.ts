/**
 * Bernoulli Restricted Boltzmann Machine (BernoulliRBM).
 * Mirrors sklearn.neural_network.BernoulliRBM.
 */

import { NotFittedError } from "../exceptions.js";

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] ?? 0) * (b[i] ?? 0);
  return s;
}

export interface BernoulliRBMOptions {
  nComponents?: number;
  learningRate?: number;
  batchSize?: number;
  nIter?: number;
  randomState?: number;
  verbose?: number;
}

/**
 * Bernoulli Restricted Boltzmann Machine trained with CD-k.
 * Mirrors sklearn.neural_network.BernoulliRBM.
 */
export class BernoulliRBM {
  nComponents: number;
  learningRate: number;
  batchSize: number;
  nIter: number;
  randomState: number;
  verbose: number;

  components_: Float64Array[] | null = null; // nComponents x nVisible
  interceptHidden_: Float64Array | null = null;
  interceptVisible_: Float64Array | null = null;
  nIter_: number = 0;

  private rng_: () => number;

  constructor(options: BernoulliRBMOptions = {}) {
    this.nComponents = options.nComponents ?? 256;
    this.learningRate = options.learningRate ?? 0.1;
    this.batchSize = options.batchSize ?? 10;
    this.nIter = options.nIter ?? 10;
    this.randomState = options.randomState ?? 0;
    this.verbose = options.verbose ?? 0;

    // Simple LCG RNG seeded by randomState
    let seed = this.randomState + 1;
    this.rng_ = () => {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return (seed >>> 0) / 0xffffffff;
    };
  }

  private sample(probs: Float64Array): Float64Array {
    const s = new Float64Array(probs.length);
    for (let i = 0; i < probs.length; i++) s[i] = this.rng_() < (probs[i] ?? 0) ? 1 : 0;
    return s;
  }

  /** Compute P(h=1 | v) for each hidden unit. */
  private propUp(v: Float64Array): Float64Array {
    const h = new Float64Array(this.nComponents);
    for (let j = 0; j < this.nComponents; j++) {
      let s = this.interceptHidden_![j] ?? 0;
      const w = this.components_![j] ?? new Float64Array(0);
      s += dot(w, v);
      h[j] = sigmoid(s);
    }
    return h;
  }

  /** Compute P(v=1 | h) for each visible unit. */
  private propDown(h: Float64Array, nVisible: number): Float64Array {
    const v = new Float64Array(nVisible);
    for (let i = 0; i < nVisible; i++) {
      let s = this.interceptVisible_![i] ?? 0;
      for (let j = 0; j < this.nComponents; j++) {
        s += ((this.components_![j] ?? new Float64Array(0))[i] ?? 0) * (h[j] ?? 0);
      }
      v[i] = sigmoid(s);
    }
    return v;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const nVisible = (X[0] ?? new Float64Array(0)).length;

    // Initialize weights
    this.components_ = Array.from({ length: this.nComponents }, () => {
      const w = new Float64Array(nVisible);
      for (let i = 0; i < nVisible; i++) w[i] = (this.rng_() - 0.5) * 0.1;
      return w;
    });
    this.interceptHidden_ = new Float64Array(this.nComponents);
    this.interceptVisible_ = new Float64Array(nVisible);

    // Contrastive Divergence (CD-1)
    for (let iter = 0; iter < this.nIter; iter++) {
      // Shuffle indices
      const perm = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(this.rng_() * (i + 1));
        const tmp = perm[i]!; perm[i] = perm[j]!; perm[j] = tmp;
      }

      for (let start = 0; start < n; start += this.batchSize) {
        const batchIdx = perm.slice(start, start + this.batchSize);
        const dW: Float64Array[] = Array.from({ length: this.nComponents }, () => new Float64Array(nVisible));
        const dHBias = new Float64Array(this.nComponents);
        const dVBias = new Float64Array(nVisible);

        for (const i of batchIdx) {
          const v0 = X[i] ?? new Float64Array(nVisible);
          const h0Prob = this.propUp(v0);
          const h0 = this.sample(h0Prob);

          // CD-1: one Gibbs step
          const v1Prob = this.propDown(h0, nVisible);
          const v1 = this.sample(v1Prob);
          const h1Prob = this.propUp(v1);

          // Accumulate gradients: <v0 h0> - <v1 h1>
          for (let j = 0; j < this.nComponents; j++) {
            const dj = dW[j] ?? new Float64Array(nVisible);
            for (let vi = 0; vi < nVisible; vi++) {
              dj[vi] = (dj[vi] ?? 0) + (v0[vi] ?? 0) * (h0Prob[j] ?? 0) - (v1[vi] ?? 0) * (h1Prob[j] ?? 0);
            }
            dHBias[j] = (dHBias[j] ?? 0) + (h0Prob[j] ?? 0) - (h1Prob[j] ?? 0);
          }
          for (let vi = 0; vi < nVisible; vi++) {
            dVBias[vi] = (dVBias[vi] ?? 0) + (v0[vi] ?? 0) - (v1[vi] ?? 0);
          }
        }

        const bs = batchIdx.length;
        const lr = this.learningRate / bs;

        for (let j = 0; j < this.nComponents; j++) {
          const wj = this.components_![j] ?? new Float64Array(nVisible);
          const dj = dW[j] ?? new Float64Array(nVisible);
          for (let vi = 0; vi < nVisible; vi++) wj[vi] = (wj[vi] ?? 0) + lr * (dj[vi] ?? 0);
          this.interceptHidden_![j] = (this.interceptHidden_![j] ?? 0) + lr * (dHBias[j] ?? 0);
        }
        for (let vi = 0; vi < nVisible; vi++) {
          this.interceptVisible_![vi] = (this.interceptVisible_![vi] ?? 0) + lr * (dVBias[vi] ?? 0);
        }
      }
      this.nIter_ = iter + 1;
    }

    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_) throw new NotFittedError("BernoulliRBM is not fitted yet.");
    return X.map((xi) => this.propUp(xi));
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  /** Pseudo-log-likelihood score (proxy for likelihood). */
  score(X: Float64Array[]): number {
    if (!this.components_) throw new NotFittedError("BernoulliRBM is not fitted yet.");
    const nVisible = (X[0] ?? new Float64Array(0)).length;
    let total = 0;
    for (const v of X) {
      // Free energy: -b_v v - sum_j log(1 + exp(b_h_j + W_j v))
      let fe = 0;
      for (let vi = 0; vi < nVisible; vi++) fe -= (this.interceptVisible_![vi] ?? 0) * (v[vi] ?? 0);
      for (let j = 0; j < this.nComponents; j++) {
        const s = (this.interceptHidden_![j] ?? 0) + dot(this.components_![j] ?? new Float64Array(0), v);
        fe -= Math.log(1 + Math.exp(s));
      }
      total += fe;
    }
    return total / X.length;
  }
}

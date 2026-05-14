/**
 * FastICA (Independent Component Analysis) and LatentDirichletAllocation.
 * Mirrors sklearn.decomposition.FastICA and LatentDirichletAllocation.
 */

import { NotFittedError } from "../exceptions.js";

function logcosh(x: number): number {
  return Math.log(Math.cosh(x));
}

function dlogcosh(x: number): number {
  return Math.tanh(x);
}

function d2logcosh(x: number): number {
  const t = Math.tanh(x);
  return 1 - t * t;
}

export type FastICAFunction = "logcosh" | "exp" | "cube";

export interface FastICAOptions {
  nComponents?: number;
  algorithm?: "parallel" | "deflation";
  fun?: FastICAFunction;
  maxIter?: number;
  tol?: number;
  whiten?: boolean;
}

export class FastICA {
  nComponents: number | null;
  algorithm: "parallel" | "deflation";
  fun: FastICAFunction;
  maxIter: number;
  tol: number;
  whiten: boolean;

  components_: Float64Array[] | null = null;
  mixing_: Float64Array[] | null = null;
  mean_: Float64Array | null = null;
  whitening_: Float64Array[] | null = null;
  nIter_: number = 0;

  constructor(options: FastICAOptions = {}) {
    this.nComponents = options.nComponents ?? null;
    this.algorithm = options.algorithm ?? "parallel";
    this.fun = options.fun ?? "logcosh";
    this.maxIter = options.maxIter ?? 200;
    this.tol = options.tol ?? 1e-4;
    this.whiten = options.whiten ?? true;
  }

  private _gFunc(x: number): [number, number] {
    switch (this.fun) {
      case "logcosh":
        return [dlogcosh(x), d2logcosh(x)];
      case "exp": {
        const ex = Math.exp(-(x * x) / 2);
        return [x * ex, (1 - x * x) * ex];
      }
      default:
        return [x * x * x, 3 * x * x];
    }
  }

  fit(X: Float64Array[]): this {
    this.fitTransform(X);
    return this;
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const k = Math.min(this.nComponents ?? p, p, n);

    // Center
    const mean = new Float64Array(p);
    for (const row of X) for (let j = 0; j < p; j++) mean[j]! += (row[j] ?? 0) / n;
    this.mean_ = mean;

    const Xc = X.map((row) => {
      const r = new Float64Array(p);
      for (let j = 0; j < p; j++) r[j] = (row[j] ?? 0) - (mean[j] ?? 0);
      return r;
    });

    // PCA whitening (simplified)
    let Xw: Float64Array[] = Xc;
    const W: Float64Array[][] = [];

    if (this.whiten) {
      // Covariance matrix (p x p), simplified via SVD-like approach
      // Use thin approach: compute XtX
      const cov: number[][] = Array.from({ length: p }, () => new Array<number>(p).fill(0));
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < p; j++) {
          for (let l = j; l < p; l++) {
            cov[j]![l]! += (Xc[i]![j] ?? 0) * (Xc[i]![l] ?? 0);
            if (l !== j) cov[l]![j]! = cov[j]![l]!;
          }
        }
      }
      for (let j = 0; j < p; j++) for (let l = 0; l < p; l++) cov[j]![l]! /= n;

      // Diagonal whitening (simplified: divide by std)
      const scales = new Float64Array(p);
      for (let j = 0; j < p; j++) scales[j] = 1 / (Math.sqrt(Math.max(cov[j]![j] ?? 1, 1e-10)));
      Xw = Xc.map((row) => row.map((v, j) => v * (scales[j] ?? 1)));
      this.whitening_ = [scales.map((s) => s)].map(() => scales);
    }

    // FastICA deflation
    const components: Float64Array[] = [];

    for (let c = 0; c < k; c++) {
      // Random init
      let w = new Float64Array(p).map(() => Math.random() - 0.5);
      let wNorm = 0;
      for (let j = 0; j < p; j++) wNorm += (w[j] ?? 0) ** 2;
      wNorm = Math.sqrt(wNorm);
      w = w.map((v) => v / wNorm);

      // Orthogonalize against previous components
      for (const wPrev of components) {
        let dot = 0;
        for (let j = 0; j < p; j++) dot += (w[j] ?? 0) * (wPrev[j] ?? 0);
        for (let j = 0; j < p; j++) w[j]! -= dot * (wPrev[j] ?? 0);
        let n2 = 0;
        for (let j = 0; j < p; j++) n2 += (w[j] ?? 0) ** 2;
        const norm = Math.sqrt(n2);
        for (let j = 0; j < p; j++) w[j]! /= norm || 1;
      }

      let converged = false;
      for (let iter = 0; iter < this.maxIter; iter++) {
        // w_new = E[x * g(w^T x)] - E[g'(w^T x)] * w
        const wNew = new Float64Array(p);
        let expG2 = 0;

        for (const xi of Xw) {
          let wx = 0;
          for (let j = 0; j < p; j++) wx += (w[j] ?? 0) * (xi[j] ?? 0);
          const [gWx, g2Wx] = this._gFunc(wx);
          for (let j = 0; j < p; j++) wNew[j]! += gWx * (xi[j] ?? 0);
          expG2 += g2Wx;
        }

        for (let j = 0; j < p; j++) {
          wNew[j] = (wNew[j]! / n) - (expG2 / n) * (w[j] ?? 0);
        }

        // Orthogonalize
        for (const wPrev of components) {
          let dot = 0;
          for (let j = 0; j < p; j++) dot += (wNew[j] ?? 0) * (wPrev[j] ?? 0);
          for (let j = 0; j < p; j++) wNew[j]! -= dot * (wPrev[j] ?? 0);
        }

        // Normalize
        let n2 = 0;
        for (let j = 0; j < p; j++) n2 += (wNew[j] ?? 0) ** 2;
        const norm = Math.sqrt(n2);
        for (let j = 0; j < p; j++) wNew[j]! /= norm || 1;

        // Check convergence: |w^T w_new| should be close to 1
        let dot = 0;
        for (let j = 0; j < p; j++) dot += (w[j] ?? 0) * (wNew[j] ?? 0);

        w = wNew;
        this.nIter_ = iter + 1;

        if (Math.abs(Math.abs(dot) - 1) < this.tol) {
          converged = true;
          break;
        }
      }

      components.push(w);
    }

    this.components_ = components;

    // Mixing matrix (pseudo-inverse of components)
    this.mixing_ = components.map((w) => new Float64Array(w));

    // Return transformed data
    return Xw.map((xi) => {
      const out = new Float64Array(k);
      for (let c = 0; c < k; c++) {
        for (let j = 0; j < p; j++) out[c]! += (components[c]![j] ?? 0) * (xi[j] ?? 0);
      }
      return out;
    });
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_ || !this.mean_) throw new NotFittedError("FastICA");
    const p = this.mean_.length;
    const k = this.components_.length;

    const Xc = X.map((row) => {
      const r = new Float64Array(p);
      for (let j = 0; j < p; j++) r[j] = (row[j] ?? 0) - (this.mean_![j] ?? 0);
      return r;
    });

    const Xw = this.whiten && this.whitening_
      ? Xc.map((row) => row.map((v, j) => v * (this.whitening_![0]![j] ?? 1)))
      : Xc;

    return Xw.map((xi) => {
      const out = new Float64Array(k);
      for (let c = 0; c < k; c++) {
        for (let j = 0; j < p; j++) out[c]! += (this.components_![c]![j] ?? 0) * (xi[j] ?? 0);
      }
      return out;
    });
  }
}

export interface LDAOptions {
  nComponents?: number;
  maxIter?: number;
  learningDecay?: number;
  learningOffset?: number;
  batchSize?: number;
}

export class LatentDirichletAllocation {
  nComponents: number;
  maxIter: number;
  learningDecay: number;
  learningOffset: number;
  batchSize: number;

  components_: Float64Array[] | null = null;
  nBatchIter_: number = 0;
  nIter_: number = 0;

  constructor(options: LDAOptions = {}) {
    this.nComponents = options.nComponents ?? 10;
    this.maxIter = options.maxIter ?? 10;
    this.learningDecay = options.learningDecay ?? 0.7;
    this.learningOffset = options.learningOffset ?? 10;
    this.batchSize = options.batchSize ?? 128;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const nFeatures = X[0]?.length ?? 0;
    const K = this.nComponents;

    // Initialize component distributions (K topics x nFeatures words)
    const lambda = Array.from({ length: K }, () => {
      const row = new Float64Array(nFeatures).map(() => Math.random() + 0.1);
      const sum = row.reduce((a, b) => a + b, 0);
      return row.map((v) => v / sum);
    });

    for (let iter = 0; iter < this.maxIter; iter++) {
      const batch: Float64Array[] = [];
      for (let b = 0; b < this.batchSize; b++) {
        batch.push(X[Math.floor(Math.random() * n)]!);
      }

      // E-step: compute document-topic distributions
      const gamma: Float64Array[] = batch.map(() => {
        const g = new Float64Array(K).fill(1.0 / K);
        return g;
      });

      // Simplified variational E-step (1 iteration)
      for (let di = 0; di < batch.length; di++) {
        const doc = batch[di]!;
        const docTotal = doc.reduce((a, b) => a + b, 0) || 1;

        for (let vi = 0; vi < nFeatures; vi++) {
          const wCount = (doc[vi] ?? 0) / docTotal;
          if (wCount < 1e-10) continue;

          // phi_dvk proportional to exp(digamma(gamma_dk)) * lambda_kv
          let phiSum = 0;
          const phi = new Float64Array(K);
          for (let k = 0; k < K; k++) {
            phi[k] = Math.exp(Math.log(gamma[di]![k] ?? 1e-10) + Math.log(lambda[k]![vi] ?? 1e-10));
            phiSum += phi[k] ?? 0;
          }

          for (let k = 0; k < K; k++) {
            gamma[di]![k]! += wCount * ((phi[k] ?? 0) / (phiSum || 1));
          }
        }

        // Normalize gamma
        const gSum = gamma[di]!.reduce((a, b) => a + b, 0) || 1;
        for (let k = 0; k < K; k++) gamma[di]![k]! /= gSum;
      }

      // M-step: update lambda
      const ro = Math.pow(this.learningOffset + iter, -this.learningDecay);

      for (let k = 0; k < K; k++) {
        const newLambda = new Float64Array(nFeatures).fill(0.1);
        for (let di = 0; di < batch.length; di++) {
          const doc = batch[di]!;
          const gk = gamma[di]![k] ?? 0;
          for (let vi = 0; vi < nFeatures; vi++) {
            newLambda[vi]! += gk * (doc[vi] ?? 0);
          }
        }
        // Normalize
        const sum = newLambda.reduce((a, b) => a + b, 0) || 1;
        for (let vi = 0; vi < nFeatures; vi++) newLambda[vi]! /= sum;

        // Interpolate
        for (let vi = 0; vi < nFeatures; vi++) {
          lambda[k]![vi] = (1 - ro) * (lambda[k]![vi] ?? 0) + ro * (newLambda[vi] ?? 0);
        }
      }
      this.nIter_ = iter + 1;
      this.nBatchIter_++;
    }

    this.components_ = lambda.map((row) => new Float64Array(row));
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_) throw new NotFittedError("LatentDirichletAllocation");
    const K = this.nComponents;
    const nFeatures = this.components_[0]?.length ?? 0;

    return X.map((doc) => {
      const docTotal = doc.reduce((a, b) => a + b, 0) || 1;
      const gamma = new Float64Array(K).fill(1.0 / K);

      // Simplified E-step
      for (let vi = 0; vi < nFeatures; vi++) {
        const wCount = (doc[vi] ?? 0) / docTotal;
        if (wCount < 1e-10) continue;

        let phiSum = 0;
        const phi = new Float64Array(K);
        for (let k = 0; k < K; k++) {
          phi[k] = Math.exp(
            Math.log(gamma[k] ?? 1e-10) +
            Math.log(this.components_![k]![vi] ?? 1e-10),
          );
          phiSum += phi[k] ?? 0;
        }

        for (let k = 0; k < K; k++) {
          gamma[k]! += wCount * ((phi[k] ?? 0) / (phiSum || 1));
        }
      }

      const sum = gamma.reduce((a, b) => a + b, 0) || 1;
      return gamma.map((v) => v / sum);
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

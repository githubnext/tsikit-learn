/**
 * Additional clustering algorithms: SelfOrganizingMap, FuzzyCMeans, AffinityPropagationExt
 * Port of sklearn-compatible clustering extensions
 */

import { NotFittedError } from "../exceptions.js";

export class SelfOrganizingMap {
  rows: number;
  cols: number;
  nFeatures: number;
  sigma: number;
  learningRate: number;
  nIter: number;
  randomState: number;

  private weights_: Float64Array[] | null = null;

  constructor(opts: {
    rows?: number;
    cols?: number;
    nFeatures?: number;
    sigma?: number;
    learningRate?: number;
    nIter?: number;
    randomState?: number;
  } = {}) {
    this.rows = opts.rows ?? 10;
    this.cols = opts.cols ?? 10;
    this.nFeatures = opts.nFeatures ?? 2;
    this.sigma = opts.sigma ?? 1.0;
    this.learningRate = opts.learningRate ?? 0.5;
    this.nIter = opts.nIter ?? 1000;
    this.randomState = opts.randomState ?? 42;
  }

  private _rng(seed: number): () => number {
    let s = seed;
    return () => {
      s = (s * 1664525 + 1013904223) & 0xffffffff;
      return (s >>> 0) / 0xffffffff;
    };
  }

  fit(X: Float64Array[]): this {
    const rng = this._rng(this.randomState);
    const nNodes = this.rows * this.cols;
    this.weights_ = Array.from({ length: nNodes }, () => {
      const w = new Float64Array(this.nFeatures);
      for (let j = 0; j < this.nFeatures; j++) w[j] = rng() * 2 - 1;
      return w;
    });
    for (let iter = 0; iter < this.nIter; iter++) {
      const t = iter / this.nIter;
      const lr = this.learningRate * Math.exp(-t * 5);
      const sig = this.sigma * Math.exp(-t * 5);
      const xi = X[Math.floor(rng() * X.length)];
      if (!xi) continue;
      let bmuIdx = 0;
      let bmuDist = Number.POSITIVE_INFINITY;
      for (let k = 0; k < nNodes; k++) {
        const w = this.weights_[k];
        if (!w) continue;
        let d = 0;
        for (let j = 0; j < this.nFeatures; j++) d += ((xi[j] ?? 0) - (w[j] ?? 0)) ** 2;
        if (d < bmuDist) { bmuDist = d; bmuIdx = k; }
      }
      const bmuRow = Math.floor(bmuIdx / this.cols);
      const bmuCol = bmuIdx % this.cols;
      for (let k = 0; k < nNodes; k++) {
        const r = Math.floor(k / this.cols);
        const c = k % this.cols;
        const dist2 = (r - bmuRow) ** 2 + (c - bmuCol) ** 2;
        const h = Math.exp(-dist2 / (2 * sig * sig + 1e-15));
        const w = this.weights_[k];
        if (!w) continue;
        for (let j = 0; j < this.nFeatures; j++) {
          w[j] = (w[j] ?? 0) + lr * h * ((xi[j] ?? 0) - (w[j] ?? 0));
        }
      }
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.weights_) throw new NotFittedError("SelfOrganizingMap not fitted.");
    return X.map(xi => {
      const result = new Float64Array(this.weights_!.length);
      for (let k = 0; k < this.weights_!.length; k++) {
        const w = this.weights_![k];
        let d = 0;
        if (w) for (let j = 0; j < this.nFeatures; j++) d += ((xi[j] ?? 0) - (w[j] ?? 0)) ** 2;
        result[k] = Math.sqrt(d);
      }
      return result;
    });
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.weights_) throw new NotFittedError("SelfOrganizingMap not fitted.");
    const labels = new Int32Array(X.length);
    for (let i = 0; i < X.length; i++) {
      const xi = X[i];
      if (!xi) continue;
      let bmu = 0;
      let bmuDist = Number.POSITIVE_INFINITY;
      for (let k = 0; k < this.weights_!.length; k++) {
        const w = this.weights_![k];
        let d = 0;
        if (w) for (let j = 0; j < this.nFeatures; j++) d += ((xi[j] ?? 0) - (w[j] ?? 0)) ** 2;
        if (d < bmuDist) { bmuDist = d; bmu = k; }
      }
      labels[i] = bmu;
    }
    return labels;
  }
}

export class FuzzyCMeans {
  nClusters: number;
  m: number;
  maxIter: number;
  tol: number;
  randomState: number;

  clusterCenters_: Float64Array[] | null = null;
  u_: Float64Array[] | null = null;
  labels_: Int32Array | null = null;

  constructor(opts: {
    nClusters?: number;
    m?: number;
    maxIter?: number;
    tol?: number;
    randomState?: number;
  } = {}) {
    this.nClusters = opts.nClusters ?? 3;
    this.m = opts.m ?? 2.0;
    this.maxIter = opts.maxIter ?? 150;
    this.tol = opts.tol ?? 1e-4;
    this.randomState = opts.randomState ?? 42;
  }

  private _rng(seed: number): () => number {
    let s = seed;
    return () => {
      s = (s * 1664525 + 1013904223) & 0xffffffff;
      return (s >>> 0) / 0xffffffff;
    };
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const c = this.nClusters;
    const rng = this._rng(this.randomState);
    let u: Float64Array[] = Array.from({ length: n }, () => {
      const row = new Float64Array(c);
      let sum = 0;
      for (let k = 0; k < c; k++) { row[k] = rng(); sum += row[k] ?? 0; }
      for (let k = 0; k < c; k++) row[k] = (row[k] ?? 0) / (sum + 1e-15);
      return row;
    });

    for (let iter = 0; iter < this.maxIter; iter++) {
      const centers: Float64Array[] = Array.from({ length: c }, () => new Float64Array(p));
      for (let k = 0; k < c; k++) {
        let wSum = 0;
        for (let i = 0; i < n; i++) {
          const uik = Math.pow(u[i]![k] ?? 0, this.m);
          wSum += uik;
          const xi = X[i];
          if (!xi) continue;
          for (let j = 0; j < p; j++) centers[k]![j] = (centers[k]![j] ?? 0) + uik * (xi[j] ?? 0);
        }
        for (let j = 0; j < p; j++) centers[k]![j] = (centers[k]![j] ?? 0) / (wSum + 1e-15);
      }
      const newU: Float64Array[] = Array.from({ length: n }, () => new Float64Array(c));
      for (let i = 0; i < n; i++) {
        const xi = X[i];
        const dists = new Float64Array(c);
        for (let k = 0; k < c; k++) {
          let d = 0;
          const ck = centers[k];
          if (xi && ck) for (let j = 0; j < p; j++) d += ((xi[j] ?? 0) - (ck[j] ?? 0)) ** 2;
          dists[k] = Math.sqrt(d) + 1e-15;
        }
        for (let k = 0; k < c; k++) {
          let s = 0;
          const dk = dists[k] ?? 1;
          for (let l = 0; l < c; l++) s += Math.pow(dk / ((dists[l] ?? 1) + 1e-15), 2 / (this.m - 1 + 1e-15));
          newU[i]![k] = 1 / (s + 1e-15);
        }
      }
      let diff = 0;
      for (let i = 0; i < n; i++) for (let k = 0; k < c; k++) diff = Math.max(diff, Math.abs((newU[i]![k] ?? 0) - (u[i]![k] ?? 0)));
      u = newU;
      if (diff < this.tol) break;
      void iter;
    }
    this.u_ = u;
    this.clusterCenters_ = Array.from({ length: c }, () => new Float64Array(p));
    for (let k = 0; k < c; k++) {
      let wSum = 0;
      for (let i = 0; i < n; i++) {
        const uik = Math.pow(u[i]![k] ?? 0, this.m);
        wSum += uik;
        const xi = X[i];
        if (!xi) continue;
        for (let j = 0; j < p; j++) this.clusterCenters_[k]![j] = (this.clusterCenters_[k]![j] ?? 0) + uik * (xi[j] ?? 0);
      }
      for (let j = 0; j < p; j++) this.clusterCenters_[k]![j] = (this.clusterCenters_[k]![j] ?? 0) / (wSum + 1e-15);
    }
    this.labels_ = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      let bestK = 0;
      let bestU = -1;
      for (let k = 0; k < c; k++) {
        if ((u[i]![k] ?? 0) > bestU) { bestU = u[i]![k] ?? 0; bestK = k; }
      }
      this.labels_[i] = bestK;
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.clusterCenters_) throw new NotFittedError("FuzzyCMeans not fitted.");
    const labels = new Int32Array(X.length);
    for (let i = 0; i < X.length; i++) {
      const xi = X[i];
      let bestK = 0;
      let bestDist = Number.POSITIVE_INFINITY;
      for (let k = 0; k < this.clusterCenters_.length; k++) {
        const ck = this.clusterCenters_[k];
        let d = 0;
        if (xi && ck) for (let j = 0; j < ck.length; j++) d += ((xi[j] ?? 0) - (ck[j] ?? 0)) ** 2;
        if (d < bestDist) { bestDist = d; bestK = k; }
      }
      labels[i] = bestK;
    }
    return labels;
  }
}

export class GaussianMixtureExt {
  nComponents: number;
  maxIter: number;
  tol: number;
  randomState: number;

  means_: Float64Array[] | null = null;
  covs_: Float64Array[][] | null = null;
  weights_: Float64Array | null = null;

  constructor(opts: { nComponents?: number; maxIter?: number; tol?: number; randomState?: number } = {}) {
    this.nComponents = opts.nComponents ?? 3;
    this.maxIter = opts.maxIter ?? 100;
    this.tol = opts.tol ?? 1e-3;
    this.randomState = opts.randomState ?? 0;
  }

  private _gaussPdf(x: Float64Array, mu: Float64Array, cov: Float64Array[]): number {
    const p = x.length;
    let det = 1;
    for (let j = 0; j < p; j++) det *= cov[j]![j] ?? 1;
    const norm = Math.pow(2 * Math.PI, p / 2) * Math.sqrt(Math.abs(det) + 1e-15);
    let exp = 0;
    for (let j = 0; j < p; j++) {
      const diff = (x[j] ?? 0) - (mu[j] ?? 0);
      exp += diff * diff / ((cov[j]![j] ?? 1) + 1e-15);
    }
    return Math.exp(-0.5 * exp) / (norm + 1e-15);
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 1;
    const c = this.nComponents;
    let rngState = this.randomState;
    const rng = () => { rngState = (rngState * 1664525 + 1013904223) & 0xffffffff; return (rngState >>> 0) / 0xffffffff; };

    this.means_ = Array.from({ length: c }, () => {
      const m = new Float64Array(p);
      for (let j = 0; j < p; j++) m[j] = rng() * 2 - 1;
      return m;
    });
    this.covs_ = Array.from({ length: c }, () => Array.from({ length: p }, () => { const r = new Float64Array(p); r[0] = 1; return r; }));
    this.weights_ = new Float64Array(c).fill(1 / c);

    for (let iter = 0; iter < this.maxIter; iter++) {
      const resp = Array.from({ length: n }, () => new Float64Array(c));
      for (let i = 0; i < n; i++) {
        let total = 0;
        for (let k = 0; k < c; k++) {
          const r = (this.weights_![k] ?? 0) * this._gaussPdf(X[i]!, this.means_![k]!, this.covs_![k]!);
          resp[i]![k] = r;
          total += r;
        }
        for (let k = 0; k < c; k++) resp[i]![k] = (resp[i]![k] ?? 0) / (total + 1e-15);
      }
      const Nk = new Float64Array(c);
      for (let i = 0; i < n; i++) for (let k = 0; k < c; k++) Nk[k] = (Nk[k] ?? 0) + (resp[i]![k] ?? 0);
      for (let k = 0; k < c; k++) {
        const nk = Nk[k] ?? 1;
        const mu = new Float64Array(p);
        for (let i = 0; i < n; i++) {
          const rik = resp[i]![k] ?? 0;
          const xi = X[i];
          if (!xi) continue;
          for (let j = 0; j < p; j++) mu[j] = (mu[j] ?? 0) + rik * (xi[j] ?? 0);
        }
        for (let j = 0; j < p; j++) mu[j] = (mu[j] ?? 0) / (nk + 1e-15);
        this.means_![k] = mu;
        const cov: Float64Array[] = Array.from({ length: p }, () => new Float64Array(p));
        for (let i = 0; i < n; i++) {
          const rik = resp[i]![k] ?? 0;
          const xi = X[i];
          if (!xi) continue;
          for (let j = 0; j < p; j++) {
            cov[j]![j] = (cov[j]![j] ?? 0) + rik * ((xi[j] ?? 0) - (mu[j] ?? 0)) ** 2;
          }
        }
        for (let j = 0; j < p; j++) cov[j]![j] = (cov[j]![j] ?? 0) / (nk + 1e-15) + 1e-6;
        this.covs_![k] = cov;
        this.weights_![k] = nk / n;
      }
      void iter;
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.means_) throw new NotFittedError("GaussianMixtureExt not fitted.");
    const labels = new Int32Array(X.length);
    for (let i = 0; i < X.length; i++) {
      let best = 0;
      let bestScore = -Number.POSITIVE_INFINITY;
      for (let k = 0; k < this.nComponents; k++) {
        const score = Math.log((this.weights_![k] ?? 0) + 1e-15) + Math.log(this._gaussPdf(X[i]!, this.means_![k]!, this.covs_![k]!) + 1e-15);
        if (score > bestScore) { bestScore = score; best = k; }
      }
      labels[i] = best;
    }
    return labels;
  }
}

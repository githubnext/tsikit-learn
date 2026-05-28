/**
 * Feature selection extensions: ReliefF, MRMR, Boruta
 * Port of sklearn-compatible feature selection methods
 */

import { NotFittedError } from "../exceptions.js";

function euclideanDist(a: Float64Array, b: Float64Array): number {
  let d = 0;
  for (let j = 0; j < a.length; j++) d += ((a[j] ?? 0) - (b[j] ?? 0)) ** 2;
  return Math.sqrt(d);
}

export class ReliefF {
  nFeatures: number;
  nNeighbors: number;
  nIter: number;
  randomState: number;

  featureImportances_: Float64Array | null = null;
  selectedIndices_: number[] | null = null;

  constructor(opts: { nFeatures?: number; nNeighbors?: number; nIter?: number; randomState?: number } = {}) {
    this.nFeatures = opts.nFeatures ?? 10;
    this.nNeighbors = opts.nNeighbors ?? 10;
    this.nIter = opts.nIter ?? 50;
    this.randomState = opts.randomState ?? 0;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const weights = new Float64Array(p);
    let seed = this.randomState;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };

    const classes = new Set<number>();
    for (let i = 0; i < n; i++) classes.add(y[i] ?? 0);
    const classCounts: Record<number, number> = {};
    for (const c of classes) {
      classCounts[c] = 0;
      for (let i = 0; i < n; i++) if ((y[i] ?? 0) === c) classCounts[c]++;
    }

    for (let iter = 0; iter < this.nIter; iter++) {
      const idx = Math.floor(rng() * n);
      const xi = X[idx]!;
      const yi = y[idx] ?? 0;
      const dists = Array.from({ length: n }, (_, j) => ({ j, d: j === idx ? Number.POSITIVE_INFINITY : euclideanDist(xi, X[j]!) }));
      dists.sort((a, b) => a.d - b.d);
      const hits: number[] = [];
      const missByClass: Record<number, number[]> = {};
      for (const c of classes) if (c !== yi) missByClass[c] = [];
      for (const { j } of dists) {
        if (hits.length >= this.nNeighbors && Object.values(missByClass).every(m => m.length >= this.nNeighbors)) break;
        const yj = y[j] ?? 0;
        if (yj === yi && hits.length < this.nNeighbors) hits.push(j);
        else if (yj !== yi && (missByClass[yj]?.length ?? 0) < this.nNeighbors) missByClass[yj]?.push(j);
      }
      for (let f = 0; f < p; f++) {
        let hitDiff = 0;
        for (const h of hits) hitDiff += Math.abs((xi[f] ?? 0) - (X[h]![f] ?? 0));
        weights[f] = (weights[f] ?? 0) - hitDiff / (this.nNeighbors * this.nIter + 1e-15);
        for (const [cls, misses] of Object.entries(missByClass)) {
          const c = Number(cls);
          const prob = (classCounts[c] ?? 0) / n;
          let missDiff = 0;
          for (const m of misses) missDiff += Math.abs((xi[f] ?? 0) - (X[m]![f] ?? 0));
          weights[f] = (weights[f] ?? 0) + prob * missDiff / (this.nNeighbors * this.nIter + 1e-15);
        }
      }
      void iter;
    }
    this.featureImportances_ = weights;
    const order = Array.from({ length: p }, (_, i) => i).sort((a, b) => (weights[b] ?? 0) - (weights[a] ?? 0));
    this.selectedIndices_ = order.slice(0, Math.min(this.nFeatures, p));
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.selectedIndices_) throw new NotFittedError("ReliefF not fitted.");
    return X.map(xi => {
      const r = new Float64Array(this.selectedIndices_!.length);
      for (let j = 0; j < this.selectedIndices_!.length; j++) r[j] = xi[this.selectedIndices_![j]!] ?? 0;
      return r;
    });
  }

  fitTransform(X: Float64Array[], y: Int32Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}

export class MRMRFeatureSelector {
  nFeatures: number;

  selectedIndices_: number[] | null = null;
  scores_: Float64Array | null = null;

  constructor(opts: { nFeatures?: number } = {}) {
    this.nFeatures = opts.nFeatures ?? 10;
  }

  private mutualInfo(x: Float64Array, y: Float64Array | Int32Array, bins = 10): number {
    const n = x.length;
    const xMin = x.reduce((a, b) => Math.min(a, b), Number.POSITIVE_INFINITY);
    const xMax = x.reduce((a, b) => Math.max(a, b), -Number.POSITIVE_INFINITY);
    const yMin = Array.from(y).reduce((a, b) => Math.min(a, b), Number.POSITIVE_INFINITY);
    const yMax = Array.from(y).reduce((a, b) => Math.max(a, b), -Number.POSITIVE_INFINITY);
    const xRange = xMax - xMin + 1e-15;
    const yRange = yMax - yMin + 1e-15;
    const joint: Float64Array[] = Array.from({ length: bins }, () => new Float64Array(bins));
    for (let i = 0; i < n; i++) {
      const xi = Math.min(bins - 1, Math.floor(((x[i] ?? 0) - xMin) / xRange * bins));
      const yi = Math.min(bins - 1, Math.floor(((y[i] ?? 0) - yMin) / yRange * bins));
      joint[xi]![yi] = (joint[xi]![yi] ?? 0) + 1 / n;
    }
    const px = new Float64Array(bins);
    const py = new Float64Array(bins);
    for (let i = 0; i < bins; i++) for (let j = 0; j < bins; j++) {
      px[i] = (px[i] ?? 0) + (joint[i]![j] ?? 0);
      py[j] = (py[j] ?? 0) + (joint[i]![j] ?? 0);
    }
    let mi = 0;
    for (let i = 0; i < bins; i++) for (let j = 0; j < bins; j++) {
      const pij = joint[i]![j] ?? 0;
      if (pij > 0) mi += pij * Math.log((pij + 1e-15) / ((px[i] ?? 1e-15) * (py[j] ?? 1e-15) + 1e-15));
    }
    return Math.max(0, mi);
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const p = X[0]?.length ?? 0;
    const k = Math.min(this.nFeatures, p);
    const yFloat = Float64Array.from(y);
    const cols = Array.from({ length: p }, (_, j) => Float64Array.from(X.map(xi => xi[j] ?? 0)));
    const relev = new Float64Array(p);
    for (let j = 0; j < p; j++) relev[j] = this.mutualInfo(cols[j]!, yFloat);
    const selected: number[] = [];
    const remaining = new Set(Array.from({ length: p }, (_, i) => i));
    for (let s = 0; s < k; s++) {
      let bestScore = -Number.POSITIVE_INFINITY;
      let bestFeat = 0;
      for (const j of remaining) {
        const red = selected.length === 0 ? 0 : selected.reduce((sum, sel) => sum + this.mutualInfo(cols[j]!, cols[sel]!), 0) / selected.length;
        const score = (relev[j] ?? 0) - red;
        if (score > bestScore) { bestScore = score; bestFeat = j; }
      }
      selected.push(bestFeat);
      remaining.delete(bestFeat);
    }
    this.selectedIndices_ = selected;
    this.scores_ = Float64Array.from(selected.map(j => relev[j] ?? 0));
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.selectedIndices_) throw new NotFittedError("MRMRFeatureSelector not fitted.");
    return X.map(xi => {
      const r = new Float64Array(this.selectedIndices_!.length);
      for (let j = 0; j < this.selectedIndices_!.length; j++) r[j] = xi[this.selectedIndices_![j]!] ?? 0;
      return r;
    });
  }

  fitTransform(X: Float64Array[], y: Int32Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}

export class BorutaFeatureSelector {
  maxIter: number;
  alpha: number;
  randomState: number;
  twoStep: boolean;

  selectedIndices_: number[] | null = null;
  importances_: Float64Array | null = null;

  constructor(opts: { maxIter?: number; alpha?: number; randomState?: number; twoStep?: boolean } = {}) {
    this.maxIter = opts.maxIter ?? 20;
    this.alpha = opts.alpha ?? 0.05;
    this.randomState = opts.randomState ?? 0;
    this.twoStep = opts.twoStep ?? true;
  }

  fit(X: Float64Array[], y: Int32Array, importanceFn?: (X: Float64Array[], y: Int32Array) => Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    let seed = this.randomState;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };

    const hits = new Float64Array(p);
    for (let iter = 0; iter < this.maxIter; iter++) {
      const shadowX = X.map(xi => {
        const shadow = new Float64Array(p);
        for (let j = 0; j < p; j++) shadow[j] = X[Math.floor(rng() * n)]![j] ?? 0;
        const combined = new Float64Array(p * 2);
        for (let j = 0; j < p; j++) { combined[j] = xi[j] ?? 0; combined[j + p] = shadow[j] ?? 0; }
        return combined;
      });
      let imps: Float64Array;
      if (importanceFn) {
        imps = importanceFn(shadowX, y);
      } else {
        imps = new Float64Array(p * 2);
        for (let j = 0; j < p * 2; j++) {
          let mi = 0;
          const col = Float64Array.from(X.map((_, i) => shadowX[i]![j] ?? 0));
          const mu = col.reduce((a, b) => a + b, 0) / n;
          for (let i = 0; i < n; i++) mi += Math.abs((col[i] ?? 0) - mu) * (y[i] ?? 0);
          imps[j] = mi / n;
        }
      }
      const shadowMax = imps.slice(p).reduce((a, b) => Math.max(a, b), 0);
      for (let j = 0; j < p; j++) if ((imps[j] ?? 0) > shadowMax) hits[j]++;
      void iter;
    }
    const threshold = this.maxIter * (1 - this.alpha);
    this.importances_ = hits;
    this.selectedIndices_ = Array.from({ length: p }, (_, i) => i).filter(i => (hits[i] ?? 0) >= threshold);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.selectedIndices_) throw new NotFittedError("BorutaFeatureSelector not fitted.");
    return X.map(xi => {
      const r = new Float64Array(this.selectedIndices_!.length);
      for (let j = 0; j < this.selectedIndices_!.length; j++) r[j] = xi[this.selectedIndices_![j]!] ?? 0;
      return r;
    });
  }
}

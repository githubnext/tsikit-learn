/**
 * Imputation extensions: MatrixCompletion, IterativeImputerExt, SoftImputeExt
 * Port of sklearn.impute extensions
 */

import { NotFittedError } from "../exceptions.js";

export class IterativeImputerExt {
  maxIter: number;
  tol: number;
  estimatorType: "linear" | "mean";
  randomState: number;

  private statistics_: Float64Array | null = null;
  private nFeatures_ = 0;

  constructor(opts: {
    maxIter?: number;
    tol?: number;
    estimatorType?: "linear" | "mean";
    randomState?: number;
  } = {}) {
    this.maxIter = opts.maxIter ?? 10;
    this.tol = opts.tol ?? 1e-3;
    this.estimatorType = opts.estimatorType ?? "linear";
    this.randomState = opts.randomState ?? 0;
  }

  fit(X: (Float64Array | (number | null)[])[]): this {
    this.nFeatures_ = X[0]?.length ?? 0;
    this.statistics_ = this._columnMeans(X);
    return this;
  }

  private _columnMeans(X: (Float64Array | (number | null)[])[]): Float64Array {
    const p = X[0]?.length ?? 0;
    const means = new Float64Array(p);
    const counts = new Float64Array(p);
    for (const xi of X) {
      for (let j = 0; j < p; j++) {
        const v = xi[j];
        if (v !== null && v !== undefined && !Number.isNaN(Number(v))) {
          means[j] = (means[j] ?? 0) + Number(v);
          counts[j]++;
        }
      }
    }
    for (let j = 0; j < p; j++) means[j] = (means[j] ?? 0) / ((counts[j] ?? 0) + 1e-15);
    return means;
  }

  transform(X: (Float64Array | (number | null)[])[]): Float64Array[] {
    if (!this.statistics_) throw new NotFittedError("IterativeImputerExt not fitted.");
    const filled: Float64Array[] = X.map(xi => {
      const r = new Float64Array(this.nFeatures_);
      for (let j = 0; j < this.nFeatures_; j++) {
        const v = xi[j];
        r[j] = (v === null || v === undefined || Number.isNaN(Number(v))) ? (this.statistics_![j] ?? 0) : Number(v);
      }
      return r;
    });
    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxChange = 0;
      for (let j = 0; j < this.nFeatures_; j++) {
        const missing = X.map((xi, i) => {
          const v = xi[j];
          return v === null || v === undefined || Number.isNaN(Number(v)) ? i : -1;
        }).filter(i => i >= 0);
        if (missing.length === 0) continue;
        const observed = X.map((xi, i) => {
          const v = xi[j];
          return (v === null || v === undefined || Number.isNaN(Number(v))) ? -1 : i;
        }).filter(i => i >= 0);
        if (observed.length === 0) continue;
        const obsMean = observed.reduce((s, i) => s + (filled[i]![j] ?? 0), 0) / observed.length;
        for (const i of missing) {
          const otherFeats = new Float64Array(this.nFeatures_ - 1);
          let fIdx = 0;
          for (let k = 0; k < this.nFeatures_; k++) if (k !== j) { otherFeats[fIdx++] = filled[i]![k] ?? 0; }
          let pred = obsMean;
          if (this.estimatorType === "linear" && observed.length > 1) {
            const obsFeats = observed.map(oi => { const r = new Float64Array(this.nFeatures_ - 1); let idx = 0; for (let k = 0; k < this.nFeatures_; k++) if (k !== j) r[idx++] = filled[oi]![k] ?? 0; return r; });
            const obsY = Float64Array.from(observed.map(oi => filled[oi]![j] ?? 0));
            let dotSelf = 0;
            let dotY = 0;
            const meanX = new Float64Array(this.nFeatures_ - 1);
            for (const feat of obsFeats) for (let k = 0; k < feat.length; k++) meanX[k] = (meanX[k] ?? 0) + (feat[k] ?? 0) / obsFeats.length;
            for (let i2 = 0; i2 < obsFeats.length; i2++) {
              const xi = obsFeats[i2]!;
              for (let k = 0; k < xi.length; k++) {
                const diff = (xi[k] ?? 0) - (meanX[k] ?? 0);
                dotSelf += diff * diff;
                dotY += diff * ((obsY[i2] ?? 0) - obsMean);
              }
            }
            const beta = dotY / (dotSelf + 1e-15);
            let xDotMean = 0;
            for (let k = 0; k < otherFeats.length; k++) xDotMean += ((otherFeats[k] ?? 0) - (meanX[k] ?? 0));
            pred = obsMean + beta * xDotMean;
          }
          const oldVal = filled[i]![j] ?? 0;
          filled[i]![j] = pred;
          maxChange = Math.max(maxChange, Math.abs(pred - oldVal));
        }
      }
      if (maxChange < this.tol) break;
      void iter;
    }
    return filled;
  }

  fitTransform(X: (Float64Array | (number | null)[])[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class MatrixCompletionImputer {
  rank: number;
  maxIter: number;
  tol: number;
  lambda_: number;

  private U_: Float64Array[] | null = null;
  private V_: Float64Array[] | null = null;
  private means_: Float64Array | null = null;

  constructor(opts: { rank?: number; maxIter?: number; tol?: number; lambda_?: number } = {}) {
    this.rank = opts.rank ?? 5;
    this.maxIter = opts.maxIter ?? 50;
    this.tol = opts.tol ?? 1e-4;
    this.lambda_ = opts.lambda_ ?? 0.1;
  }

  fit(X: (number | null | undefined)[][]): this {
    const m = X.length;
    const n = X[0]?.length ?? 0;
    const r = this.rank;
    let seed = 42;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    this.means_ = new Float64Array(n);
    const counts = new Float64Array(n);
    for (const xi of X) for (let j = 0; j < n; j++) {
      const v = xi[j];
      if (v !== null && v !== undefined) { this.means_[j] = (this.means_[j] ?? 0) + v; counts[j]++; }
    }
    for (let j = 0; j < n; j++) this.means_[j] = (this.means_[j] ?? 0) / ((counts[j] ?? 1) + 1e-15);
    this.U_ = Array.from({ length: m }, () => { const row = new Float64Array(r); for (let k = 0; k < r; k++) row[k] = rng() * 0.01; return row; });
    this.V_ = Array.from({ length: n }, () => { const row = new Float64Array(r); for (let k = 0; k < r; k++) row[k] = rng() * 0.01; return row; });
    for (let iter = 0; iter < this.maxIter; iter++) {
      let totalLoss = 0;
      for (let i = 0; i < m; i++) {
        const VtV = Array.from({ length: r }, (_, k) => new Float64Array(r));
        const Vr = new Float64Array(r);
        for (let j = 0; j < n; j++) {
          const v = X[i]![j];
          if (v === null || v === undefined) continue;
          const vj = this.V_![j]!;
          const rij = v - (this.means_![j] ?? 0);
          for (let k = 0; k < r; k++) { Vr[k] = (Vr[k] ?? 0) + rij * (vj[k] ?? 0); for (let l = 0; l < r; l++) VtV[k]![l] = (VtV[k]![l] ?? 0) + (vj[k] ?? 0) * (vj[l] ?? 0); }
        }
        for (let k = 0; k < r; k++) VtV[k]![k] = (VtV[k]![k] ?? 0) + this.lambda_;
        for (let k = 0; k < r; k++) this.U_![i]![k] = (Vr[k] ?? 0) / ((VtV[k]![k] ?? 1) + 1e-15);
        totalLoss += 0;
      }
      for (let j = 0; j < n; j++) {
        const UtU = Array.from({ length: r }, (_, k) => new Float64Array(r));
        const Ur = new Float64Array(r);
        for (let i = 0; i < m; i++) {
          const v = X[i]![j];
          if (v === null || v === undefined) continue;
          const ui = this.U_![i]!;
          const rij = v - (this.means_![j] ?? 0);
          for (let k = 0; k < r; k++) { Ur[k] = (Ur[k] ?? 0) + rij * (ui[k] ?? 0); for (let l = 0; l < r; l++) UtU[k]![l] = (UtU[k]![l] ?? 0) + (ui[k] ?? 0) * (ui[l] ?? 0); }
        }
        for (let k = 0; k < r; k++) UtU[k]![k] = (UtU[k]![k] ?? 0) + this.lambda_;
        for (let k = 0; k < r; k++) this.V_![j]![k] = (Ur[k] ?? 0) / ((UtU[k]![k] ?? 1) + 1e-15);
      }
      void totalLoss;
      void iter;
    }
    return this;
  }

  transform(X: (number | null | undefined)[][]): Float64Array[] {
    if (!this.U_ || !this.V_ || !this.means_) throw new NotFittedError("MatrixCompletionImputer not fitted.");
    return X.map((xi, i) => {
      const r = new Float64Array(xi.length);
      for (let j = 0; j < xi.length; j++) {
        const v = xi[j];
        if (v !== null && v !== undefined) {
          r[j] = v;
        } else {
          const ui = this.U_![i] ?? new Float64Array(this.rank);
          const vj = this.V_![j] ?? new Float64Array(this.rank);
          let pred = this.means_![j] ?? 0;
          for (let k = 0; k < this.rank; k++) pred += (ui[k] ?? 0) * (vj[k] ?? 0);
          r[j] = pred;
        }
      }
      return r;
    });
  }

  fitTransform(X: (number | null | undefined)[][]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class KNNImputerExt {
  nNeighbors: number;
  weights: "uniform" | "distance";

  private trainingData_: Float64Array[] | null = null;
  private nFeatures_ = 0;

  constructor(opts: { nNeighbors?: number; weights?: "uniform" | "distance" } = {}) {
    this.nNeighbors = opts.nNeighbors ?? 5;
    this.weights = opts.weights ?? "uniform";
  }

  fit(X: Float64Array[]): this {
    this.trainingData_ = X.map(xi => xi.slice());
    this.nFeatures_ = X[0]?.length ?? 0;
    return this;
  }

  transform(X: (number | null | undefined)[][]): Float64Array[] {
    if (!this.trainingData_) throw new NotFittedError("KNNImputerExt not fitted.");
    return X.map(xi => {
      const r = new Float64Array(this.nFeatures_);
      const missingCols: number[] = [];
      for (let j = 0; j < this.nFeatures_; j++) {
        const v = xi[j];
        if (v === null || v === undefined || Number.isNaN(Number(v))) missingCols.push(j);
        else r[j] = Number(v);
      }
      if (missingCols.length === 0) return r;
      const availCols = Array.from({ length: this.nFeatures_ }, (_, j) => j).filter(j => !missingCols.includes(j));
      const dists = this.trainingData_!.map(train => {
        let d = 0;
        let count = 0;
        for (const j of availCols) {
          d += ((Number(xi[j] ?? 0)) - (train[j] ?? 0)) ** 2;
          count++;
        }
        return count > 0 ? Math.sqrt(d / count) : Number.POSITIVE_INFINITY;
      });
      const order = Array.from({ length: dists.length }, (_, i) => i).sort((a, b) => (dists[a] ?? 0) - (dists[b] ?? 0));
      const knn = order.slice(0, this.nNeighbors);
      for (const j of missingCols) {
        let weightSum = 0;
        let valSum = 0;
        for (const k of knn) {
          const w = this.weights === "distance" ? 1 / ((dists[k] ?? 0) + 1e-15) : 1;
          valSum += w * (this.trainingData_![k]![j] ?? 0);
          weightSum += w;
        }
        r[j] = valSum / (weightSum + 1e-15);
      }
      return r;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X as unknown as (number | null | undefined)[][]);
  }
}

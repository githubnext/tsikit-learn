/**
 * Neighbors extensions: NeighborhoodComponentsAnalysis, LocallyLinearEmbedding neighbors.
 * Mirrors sklearn.neighbors advanced methods.
 */

import { BaseEstimator } from "../base.js";

export interface NearestNeighborsGraphParams {
  n_neighbors?: number;
  mode?: "connectivity" | "distance";
  metric?: "euclidean" | "manhattan";
}

/** NearestNeighborsGraph: construct graph of k-nearest neighbors. */
export class NearestNeighborsGraph extends BaseEstimator {
  n_neighbors: number;
  mode: "connectivity" | "distance";
  metric: "euclidean" | "manhattan";
  X_: Float64Array[] = [];

  constructor(params: NearestNeighborsGraphParams = {}) {
    super();
    this.n_neighbors = params.n_neighbors ?? 5;
    this.mode = params.mode ?? "connectivity";
    this.metric = params.metric ?? "euclidean";
  }

  fit(X: Float64Array[]): this {
    this.X_ = X;
    return this;
  }

  kneighbors_graph(): { indices: Int32Array[]; distances: Float64Array[] } {
    const n = this.X_.length;
    const k = this.n_neighbors;
    const indices: Int32Array[] = [];
    const distances: Float64Array[] = [];
    for (let i = 0; i < n; i++) {
      const dists = this.X_.map((xj, j) => ({ j, d: this._dist(this.X_[i]!, xj) }));
      dists.sort((a, b) => a.d - b.d);
      const neighbors = dists.slice(1, k + 1);
      indices.push(new Int32Array(neighbors.map((x) => x.j)));
      distances.push(new Float64Array(neighbors.map((x) => x.d)));
    }
    return { indices, distances };
  }

  transform(X: Float64Array[]): { indices: Int32Array[]; distances: Float64Array[] } {
    const k = this.n_neighbors;
    const indices: Int32Array[] = [];
    const distances: Float64Array[] = [];
    for (const xi of X) {
      const dists = this.X_.map((xj, j) => ({ j, d: this._dist(xi, xj) }));
      dists.sort((a, b) => a.d - b.d);
      const neighbors = dists.slice(0, k);
      indices.push(new Int32Array(neighbors.map((x) => x.j)));
      distances.push(new Float64Array(neighbors.map((x) => x.d)));
    }
    return { indices, distances };
  }

  private _dist(a: Float64Array, b: Float64Array): number {
    let d = 0;
    for (let k = 0; k < a.length; k++) {
      const diff = (a[k] ?? 0) - (b[k] ?? 0);
      if (this.metric === "manhattan") d += Math.abs(diff);
      else d += diff * diff;
    }
    return this.metric === "manhattan" ? d : Math.sqrt(d);
  }
}

export interface LocalOutlierFactorExtParams {
  n_neighbors?: number;
  contamination?: number;
  metric?: "euclidean" | "manhattan";
  novelty?: boolean;
}

/** LocalOutlierFactor: unsupervised outlier detection. */
export class LocalOutlierFactorExt extends BaseEstimator {
  n_neighbors: number;
  contamination: number;
  metric: "euclidean" | "manhattan";
  novelty: boolean;
  negative_outlier_factor_: Float64Array = new Float64Array(0);
  threshold_: number = -1.5;
  X_: Float64Array[] = [];

  constructor(params: LocalOutlierFactorExtParams = {}) {
    super();
    this.n_neighbors = params.n_neighbors ?? 20;
    this.contamination = params.contamination ?? 0.1;
    this.metric = params.metric ?? "euclidean";
    this.novelty = params.novelty ?? false;
  }

  fit(X: Float64Array[]): this {
    this.X_ = X;
    const n = X.length;
    const k = this.n_neighbors;
    const knn = this._computeKNN(X);
    const lrd = this._computeLRD(X, knn, k);
    const lofs = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < k; j++) s += (lrd[knn.indices[i]?.[j] ?? 0] ?? 0) / (lrd[i] ?? 1);
      lofs[i] = s / k;
    }
    this.negative_outlier_factor_ = new Float64Array(lofs.map((v) => -v));
    const sorted = Array.from(lofs).sort((a, b) => b - a);
    const cutoff = Math.floor(n * (1 - this.contamination));
    this.threshold_ = -(sorted[cutoff] ?? sorted[sorted.length - 1] ?? 1.5);
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.novelty) throw new Error("Set novelty=true for predict on new data");
    return new Int32Array(X.map((xi) => this._score(xi) >= this.threshold_ ? 1 : -1));
  }

  decision_function(X: Float64Array[]): Float64Array {
    return new Float64Array(X.map((xi) => this._score(xi) - this.threshold_));
  }

  private _score(xi: Float64Array): number {
    const k = this.n_neighbors;
    const knn = this.X_.map((xj, j) => ({ j, d: this._dist(xi, xj) })).sort((a, b) => a.d - b.d).slice(0, k);
    const rdists = knn.map((nb) => {
      const trainKnn = this.X_.map((xj, j2) => ({ j: j2, d: this._dist(this.X_[nb.j]!, xj) })).sort((a, b) => a.d - b.d).slice(1, k + 1);
      const kDist = trainKnn[k - 1]?.d ?? nb.d;
      return Math.max(kDist, nb.d);
    });
    const lrdI = k / rdists.reduce((s, d) => s + d, 0);
    let s = 0;
    for (const nb of knn) {
      const trainKnn = this.X_.map((xj, j) => ({ j, d: this._dist(this.X_[nb.j]!, xj) })).sort((a, b) => a.d - b.d).slice(1, k + 1);
      const rdistsNb = trainKnn.map((nb2) => {
        const kd2 = this.X_.map((xj, j2) => ({ j: j2, d: this._dist(this.X_[nb2.j]!, xj) })).sort((a, b) => a.d - b.d).slice(1, k)[k - 1]?.d ?? 0;
        return Math.max(kd2, nb2.d);
      });
      const lrdNb = k / rdistsNb.reduce((sum, d) => sum + d, 1e-10);
      s += lrdNb / lrdI;
    }
    return -(s / k);
  }

  private _computeKNN(X: Float64Array[]): { indices: Int32Array[]; distances: Float64Array[] } {
    const n = X.length;
    const k = this.n_neighbors;
    const indices: Int32Array[] = [];
    const distances: Float64Array[] = [];
    for (let i = 0; i < n; i++) {
      const dists = X.map((xj, j) => ({ j, d: this._dist(X[i]!, xj) }));
      dists.sort((a, b) => a.d - b.d);
      const neighbors = dists.slice(1, k + 1);
      indices.push(new Int32Array(neighbors.map((x) => x.j)));
      distances.push(new Float64Array(neighbors.map((x) => x.d)));
    }
    return { indices, distances };
  }

  private _computeLRD(
    X: Float64Array[],
    knn: { indices: Int32Array[]; distances: Float64Array[] },
    k: number,
  ): Float64Array {
    const n = X.length;
    const lrd = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < k; j++) {
        const nb = knn.indices[i]?.[j] ?? 0;
        const kDist = knn.distances[nb]?.[k - 1] ?? knn.distances[i]?.[j] ?? 1;
        sum += Math.max(kDist, knn.distances[i]?.[j] ?? 0);
      }
      lrd[i] = k / Math.max(sum, 1e-10);
    }
    return lrd;
  }

  private _dist(a: Float64Array, b: Float64Array): number {
    let d = 0;
    for (let k = 0; k < a.length; k++) {
      const diff = (a[k] ?? 0) - (b[k] ?? 0);
      if (this.metric === "manhattan") d += Math.abs(diff);
      else d += diff * diff;
    }
    return this.metric === "manhattan" ? d : Math.sqrt(d);
  }
}

export interface KRadiusNeighborsClassifierParams {
  radius?: number;
  outlier_label?: number;
  metric?: "euclidean" | "manhattan";
}

/** RadiusNeighborsClassifier: classify based on neighbors within radius. */
export class RadiusNeighborsClassifierExt extends BaseEstimator {
  radius: number;
  outlier_label: number;
  metric: "euclidean" | "manhattan";
  X_: Float64Array[] = [];
  y_: Int32Array = new Int32Array(0);
  classes_: Int32Array = new Int32Array(0);

  constructor(params: KRadiusNeighborsClassifierParams = {}) {
    super();
    this.radius = params.radius ?? 1.0;
    this.outlier_label = params.outlier_label ?? -1;
    this.metric = params.metric ?? "euclidean";
  }

  fit(X: Float64Array[], y: Int32Array): this {
    this.X_ = X;
    this.y_ = y;
    this.classes_ = new Int32Array([...new Set(Array.from(y))].sort((a, b) => a - b));
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    return new Int32Array(X.map((xi) => {
      const neighbors: number[] = [];
      for (let i = 0; i < this.X_.length; i++) {
        if (this._dist(xi, this.X_[i]!) <= this.radius) neighbors.push(this.y_[i] ?? 0);
      }
      if (neighbors.length === 0) return this.outlier_label;
      const counts = new Map<number, number>();
      for (const c of neighbors) counts.set(c, (counts.get(c) ?? 0) + 1);
      let best = this.outlier_label, bestCnt = 0;
      for (const [c, cnt] of counts) if (cnt > bestCnt) { best = c; bestCnt = cnt; }
      return best;
    }));
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) correct++;
    return correct / y.length;
  }

  private _dist(a: Float64Array, b: Float64Array): number {
    let d = 0;
    for (let k = 0; k < a.length; k++) {
      const diff = (a[k] ?? 0) - (b[k] ?? 0);
      if (this.metric === "manhattan") d += Math.abs(diff);
      else d += diff * diff;
    }
    return this.metric === "manhattan" ? d : Math.sqrt(d);
  }
}

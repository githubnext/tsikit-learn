/**
 * Semi-supervised learning extensions: FlexConSemi, TSVM.
 * Mirrors sklearn.semi_supervised advanced methods.
 */

import { BaseEstimator } from "../base.js";

export interface TSVMParams {
  C?: number;
  Cl?: number;
  Cu?: number;
  kernel?: "rbf" | "linear";
  gamma?: number;
  max_iter?: number;
}

/** Transductive SVM (TSVM): semi-supervised SVM classifier. */
export class TSVM extends BaseEstimator {
  C: number;
  Cl: number;
  Cu: number;
  kernel: "rbf" | "linear";
  gamma: number;
  max_iter: number;
  X_: Float64Array[] = [];
  y_: Int32Array = new Int32Array(0);
  alpha_: Float64Array = new Float64Array(0);
  b_ = 0;

  constructor(params: TSVMParams = {}) {
    super();
    this.C = params.C ?? 1.0;
    this.Cl = params.Cl ?? 0.1;
    this.Cu = params.Cu ?? 0.001;
    this.kernel = params.kernel ?? "rbf";
    this.gamma = params.gamma ?? 0.1;
    this.max_iter = params.max_iter ?? 100;
  }

  private _k(a: Float64Array, b: Float64Array): number {
    if (this.kernel === "linear") {
      let s = 0; for (let k = 0; k < a.length; k++) s += (a[k] ?? 0) * (b[k] ?? 0); return s;
    }
    let d = 0; for (let k = 0; k < a.length; k++) d += ((a[k] ?? 0) - (b[k] ?? 0)) ** 2;
    return Math.exp(-this.gamma * d);
  }

  fit(X: Float64Array[], y: Int32Array, Xu?: Float64Array[]): this {
    const labeled = X;
    const unlabeled = Xu ?? [];
    const allX = [...labeled, ...unlabeled];
    const n = labeled.length, nu = unlabeled.length, N = allX.length;
    this.X_ = allX;
    // Initialize pseudo-labels for unlabeled data
    const allY = new Int32Array(N);
    for (let i = 0; i < n; i++) allY[i] = y[i] ?? 0;
    for (let i = 0; i < nu; i++) allY[n + i] = i % 2 === 0 ? 1 : -1;
    // Alternating optimization
    const alpha = new Float64Array(N).fill(0.1);
    for (let iter = 0; iter < this.max_iter; iter++) {
      // Update SVM on all data
      for (let i = 0; i < N; i++) {
        let fi = this.b_;
        for (let j = 0; j < N; j++) fi += (alpha[j] ?? 0) * (allY[j] ?? 0) * this._k(allX[j]!, allX[i]!);
        const C = i < n ? this.C : this.Cl;
        const newA = Math.max(0, Math.min(C, (alpha[i] ?? 0) + 0.01 * ((allY[i] ?? 0) * fi < 1 ? 1 : 0)));
        alpha[i] = newA;
      }
      // Re-assign pseudo-labels for unlabeled
      let bSum = 0, bCnt = 0;
      for (let i = 0; i < N; i++) {
        if ((alpha[i] ?? 0) > 0) { let f = 0; for (let j = 0; j < N; j++) f += (alpha[j] ?? 0) * (allY[j] ?? 0) * this._k(allX[j]!, allX[i]!); bSum += (allY[i] ?? 0) - f; bCnt++; }
      }
      this.b_ = bCnt > 0 ? bSum / bCnt : 0;
      for (let i = n; i < N; i++) {
        let fi = this.b_;
        for (let j = 0; j < N; j++) fi += (alpha[j] ?? 0) * (allY[j] ?? 0) * this._k(allX[j]!, allX[i]!);
        allY[i] = fi >= 0 ? 1 : -1;
      }
    }
    this.alpha_ = alpha;
    this.y_ = allY;
    return this;
  }

  decision_function(X: Float64Array[]): Float64Array {
    return new Float64Array(X.map((xi) => {
      let s = this.b_;
      for (let j = 0; j < this.X_.length; j++) s += (this.alpha_[j] ?? 0) * (this.y_[j] ?? 0) * this._k(this.X_[j]!, xi);
      return s;
    }));
  }

  predict(X: Float64Array[]): Int32Array {
    const df = this.decision_function(X);
    return new Int32Array(df.map((v) => v >= 0 ? 1 : -1));
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let c = 0;
    for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) c++;
    return c / y.length;
  }
}

export interface FlexConParams {
  base_estimator?: null;
  threshold?: number;
  max_iter?: number;
}

/** FlexCon: flexible confidence-based self-training. */
export class FlexCon extends BaseEstimator {
  threshold: number;
  max_iter: number;
  classes_: Int32Array = new Int32Array(0);
  X_: Float64Array[] = [];
  y_: Int32Array = new Int32Array(0);

  constructor(params: FlexConParams = {}) {
    super();
    this.threshold = params.threshold ?? 0.95;
    this.max_iter = params.max_iter ?? 10;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const classes = [...new Set(Array.from(y).filter((v) => v !== -1))].sort((a, b) => a - b);
    this.classes_ = new Int32Array(classes);
    // Start with labeled data
    let Xl = X.filter((_, i) => (y[i] ?? -1) !== -1);
    let yl = new Int32Array(Array.from(y).filter((v) => v !== -1));
    let Xu = X.filter((_, i) => (y[i] ?? -1) === -1);
    for (let iter = 0; iter < this.max_iter; iter++) {
      if (Xu.length === 0) break;
      // Simple kNN as base estimator
      const knn = this._knnPredict(Xl, yl, Xu);
      const confident: { x: Float64Array; c: number }[] = [];
      const unconfident: Float64Array[] = [];
      for (let i = 0; i < Xu.length; i++) {
        if (knn.proba[i]! >= this.threshold) confident.push({ x: Xu[i]!, c: knn.labels[i] ?? 0 });
        else unconfident.push(Xu[i]!);
      }
      if (confident.length === 0) break;
      Xl = [...Xl, ...confident.map((x) => x.x)];
      yl = new Int32Array([...Array.from(yl), ...confident.map((x) => x.c)]);
      Xu = unconfident;
    }
    this.X_ = Xl;
    this.y_ = yl;
    return this;
  }

  private _knnPredict(
    X: Float64Array[],
    y: Int32Array,
    Xtest: Float64Array[],
    k = 5,
  ): { labels: Int32Array; proba: Float64Array } {
    const labels = new Int32Array(Xtest.length);
    const proba = new Float64Array(Xtest.length);
    for (let i = 0; i < Xtest.length; i++) {
      const dists = X.map((xi, j) => {
        let d = 0;
        for (let f = 0; f < xi.length; f++) d += ((xi[f] ?? 0) - (Xtest[i]?.[f] ?? 0)) ** 2;
        return { j, d };
      }).sort((a, b) => a.d - b.d).slice(0, k);
      const votes = new Map<number, number>();
      for (const { j } of dists) votes.set(y[j] ?? 0, (votes.get(y[j] ?? 0) ?? 0) + 1);
      let best = 0, bestVotes = 0;
      for (const [c, v] of votes) if (v > bestVotes) { best = c; bestVotes = v; }
      labels[i] = best;
      proba[i] = bestVotes / k;
    }
    return { labels, proba };
  }

  predict(X: Float64Array[]): Int32Array {
    const { labels } = this._knnPredict(this.X_, this.y_, X);
    return labels;
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let c = 0;
    for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) c++;
    return c / y.length;
  }
}

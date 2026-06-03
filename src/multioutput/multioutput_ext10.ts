/**
 * Multioutput extensions: Classifier chain variants, multi-target regression.
 * Mirrors sklearn.multioutput additional methods.
 */

import { BaseEstimator } from "../base.js";

type FitPredict = {
  fit(X: Float64Array[], y: Int32Array): void;
  predict(X: Float64Array[]): Int32Array;
};

type FitPredictReg = {
  fit(X: Float64Array[], y: Float64Array): void;
  predict(X: Float64Array[]): Float64Array;
};

/** Monte Carlo classifier chain with random ordering. */
export class MonteCarloClassifierChain extends BaseEstimator {
  n_chains: number;
  estimatorFactory: () => FitPredict;
  chains_: Array<{ order: Int32Array; estimators: FitPredict[] }> = [];

  constructor(estimatorFactory: () => FitPredict, params: { n_chains?: number } = {}) {
    super();
    this.estimatorFactory = estimatorFactory;
    this.n_chains = params.n_chains ?? 10;
  }

  fit(X: Float64Array[], Y: Int32Array[]): this {
    const k = Y[0]?.length ?? 0;
    for (let c = 0; c < this.n_chains; c++) {
      const order = new Int32Array(k).map((_, i) => i).sort(() => Math.random() - 0.5);
      const estimators: FitPredict[] = [];
      let augX = X.map(row => new Float64Array(row));
      for (let oi = 0; oi < k; oi++) {
        const j = order[oi]!;
        const yj = new Int32Array(Y.map(row => row[j] ?? 0));
        const est = this.estimatorFactory();
        est.fit(augX, yj);
        estimators.push(est);
        const preds = est.predict(augX);
        augX = augX.map((row, i) => {
          const newRow = new Float64Array(row.length + 1);
          for (let f = 0; f < row.length; f++) newRow[f] = row[f] ?? 0;
          newRow[row.length] = preds[i] ?? 0;
          return newRow;
        });
      }
      this.chains_.push({ order, estimators });
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array[] {
    const n = X.length;
    const k = this.chains_[0]?.order.length ?? 0;
    const votes: Array<Float64Array> = Array.from({ length: n }, () => new Float64Array(k));

    for (const chain of this.chains_) {
      let augX = X.map(row => new Float64Array(row));
      const preds: number[][] = Array.from({ length: n }, () => new Array(k).fill(0) as number[]);
      for (let oi = 0; oi < k; oi++) {
        const j = chain.order[oi]!;
        const p = chain.estimators[oi]!.predict(augX);
        for (let i = 0; i < n; i++) preds[i]![j] = p[i] ?? 0;
        augX = augX.map((row, i) => {
          const newRow = new Float64Array(row.length + 1);
          for (let f = 0; f < row.length; f++) newRow[f] = row[f] ?? 0;
          newRow[row.length] = p[i] ?? 0;
          return newRow;
        });
      }
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < k; j++) {
          if ((preds[i]?.[j] ?? 0) === 1) votes[i]![j] = (votes[i]?.[j] ?? 0) + 1;
        }
      }
    }

    return Array.from({ length: n }, (_, i) =>
      new Int32Array(k).map((_, j) => ((votes[i]?.[j] ?? 0) > this.n_chains / 2 ? 1 : 0)),
    );
  }
}

/** Multi-target Gaussian process regression. */
export class MultiTargetGP extends BaseEstimator {
  estimators_: Array<FitPredictReg> = [];
  estimatorFactory: () => FitPredictReg;
  n_targets_ = 0;

  constructor(estimatorFactory: () => FitPredictReg) {
    super();
    this.estimatorFactory = estimatorFactory;
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const k = Y[0]?.length ?? 0;
    this.n_targets_ = k;
    this.estimators_ = [];
    for (let j = 0; j < k; j++) {
      const yj = new Float64Array(Y.map(row => row[j] ?? 0));
      const est = this.estimatorFactory();
      est.fit(X, yj);
      this.estimators_.push(est);
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const k = this.n_targets_;
    const out: Float64Array[] = Array.from({ length: n }, () => new Float64Array(k));
    for (let j = 0; j < k; j++) {
      const preds = this.estimators_[j]!.predict(X);
      for (let i = 0; i < n; i++) out[i]![j] = preds[i] ?? 0;
    }
    return out;
  }
}

/** Structured output prediction with Hamming loss. */
export function multilabelHammingLoss(
  y_true: Int32Array[],
  y_pred: Int32Array[],
): number {
  const n = y_true.length;
  const k = y_true[0]?.length ?? 0;
  let errors = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < k; j++) {
      if ((y_true[i]?.[j] ?? 0) !== (y_pred[i]?.[j] ?? 0)) errors++;
    }
  }
  return errors / (n * k || 1);
}

/** Subset accuracy for multi-label predictions. */
export function subsetAccuracy(y_true: Int32Array[], y_pred: Int32Array[]): number {
  const n = y_true.length;
  let correct = 0;
  for (let i = 0; i < n; i++) {
    const k = y_true[i]?.length ?? 0;
    let match = true;
    for (let j = 0; j < k; j++) {
      if ((y_true[i]?.[j] ?? 0) !== (y_pred[i]?.[j] ?? 0)) { match = false; break; }
    }
    if (match) correct++;
  }
  return correct / (n || 1);
}

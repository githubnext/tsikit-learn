/**
 * Naive Bayes extensions: OnlineNB, NegationNaiveBayes.
 * Mirrors sklearn.naive_bayes advanced classifiers.
 */

import { BaseEstimator } from "../base.js";

export interface OnlineNaiveBayesParams {
  alpha?: number;
  fit_prior?: boolean;
}

/** Online Multinomial Naive Bayes with partial_fit. */
export class OnlineMultinomialNB extends BaseEstimator {
  alpha: number;
  fit_prior: boolean;
  class_log_prior_: Float64Array = new Float64Array(0);
  feature_log_prob_: Float64Array[] = [];
  classes_: Int32Array = new Int32Array(0);
  class_count_: Float64Array = new Float64Array(0);
  feature_count_: Float64Array[] = [];
  n_features_in_ = 0;

  constructor(params: OnlineNaiveBayesParams = {}) {
    super();
    this.alpha = params.alpha ?? 1.0;
    this.fit_prior = params.fit_prior ?? true;
  }

  partial_fit(X: Float64Array[], y: Int32Array, classes?: Int32Array): this {
    const allClasses = classes ? Array.from(classes) : [...new Set(Array.from(y))].sort((a, b) => a - b);
    if (this.class_count_.length === 0) {
      this.classes_ = new Int32Array(allClasses);
      const k = allClasses.length;
      this.n_features_in_ = X[0]?.length ?? 0;
      this.class_count_ = new Float64Array(k);
      this.feature_count_ = Array.from({ length: k }, () => new Float64Array(this.n_features_in_));
    }
    const classIdx = new Map(Array.from(this.classes_).map((c, i) => [c, i]));
    for (let i = 0; i < y.length; i++) {
      const ci = classIdx.get(y[i] ?? 0) ?? 0;
      this.class_count_[ci] = (this.class_count_[ci] ?? 0) + 1;
      for (let k = 0; k < this.n_features_in_; k++) {
        this.feature_count_[ci]![k] = (this.feature_count_[ci]![k] ?? 0) + (X[i]?.[k] ?? 0);
      }
    }
    this._updateLogProb();
    return this;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    this.class_count_ = new Float64Array(0);
    return this.partial_fit(X, y);
  }

  private _updateLogProb(): void {
    const k = this.classes_.length;
    const totalCount = Array.from(this.class_count_).reduce((s, v) => s + v, 0);
    this.class_log_prior_ = new Float64Array(k).map((_, i) =>
      this.fit_prior ? Math.log(Math.max(this.class_count_[i] ?? 0, 1e-10) / Math.max(totalCount, 1)) : -Math.log(k),
    );
    this.feature_log_prob_ = this.feature_count_.map((fc) => {
      const total = Array.from(fc).reduce((s, v) => s + v, 0) + this.alpha * this.n_features_in_;
      return new Float64Array(fc.map((v) => Math.log((v + this.alpha) / total)));
    });
  }

  predict_log_proba(X: Float64Array[]): Float64Array[] {
    return X.map((xi) =>
      new Float64Array(this.classes_.length).map((_, c) => {
        let logp = this.class_log_prior_[c] ?? 0;
        const flp = this.feature_log_prob_[c];
        if (flp) for (let k = 0; k < xi.length; k++) logp += (xi[k] ?? 0) * (flp[k] ?? 0);
        return logp;
      }),
    );
  }

  predict(X: Float64Array[]): Int32Array {
    const logProba = this.predict_log_proba(X);
    return new Int32Array(logProba.map((row) => {
      let best = 0, bestV = -Number.POSITIVE_INFINITY;
      for (let i = 0; i < row.length; i++) if ((row[i] ?? -Infinity) > bestV) { best = this.classes_[i] ?? 0; bestV = row[i] ?? -Infinity; }
      return best;
    }));
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let c = 0;
    for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) c++;
    return c / y.length;
  }
}

export interface NegationNBParams {
  alpha?: number;
}

/** Negation Naive Bayes: uses negated class counts for better estimation. */
export class NegationNB extends BaseEstimator {
  alpha: number;
  classes_: Int32Array = new Int32Array(0);
  class_count_: Float64Array = new Float64Array(0);
  feature_count_: Float64Array[] = [];
  n_features_in_ = 0;

  constructor(params: NegationNBParams = {}) {
    super();
    this.alpha = params.alpha ?? 1.0;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const classes = [...new Set(Array.from(y))].sort((a, b) => a - b);
    const k = classes.length;
    const nf = X[0]?.length ?? 0;
    this.classes_ = new Int32Array(classes);
    this.n_features_in_ = nf;
    this.class_count_ = new Float64Array(k);
    this.feature_count_ = Array.from({ length: k }, () => new Float64Array(nf));
    const classIdx = new Map(classes.map((c, i) => [c, i]));
    for (let i = 0; i < y.length; i++) {
      const ci = classIdx.get(y[i] ?? 0) ?? 0;
      this.class_count_[ci] = (this.class_count_[ci] ?? 0) + 1;
      for (let f = 0; f < nf; f++) this.feature_count_[ci]![f] = (this.feature_count_[ci]![f] ?? 0) + (X[i]?.[f] ?? 0);
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    const k = this.classes_.length;
    const nf = this.n_features_in_;
    // Negation: score = log P(x|class) - log P(x|NOT class)
    const totalFeat = new Float64Array(nf);
    for (let c = 0; c < k; c++) for (let f = 0; f < nf; f++) totalFeat[f] = (totalFeat[f] ?? 0) + (this.feature_count_[c]?.[f] ?? 0);
    return new Int32Array(X.map((xi) => {
      let best = 0, bestScore = -Number.POSITIVE_INFINITY;
      for (let c = 0; c < k; c++) {
        let score = Math.log(Math.max(this.class_count_[c] ?? 0, 1));
        const fc = this.feature_count_[c];
        const notTotal = Array.from(totalFeat).reduce((s, v) => s + v, 0) - Array.from(fc ?? []).reduce((s, v) => s + v, 0) + this.alpha * nf;
        for (let f = 0; f < nf; f++) {
          const notFc = (totalFeat[f] ?? 0) - (fc?.[f] ?? 0) + this.alpha;
          score += (xi[f] ?? 0) * Math.log(Math.max(notFc / notTotal, 1e-10));
        }
        if (score > bestScore) { best = this.classes_[c] ?? 0; bestScore = score; }
      }
      return best;
    }));
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let c = 0;
    for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) c++;
    return c / y.length;
  }
}

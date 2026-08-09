/**
 * Naive Bayes extensions: Complement NB, Out-of-core NB, Mixed NB.
 * Mirrors sklearn.naive_bayes additional methods.
 */

import { BaseEstimator } from "../base.js";

/** Complement Naive Bayes: uses complement of each class for estimation. */
export class ComplementNB extends BaseEstimator {
  alpha: number;
  norm: boolean;
  class_log_prior_: Float64Array = new Float64Array(0);
  feature_log_prob_: Float64Array[] = [];
  classes_: Int32Array = new Int32Array(0);

  constructor(params: { alpha?: number; norm?: boolean } = {}) {
    super();
    this.alpha = params.alpha ?? 1.0;
    this.norm = params.norm ?? false;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    const classes = [...new Set(Array.from(y))].sort((a, b) => a - b);
    this.classes_ = new Int32Array(classes);
    const k = classes.length;

    const classCounts = new Float64Array(k);
    const featureCounts: Float64Array[] = Array.from({ length: k }, () => new Float64Array(d));

    for (let i = 0; i < n; i++) {
      const ci = classes.indexOf(y[i] ?? 0);
      if (ci < 0) continue;
      classCounts[ci] = (classCounts[ci] ?? 0) + 1;
      for (let f = 0; f < d; f++) featureCounts[ci]![f] = (featureCounts[ci]?.[f] ?? 0) + (X[i]?.[f] ?? 0);
    }

    this.class_log_prior_ = classCounts.map(c => Math.log(c / n || 1e-10));

    // Complement: sum of all classes except current
    this.feature_log_prob_ = Array.from({ length: k }, (_, ci) => {
      const complement = new Float64Array(d);
      let totalComplement = 0;
      for (let cj = 0; cj < k; cj++) {
        if (cj === ci) continue;
        for (let f = 0; f < d; f++) complement[f] = (complement[f] ?? 0) + (featureCounts[cj]?.[f] ?? 0) + this.alpha;
        totalComplement += (classCounts[cj] ?? 0);
      }
      const totalSmooth = totalComplement + d * this.alpha;
      const logProb = complement.map(v => -Math.log(v / totalSmooth || 1e-10));
      if (this.norm) {
        const normFactor = logProb.reduce((s, v) => s + v * v, 0);
        return logProb.map(v => v / (Math.sqrt(normFactor) || 1));
      }
      return logProb;
    });
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    return new Int32Array(X.map(row => {
      const scores = this.feature_log_prob_.map((logP, ci) => {
        let s = this.class_log_prior_[ci] ?? 0;
        for (let f = 0; f < row.length; f++) s += (row[f] ?? 0) * (logP[f] ?? 0);
        return s;
      });
      let best = 0;
      for (let i = 1; i < scores.length; i++) if ((scores[i] ?? -Infinity) > (scores[best] ?? -Infinity)) best = i;
      return this.classes_[best] ?? 0;
    }));
  }
}

/** Mixed Naive Bayes: combines Gaussian and Multinomial NB for heterogeneous features. */
export class MixedNB extends BaseEstimator {
  continuous_cols: number[];
  discrete_cols: number[];
  alpha: number;
  gaussian_mean_: Float64Array[] = [];
  gaussian_var_: Float64Array[] = [];
  multinomial_log_prob_: Float64Array[][] = [];
  class_log_prior_: Float64Array = new Float64Array(0);
  classes_: Int32Array = new Int32Array(0);

  constructor(params: {
    continuous_cols?: number[];
    discrete_cols?: number[];
    alpha?: number;
  } = {}) {
    super();
    this.continuous_cols = params.continuous_cols ?? [];
    this.discrete_cols = params.discrete_cols ?? [];
    this.alpha = params.alpha ?? 1.0;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const classes = [...new Set(Array.from(y))].sort((a, b) => a - b);
    this.classes_ = new Int32Array(classes);
    const k = classes.length;
    const nc = this.continuous_cols.length;
    const nd = this.discrete_cols.length;

    const classCounts = new Float64Array(k);
    const gaussSums: Float64Array[] = Array.from({ length: k }, () => new Float64Array(nc));
    const gaussSumSq: Float64Array[] = Array.from({ length: k }, () => new Float64Array(nc));
    const discountCounts: Float64Array[] = Array.from({ length: k }, () => new Float64Array(nd));

    for (let i = 0; i < n; i++) {
      const ci = classes.indexOf(y[i] ?? 0);
      if (ci < 0) continue;
      classCounts[ci] = (classCounts[ci] ?? 0) + 1;
      for (let fi = 0; fi < nc; fi++) {
        const v = X[i]?.[this.continuous_cols[fi] ?? 0] ?? 0;
        gaussSums[ci]![fi] = (gaussSums[ci]?.[fi] ?? 0) + v;
        gaussSumSq[ci]![fi] = (gaussSumSq[ci]?.[fi] ?? 0) + v * v;
      }
      for (let fi = 0; fi < nd; fi++) {
        discountCounts[ci]![fi] = (discountCounts[ci]?.[fi] ?? 0) + (X[i]?.[this.discrete_cols[fi] ?? 0] ?? 0);
      }
    }

    this.class_log_prior_ = classCounts.map(c => Math.log(c / n || 1e-10));
    this.gaussian_mean_ = gaussSums.map((s, ci) => s.map((v, fi) => v / (classCounts[ci] ?? 1)));
    this.gaussian_var_ = gaussSumSq.map((sq, ci) =>
      sq.map((v, fi) => Math.max(1e-9, v / (classCounts[ci] ?? 1) - (this.gaussian_mean_[ci]?.[fi] ?? 0) ** 2)),
    );
    this.multinomial_log_prob_ = discountCounts.map((cnt, ci) => {
      const total = cnt.reduce((s, v) => s + v, 0) + nd * this.alpha;
      return [cnt.map(v => Math.log((v + this.alpha) / total || 1e-10))];
    });
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    const k = this.classes_.length;
    const nc = this.continuous_cols.length;
    const nd = this.discrete_cols.length;
    return new Int32Array(X.map(row => {
      let best = 0;
      let bestScore = -Infinity;
      for (let ci = 0; ci < k; ci++) {
        let s = this.class_log_prior_[ci] ?? 0;
        for (let fi = 0; fi < nc; fi++) {
          const v = row[this.continuous_cols[fi] ?? 0] ?? 0;
          const mean = this.gaussian_mean_[ci]?.[fi] ?? 0;
          const variance = this.gaussian_var_[ci]?.[fi] ?? 1;
          s += -0.5 * Math.log(2 * Math.PI * variance) - (v - mean) ** 2 / (2 * variance);
        }
        for (let fi = 0; fi < nd; fi++) {
          s += (row[this.discrete_cols[fi] ?? 0] ?? 0) * (this.multinomial_log_prob_[ci]?.[0]?.[fi] ?? 0);
        }
        if (s > bestScore) { bestScore = s; best = ci; }
      }
      return this.classes_[best] ?? 0;
    }));
  }
}

/** Bernoulli Naive Bayes variant with calibrated priors. */
export class CalibratedBernoulliNB extends BaseEstimator {
  alpha: number;
  calibration_alpha: number;
  class_log_prior_: Float64Array = new Float64Array(0);
  feature_log_prob_: Float64Array[] = [];
  classes_: Int32Array = new Int32Array(0);

  constructor(params: { alpha?: number; calibration_alpha?: number } = {}) {
    super();
    this.alpha = params.alpha ?? 1.0;
    this.calibration_alpha = params.calibration_alpha ?? 0.5;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    const classes = [...new Set(Array.from(y))].sort((a, b) => a - b);
    this.classes_ = new Int32Array(classes);
    const k = classes.length;
    const classCounts = new Float64Array(k);
    const featureCounts: Float64Array[] = Array.from({ length: k }, () => new Float64Array(d));

    for (let i = 0; i < n; i++) {
      const ci = classes.indexOf(y[i] ?? 0);
      if (ci < 0) continue;
      classCounts[ci] = (classCounts[ci] ?? 0) + 1;
      for (let f = 0; f < d; f++) {
        if ((X[i]?.[f] ?? 0) > 0) featureCounts[ci]![f] = (featureCounts[ci]?.[f] ?? 0) + 1;
      }
    }

    this.class_log_prior_ = classCounts.map(c => Math.log((c + this.calibration_alpha) / (n + k * this.calibration_alpha) || 1e-10));
    this.feature_log_prob_ = featureCounts.map((cnt, ci) => {
      const nc = classCounts[ci] ?? 0;
      return cnt.map(v => Math.log((v + this.alpha) / (nc + 2 * this.alpha) || 1e-10));
    });
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    return new Int32Array(X.map(row => {
      const scores = this.feature_log_prob_.map((logP, ci) => {
        let s = this.class_log_prior_[ci] ?? 0;
        for (let f = 0; f < row.length; f++) {
          if ((row[f] ?? 0) > 0) s += logP[f] ?? 0;
          else s += Math.log(1 - Math.exp(logP[f] ?? 0) || 1e-10);
        }
        return s;
      });
      let best = 0;
      for (let i = 1; i < scores.length; i++) if ((scores[i] ?? -Infinity) > (scores[best] ?? -Infinity)) best = i;
      return this.classes_[best] ?? 0;
    }));
  }
}

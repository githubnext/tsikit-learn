/**
 * Multioutput extensions: MultiOutputProbabilistic, MultiTaskLasso, IndependentMultiOutput
 * Port of sklearn.multioutput extensions
 */

import { NotFittedError } from "../exceptions.js";

export interface RegressionEstimator {
  coef_?: Float64Array | null;
  intercept_?: number;
  fit(X: Float64Array[], y: Float64Array): this;
  predict(X: Float64Array[]): Float64Array;
  score?(X: Float64Array[], y: Float64Array): number;
}

export interface ClassifierEstimator {
  fit(X: Float64Array[], y: Int32Array): this;
  predict(X: Float64Array[]): Int32Array;
  predictProba?(X: Float64Array[]): Float64Array[];
}

export class MultiOutputRegressorExt {
  estimatorFactory: () => RegressionEstimator;
  nJobs: number;

  private estimators_: RegressionEstimator[] | null = null;
  private nOutputs_ = 0;

  constructor(opts: {
    estimatorFactory?: () => RegressionEstimator;
    nJobs?: number;
  } = {}) {
    this.nJobs = opts.nJobs ?? 1;
    this.estimatorFactory = opts.estimatorFactory ?? (() => ({
      coef_: null as Float64Array | null,
      intercept_: 0,
      fit(X: Float64Array[], y: Float64Array) {
        const n = X.length;
        const p = X[0]?.length ?? 0;
        let sx = new Float64Array(p);
        let sy = 0;
        for (let i = 0; i < n; i++) {
          sy += y[i] ?? 0;
          for (let j = 0; j < p; j++) sx[j] = (sx[j] ?? 0) + (X[i]![j] ?? 0);
        }
        for (let j = 0; j < p; j++) sx[j] = (sx[j] ?? 0) / n;
        sy /= n;
        const coef = new Float64Array(p);
        let denom = 0;
        for (let j = 0; j < p; j++) {
          let num = 0;
          for (let i = 0; i < n; i++) num += ((X[i]![j] ?? 0) - (sx[j] ?? 0)) * ((y[i] ?? 0) - sy);
          for (let i = 0; i < n; i++) denom += ((X[i]![j] ?? 0) - (sx[j] ?? 0)) ** 2;
          coef[j] = num / (denom + 1e-15);
        }
        this.coef_ = coef;
        this.intercept_ = sy - sx.reduce((s, v, j) => s + (v ?? 0) * (coef[j] ?? 0), 0);
        return this;
      },
      predict(X: Float64Array[]) {
        return Float64Array.from(X.map(xi => {
          let val = this.intercept_;
          for (let j = 0; j < xi.length; j++) val += (xi[j] ?? 0) * (this.coef_![j] ?? 0);
          return val;
        }));
      },
    }));
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    this.nOutputs_ = Y[0]?.length ?? 0;
    this.estimators_ = Array.from({ length: this.nOutputs_ }, (_, k) => {
      const yk = Float64Array.from(Y.map(yi => yi[k] ?? 0));
      return this.estimatorFactory().fit(X, yk);
    });
    return this;
  }

  predict(X: Float64Array[]): Float64Array[] {
    if (!this.estimators_) throw new NotFittedError("MultiOutputRegressorExt not fitted.");
    const preds = this.estimators_.map(est => est.predict(X));
    return X.map((_, i) => Float64Array.from({ length: this.nOutputs_ }, (__, k) => preds[k]![i] ?? 0));
  }

  score(X: Float64Array[], Y: Float64Array[]): number {
    const preds = this.predict(X);
    let totalR2 = 0;
    for (let k = 0; k < this.nOutputs_; k++) {
      const yk = Float64Array.from(Y.map(yi => yi[k] ?? 0));
      const ykPred = Float64Array.from(preds.map(yi => yi[k] ?? 0));
      const mean = yk.reduce((a, b) => a + b, 0) / yk.length;
      let ss_res = 0;
      let ss_tot = 0;
      for (let i = 0; i < yk.length; i++) {
        ss_res += ((yk[i] ?? 0) - (ykPred[i] ?? 0)) ** 2;
        ss_tot += ((yk[i] ?? 0) - mean) ** 2;
      }
      totalR2 += 1 - ss_res / (ss_tot + 1e-15);
    }
    return totalR2 / this.nOutputs_;
  }
}

export class ClassifierChainExt {
  classifiers: ClassifierEstimator[];
  order: number[] | null;

  private fitted_ = false;
  private nClasses_: number[] = [];

  constructor(opts: {
    classifiers?: ClassifierEstimator[];
    order?: number[];
  } = {}) {
    this.classifiers = opts.classifiers ?? [];
    this.order = opts.order ?? null;
  }

  fit(X: Float64Array[], Y: Int32Array[]): this {
    const nOutputs = Y[0]?.length ?? 0;
    const order = this.order ?? Array.from({ length: nOutputs }, (_, i) => i);
    this.nClasses_ = Array(nOutputs).fill(2);
    let augX = X.map(xi => xi.slice());
    for (const k of order) {
      const yk = Int32Array.from(Y.map(yi => yi[k] ?? 0));
      this.classifiers[k]?.fit(augX, yk);
      const preds = this.classifiers[k]?.predict(augX);
      augX = augX.map((xi, i) => {
        const r = new Float64Array(xi.length + 1);
        for (let j = 0; j < xi.length; j++) r[j] = xi[j] ?? 0;
        r[xi.length] = preds?.[i] ?? 0;
        return r;
      });
    }
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Int32Array[] {
    if (!this.fitted_) throw new NotFittedError("ClassifierChainExt not fitted.");
    const nOutputs = this.classifiers.length;
    const order = this.order ?? Array.from({ length: nOutputs }, (_, i) => i);
    const preds: Int32Array[] = Array.from({ length: nOutputs }, () => new Int32Array(X.length));
    let augX = X.map(xi => xi.slice());
    for (const k of order) {
      const kPreds = this.classifiers[k]?.predict(augX);
      if (kPreds) for (let i = 0; i < X.length; i++) preds[k]![i] = kPreds[i] ?? 0;
      augX = augX.map((xi, i) => {
        const r = new Float64Array(xi.length + 1);
        for (let j = 0; j < xi.length; j++) r[j] = xi[j] ?? 0;
        r[xi.length] = kPreds?.[i] ?? 0;
        return r;
      });
    }
    return X.map((_, i) => Int32Array.from({ length: nOutputs }, (__, k) => preds[k]![i] ?? 0));
  }
}

export class RegressorChainExt {
  regressors: RegressionEstimator[];
  order: number[] | null;
  private fitted_ = false;

  constructor(opts: {
    regressors?: RegressionEstimator[];
    order?: number[];
  } = {}) {
    this.regressors = opts.regressors ?? [];
    this.order = opts.order ?? null;
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const nOutputs = Y[0]?.length ?? 0;
    const order = this.order ?? Array.from({ length: nOutputs }, (_, i) => i);
    let augX = X.map(xi => xi.slice());
    for (const k of order) {
      const yk = Float64Array.from(Y.map(yi => yi[k] ?? 0));
      this.regressors[k]?.fit(augX, yk);
      const predsK = this.regressors[k]?.predict(augX);
      augX = augX.map((xi, i) => {
        const r = new Float64Array(xi.length + 1);
        for (let j = 0; j < xi.length; j++) r[j] = xi[j] ?? 0;
        r[xi.length] = predsK?.[i] ?? 0;
        return r;
      });
    }
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new NotFittedError("RegressorChainExt not fitted.");
    const nOutputs = this.regressors.length;
    const order = this.order ?? Array.from({ length: nOutputs }, (_, i) => i);
    const preds: Float64Array[] = Array.from({ length: nOutputs }, () => new Float64Array(X.length));
    let augX = X.map(xi => xi.slice());
    for (const k of order) {
      const kPreds = this.regressors[k]?.predict(augX);
      if (kPreds) for (let i = 0; i < X.length; i++) preds[k]![i] = kPreds[i] ?? 0;
      augX = augX.map((xi, i) => {
        const r = new Float64Array(xi.length + 1);
        for (let j = 0; j < xi.length; j++) r[j] = xi[j] ?? 0;
        r[xi.length] = kPreds?.[i] ?? 0;
        return r;
      });
    }
    return X.map((_, i) => Float64Array.from({ length: nOutputs }, (__, k) => preds[k]![i] ?? 0));
  }
}

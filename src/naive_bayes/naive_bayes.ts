/**
 * Naive Bayes classifiers.
 * Mirrors sklearn.naive_bayes: GaussianNB, MultinomialNB, BernoulliNB.
 */

import { NotFittedError } from "../exceptions.js";

export class GaussianNB {
  varSmoothing: number;

  classPrior_: Float64Array | null = null;
  thetaMean_: Float64Array[] | null = null;
  thetaVar_: Float64Array[] | null = null;
  classes_: Float64Array | null = null;

  constructor(options: { varSmoothing?: number } = {}) {
    this.varSmoothing = options.varSmoothing ?? 1e-9;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const uniqueClasses = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
    this.classes_ = new Float64Array(uniqueClasses);
    const nClasses = uniqueClasses.length;
    const classToIdx = new Map(uniqueClasses.map((c, i) => [c, i]));

    const means: Float64Array[] = Array.from({ length: nClasses }, () => new Float64Array(p));
    const vars: Float64Array[] = Array.from({ length: nClasses }, () => new Float64Array(p));
    const counts = new Int32Array(nClasses);

    for (let i = 0; i < n; i++) {
      const c = classToIdx.get(y[i] ?? 0) ?? 0;
      counts[c] = (counts[c] ?? 0) + 1;
      const xi = X[i] ?? new Float64Array(p);
      const mean = means[c] ?? new Float64Array(p);
      for (let j = 0; j < p; j++) {
        mean[j] = (mean[j] ?? 0) + (xi[j] ?? 0);
      }
    }

    for (let c = 0; c < nClasses; c++) {
      const cnt = counts[c] ?? 1;
      const mean = means[c] ?? new Float64Array(p);
      for (let j = 0; j < p; j++) {
        mean[j] = (mean[j] ?? 0) / cnt;
      }
    }

    // Compute variance
    for (let i = 0; i < n; i++) {
      const c = classToIdx.get(y[i] ?? 0) ?? 0;
      const xi = X[i] ?? new Float64Array(p);
      const mean = means[c] ?? new Float64Array(p);
      const variance = vars[c] ?? new Float64Array(p);
      for (let j = 0; j < p; j++) {
        variance[j] = (variance[j] ?? 0) + ((xi[j] ?? 0) - (mean[j] ?? 0)) ** 2;
      }
    }

    for (let c = 0; c < nClasses; c++) {
      const cnt = counts[c] ?? 1;
      const variance = vars[c] ?? new Float64Array(p);
      for (let j = 0; j < p; j++) {
        variance[j] = (variance[j] ?? 0) / cnt + this.varSmoothing;
      }
    }

    this.thetaMean_ = means;
    this.thetaVar_ = vars;
    this.classPrior_ = new Float64Array(nClasses);
    for (let c = 0; c < nClasses; c++) {
      this.classPrior_[c] = (counts[c] ?? 0) / n;
    }

    return this;
  }

  predictLogProba(X: Float64Array[]): Float64Array[] {
    if (this.classes_ === null) throw new NotFittedError("GaussianNB");
    const nClasses = this.classes_.length;
    const p = (X[0] ?? new Float64Array(0)).length;

    return X.map((xi) => {
      const logProba = new Float64Array(nClasses);
      for (let c = 0; c < nClasses; c++) {
        let logP = Math.log((this.classPrior_ as Float64Array)[c] ?? 1e-10);
        const mean = (this.thetaMean_ as Float64Array[])[c] ?? new Float64Array(p);
        const variance = (this.thetaVar_ as Float64Array[])[c] ?? new Float64Array(p);
        for (let j = 0; j < p; j++) {
          const xij = xi[j] ?? 0;
          const mu = mean[j] ?? 0;
          const sig2 = variance[j] ?? 1e-9;
          logP -= 0.5 * Math.log(2 * Math.PI * sig2);
          logP -= ((xij - mu) ** 2) / (2 * sig2);
        }
        logProba[c] = logP;
      }
      return logProba;
    });
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.classes_ === null) throw new NotFittedError("GaussianNB");
    const classes = this.classes_;
    const logProba = this.predictLogProba(X);
    return new Float64Array(
      logProba.map((lp) => {
        let maxIdx = 0;
        let maxVal = lp[0] ?? Number.NEGATIVE_INFINITY;
        for (let c = 1; c < lp.length; c++) {
          if ((lp[c] ?? Number.NEGATIVE_INFINITY) > maxVal) {
            maxVal = lp[c] ?? Number.NEGATIVE_INFINITY;
            maxIdx = c;
          }
        }
        return classes[maxIdx] ?? 0;
      }),
    );
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      if (pred[i] === y[i]) correct++;
    }
    return correct / y.length;
  }
}

export class MultinomialNB {
  alpha: number;

  featureLogProb_: Float64Array[] | null = null;
  classLogPrior_: Float64Array | null = null;
  classes_: Float64Array | null = null;

  constructor(options: { alpha?: number } = {}) {
    this.alpha = options.alpha ?? 1.0;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const uniqueClasses = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
    this.classes_ = new Float64Array(uniqueClasses);
    const nClasses = uniqueClasses.length;
    const classToIdx = new Map(uniqueClasses.map((c, i) => [c, i]));

    const counts: Float64Array[] = Array.from({ length: nClasses }, () => new Float64Array(p));
    const classCounts = new Float64Array(nClasses);

    for (let i = 0; i < n; i++) {
      const c = classToIdx.get(y[i] ?? 0) ?? 0;
      classCounts[c] = (classCounts[c] ?? 0) + 1;
      const xi = X[i] ?? new Float64Array(p);
      const count = counts[c] ?? new Float64Array(p);
      for (let j = 0; j < p; j++) {
        count[j] = (count[j] ?? 0) + (xi[j] ?? 0);
      }
    }

    this.classLogPrior_ = new Float64Array(
      Array.from(classCounts).map((c) => Math.log(c / n)),
    );

    this.featureLogProb_ = counts.map((count) => {
      const total = Array.from(count).reduce((a, b) => a + b, 0) + this.alpha * p;
      return new Float64Array(count.map((c) => Math.log((c + this.alpha) / total)));
    });

    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.classes_ === null) throw new NotFittedError("MultinomialNB");
    const classes = this.classes_;
    const nClasses = classes.length;
    const p = (X[0] ?? new Float64Array(0)).length;

    return new Float64Array(
      X.map((xi) => {
        let maxIdx = 0;
        let maxScore = Number.NEGATIVE_INFINITY;
        for (let c = 0; c < nClasses; c++) {
          let score = (this.classLogPrior_ as Float64Array)[c] ?? 0;
          const flp = (this.featureLogProb_ as Float64Array[])[c] ?? new Float64Array(p);
          for (let j = 0; j < p; j++) {
            score += (xi[j] ?? 0) * (flp[j] ?? 0);
          }
          if (score > maxScore) {
            maxScore = score;
            maxIdx = c;
          }
        }
        return classes[maxIdx] ?? 0;
      }),
    );
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      if (pred[i] === y[i]) correct++;
    }
    return correct / y.length;
  }
}

export class BernoulliNB {
  alpha: number;
  binarize: number | null;

  featureLogProb_: Float64Array[] | null = null;
  featureLogNegProb_: Float64Array[] | null = null;
  classLogPrior_: Float64Array | null = null;
  classes_: Float64Array | null = null;

  constructor(options: { alpha?: number; binarize?: number | null } = {}) {
    this.alpha = options.alpha ?? 1.0;
    this.binarize = options.binarize ?? 0.0;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const threshold = this.binarize ?? 0.0;
    const uniqueClasses = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
    this.classes_ = new Float64Array(uniqueClasses);
    const nClasses = uniqueClasses.length;
    const classToIdx = new Map(uniqueClasses.map((c, i) => [c, i]));

    const counts: Float64Array[] = Array.from({ length: nClasses }, () => new Float64Array(p));
    const classCounts = new Float64Array(nClasses);

    for (let i = 0; i < n; i++) {
      const c = classToIdx.get(y[i] ?? 0) ?? 0;
      classCounts[c] = (classCounts[c] ?? 0) + 1;
      const xi = X[i] ?? new Float64Array(p);
      const count = counts[c] ?? new Float64Array(p);
      for (let j = 0; j < p; j++) {
        if ((xi[j] ?? 0) > threshold) count[j] = (count[j] ?? 0) + 1;
      }
    }

    this.classLogPrior_ = new Float64Array(
      Array.from(classCounts).map((c) => Math.log(c / n)),
    );

    this.featureLogProb_ = counts.map((count, c) => {
      const total = classCounts[c] ?? 1;
      return new Float64Array(count.map((cnt) => Math.log((cnt + this.alpha) / (total + 2 * this.alpha))));
    });

    this.featureLogNegProb_ = this.featureLogProb_.map((logProb) =>
      new Float64Array(logProb.map((lp) => Math.log(1 - Math.exp(lp)))),
    );

    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.classes_ === null) throw new NotFittedError("BernoulliNB");
    const classes = this.classes_;
    const nClasses = classes.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const threshold = this.binarize ?? 0.0;

    return new Float64Array(
      X.map((xi) => {
        let maxIdx = 0;
        let maxScore = Number.NEGATIVE_INFINITY;
        for (let c = 0; c < nClasses; c++) {
          let score = (this.classLogPrior_ as Float64Array)[c] ?? 0;
          const flp = (this.featureLogProb_ as Float64Array[])[c] ?? new Float64Array(p);
          const flnp = (this.featureLogNegProb_ as Float64Array[])[c] ?? new Float64Array(p);
          for (let j = 0; j < p; j++) {
            score += (xi[j] ?? 0) > threshold ? (flp[j] ?? 0) : (flnp[j] ?? 0);
          }
          if (score > maxScore) {
            maxScore = score;
            maxIdx = c;
          }
        }
        return classes[maxIdx] ?? 0;
      }),
    );
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      if (pred[i] === y[i]) correct++;
    }
    return correct / y.length;
  }
}

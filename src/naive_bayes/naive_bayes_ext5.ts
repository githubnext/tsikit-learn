/**
 * Online/incremental Naive Bayes classifiers.
 */

export class OnlineGaussianNB {
  nClasses: number;
  varSmoothingInit: number;
  priors: Float64Array | null = null;
  classMeans_: Float64Array[] | null = null;
  classVar_: Float64Array[] | null = null;
  classCounts_: Float64Array | null = null;
  nFeaturesIn_: number = 0;

  constructor(nClasses = 2, varSmoothingInit = 1e-9) {
    this.nClasses = nClasses;
    this.varSmoothingInit = varSmoothingInit;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const p = X[0]?.length ?? 0;
    this.nFeaturesIn_ = p;
    this.classMeans_ = Array.from({ length: this.nClasses }, () => new Float64Array(p));
    this.classVar_ = Array.from({ length: this.nClasses }, () => new Float64Array(p).fill(1.0));
    this.classCounts_ = new Float64Array(this.nClasses);
    // Online Welford's algorithm
    const M2: Float64Array[] = Array.from({ length: this.nClasses }, () => new Float64Array(p));

    for (let i = 0; i < X.length; i++) {
      const c = y[i] ?? 0;
      (this.classCounts_ as Float64Array)[c]++;
      const count = (this.classCounts_ as Float64Array)[c] ?? 1;
      for (let j = 0; j < p; j++) {
        const x = X[i]?.[j] ?? 0;
        const delta = x - ((this.classMeans_[c] as Float64Array)[j] ?? 0);
        (this.classMeans_[c] as Float64Array)[j] += delta / count;
        const delta2 = x - ((this.classMeans_[c] as Float64Array)[j] ?? 0);
        (M2[c] as Float64Array)[j] += delta * delta2;
      }
    }

    const n = X.length;
    for (let c = 0; c < this.nClasses; c++) {
      const count = Math.max((this.classCounts_ as Float64Array)[c] ?? 1, 2);
      for (let j = 0; j < p; j++) {
        (this.classVar_[c] as Float64Array)[j] = ((M2[c] as Float64Array)[j] ?? 0) / (count - 1) + this.varSmoothingInit;
      }
    }

    this.priors = Float64Array.from(this.classCounts_, (c) => c / Math.max(n, 1));
    return this;
  }

  partialFit(X: Float64Array[], y: Int32Array): this {
    if (!this.classMeans_) return this.fit(X, y);
    // Online update using Welford's
    const p = this.nFeaturesIn_;
    const M2: Float64Array[] = this.classVar_!.map((v) => new Float64Array(v).map((s, j) => s * Math.max((this.classCounts_?.[j] ?? 1) - 1, 1)));

    for (let i = 0; i < X.length; i++) {
      const c = y[i] ?? 0;
      (this.classCounts_ as Float64Array)[c]++;
      const count = (this.classCounts_ as Float64Array)[c] ?? 1;
      for (let j = 0; j < p; j++) {
        const x = X[i]?.[j] ?? 0;
        const delta = x - ((this.classMeans_![c] as Float64Array)[j] ?? 0);
        (this.classMeans_![c] as Float64Array)[j] += delta / count;
        const delta2 = x - ((this.classMeans_![c] as Float64Array)[j] ?? 0);
        (M2[c] as Float64Array)[j] += delta * delta2;
      }
    }
    const total = (this.classCounts_ as Float64Array).reduce((s, v) => s + v, 0);
    for (let c = 0; c < this.nClasses; c++) {
      const count = Math.max((this.classCounts_ as Float64Array)[c] ?? 1, 2);
      for (let j = 0; j < p; j++) {
        (this.classVar_![c] as Float64Array)[j] = ((M2[c] as Float64Array)[j] ?? 0) / (count - 1) + this.varSmoothingInit;
      }
      (this.priors as Float64Array)[c] = ((this.classCounts_ as Float64Array)[c] ?? 0) / total;
    }
    return this;
  }

  private _logLikelihood(x: Float64Array, c: number): number {
    const p = this.nFeaturesIn_;
    let logL = Math.log(Math.max((this.priors as Float64Array)[c] ?? 1e-15, 1e-15));
    for (let j = 0; j < p; j++) {
      const v = (this.classVar_![c] as Float64Array)[j] ?? 1;
      const mu = (this.classMeans_![c] as Float64Array)[j] ?? 0;
      logL -= 0.5 * Math.log(2 * Math.PI * v) + (((x[j] ?? 0) - mu) ** 2) / (2 * v);
    }
    return logL;
  }

  predictLogProba(X: Float64Array[]): Float64Array[] {
    if (!this.classMeans_) throw new Error("Not fitted");
    return X.map((x) => Float64Array.from({ length: this.nClasses }, (_, c) => this._logLikelihood(x, c)));
  }

  predictProba(X: Float64Array[]): Float64Array[] {
    return this.predictLogProba(X).map((row) => {
      const maxV = Math.max(...Array.from(row));
      const exps = row.map((v) => Math.exp(v - maxV));
      const sum = exps.reduce((s, v) => s + v, 0);
      return exps.map((v) => v / Math.max(sum, 1e-12));
    });
  }

  predict(X: Float64Array[]): Int32Array {
    return Int32Array.from(this.predictLogProba(X), (row) => {
      let best = 0, bestV = -Number.POSITIVE_INFINITY;
      for (let c = 0; c < row.length; c++) if ((row[c] ?? -Number.POSITIVE_INFINITY) > bestV) { bestV = row[c] ?? -Number.POSITIVE_INFINITY; best = c; }
      return best;
    });
  }

  score(X: Float64Array[], y: Int32Array): number {
    const preds = this.predict(X);
    return preds.filter((p, i) => p === (y[i] ?? -1)).length / Math.max(preds.length, 1);
  }
}

export class ComplementNB {
  alpha: number;
  nClasses: number;
  featureLogProb_: Float64Array[] | null = null;
  classPriors_: Float64Array | null = null;
  nFeaturesIn_: number = 0;

  constructor(alpha = 1.0, nClasses = 2) {
    this.alpha = alpha;
    this.nClasses = nClasses;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length, p = X[0]?.length ?? 0;
    this.nFeaturesIn_ = p;
    const classCounts = new Float64Array(this.nClasses);
    const featureCounts: Float64Array[] = Array.from({ length: this.nClasses }, () => new Float64Array(p));

    for (let i = 0; i < n; i++) {
      const c = y[i] ?? 0;
      classCounts[c]++;
      for (let j = 0; j < p; j++) (featureCounts[c] as Float64Array)[j] += X[i]?.[j] ?? 0;
    }

    this.classPriors_ = Float64Array.from(classCounts, (c) => Math.log(c / n));

    // Complement: use sum of other classes
    this.featureLogProb_ = Array.from({ length: this.nClasses }, (_, c) => {
      const complementCounts = new Float64Array(p);
      for (let c2 = 0; c2 < this.nClasses; c2++) {
        if (c2 !== c) for (let j = 0; j < p; j++) complementCounts[j] += (featureCounts[c2] as Float64Array)[j] ?? 0;
      }
      const total = complementCounts.reduce((s, v) => s + v + this.alpha, 0);
      return complementCounts.map((v) => Math.log((v + this.alpha) / total));
    });
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.featureLogProb_ || !this.classPriors_) throw new Error("Not fitted");
    return Int32Array.from(X, (x) => {
      let best = 0, bestScore = Number.POSITIVE_INFINITY;
      for (let c = 0; c < this.nClasses; c++) {
        const score = -((this.classPriors_ as Float64Array)[c] ?? 0) + (this.featureLogProb_![c] as Float64Array).reduce((s, lp, j) => s - lp * (x[j] ?? 0), 0);
        if (score < bestScore) { bestScore = score; best = c; }
      }
      return best;
    });
  }

  score(X: Float64Array[], y: Int32Array): number {
    const preds = this.predict(X);
    return preds.filter((p, i) => p === (y[i] ?? -1)).length / Math.max(preds.length, 1);
  }
}

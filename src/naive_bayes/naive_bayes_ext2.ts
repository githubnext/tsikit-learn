/**
 * NaiveBayes extended: ComplementNB (extended), OutOfCoreNB, MultinomialNB extended.
 */

export class ComplementNBExt {
  private complementLogProb_: Float64Array[] = [];
  private classPriors_: Float64Array = new Float64Array(0);
  private classes_: Int32Array = new Int32Array(0);
  private alpha: number;

  constructor(params: { alpha?: number; normComplement?: boolean } = {}) {
    this.alpha = params.alpha ?? 1.0;
    void params.normComplement;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const classSet = new Set<number>();
    for (const c of y) classSet.add(c);
    this.classes_ = new Int32Array([...classSet].sort((a, b) => a - b));
    const nClasses = this.classes_.length;
    const nF = X[0]?.length ?? 0;
    const classCounts = new Int32Array(nClasses);
    const featureSums: Float64Array[] = Array.from({ length: nClasses }, () => new Float64Array(nF));
    const n = X.length;
    for (let i = 0; i < n; i++) {
      const ci = this.classes_.indexOf(y[i]!);
      if (ci < 0) continue;
      classCounts[ci] = (classCounts[ci] ?? 0) + 1;
      const x = X[i]!;
      for (let f = 0; f < nF; f++) featureSums[ci]![f] = (featureSums[ci]![f] ?? 0) + (x[f] ?? 0);
    }
    this.classPriors_ = new Float64Array(nClasses);
    for (let k = 0; k < nClasses; k++) this.classPriors_[k] = Math.log((classCounts[k] ?? 0) / Math.max(n, 1));
    // Complement: for each class k, compute log P(f | not k)
    this.complementLogProb_ = Array.from({ length: nClasses }, (_, k) => {
      const complementSums = new Float64Array(nF);
      for (let j = 0; j < nClasses; j++) {
        if (j === k) continue;
        for (let f = 0; f < nF; f++) complementSums[f] = (complementSums[f] ?? 0) + (featureSums[j]?.[f] ?? 0);
      }
      const total = complementSums.reduce((a, b) => a + b, 0) + nF * this.alpha;
      return new Float64Array(complementSums.map((v) => Math.log((v + this.alpha) / Math.max(total, 1e-10))));
    });
    return this;
  }

  predictLogProba(X: Float64Array[]): Float64Array[] {
    return X.map((x) => {
      const logProbs = new Float64Array(this.classes_.length);
      for (let k = 0; k < this.classes_.length; k++) {
        let logP = this.classPriors_[k] ?? 0;
        // Complement NB: use negative complement log probs
        for (let f = 0; f < x.length; f++) logP -= (this.complementLogProb_[k]?.[f] ?? 0) * (x[f] ?? 0);
        logProbs[k] = logP;
      }
      return logProbs;
    });
  }

  predict(X: Float64Array[]): Int32Array {
    const logProbs = this.predictLogProba(X);
    return new Int32Array(logProbs.map((lp) => {
      let best = 0, bestV = -Number.POSITIVE_INFINITY;
      for (let k = 0; k < lp.length; k++) if ((lp[k] ?? 0) > bestV) { bestV = lp[k] ?? 0; best = k; }
      return this.classes_[best] ?? 0;
    }));
  }
}

export class OutOfCoreNBClassifier {
  private featureCounts_: Float64Array[] = [];
  private classCounts_: Int32Array = new Int32Array(0);
  private classes_: Int32Array = new Int32Array(0);
  private nSeen_ = 0;

  constructor(private readonly alpha = 1.0) {}

  partialFit(X: Float64Array[], y: Int32Array, classes?: Int32Array): this {
    if (this.classes_.length === 0) {
      const classSet = new Set<number>();
      if (classes) for (const c of classes) classSet.add(c);
      for (const c of y) classSet.add(c);
      this.classes_ = new Int32Array([...classSet].sort((a, b) => a - b));
      const nF = X[0]?.length ?? 0;
      this.featureCounts_ = Array.from({ length: this.classes_.length }, () => new Float64Array(nF));
      this.classCounts_ = new Int32Array(this.classes_.length);
    }
    const nF = X[0]?.length ?? 0;
    for (let i = 0; i < X.length; i++) {
      const ci = this.classes_.indexOf(y[i]!);
      if (ci < 0) continue;
      this.classCounts_[ci] = (this.classCounts_[ci] ?? 0) + 1;
      this.nSeen_++;
      for (let f = 0; f < nF; f++) {
        this.featureCounts_[ci]![f] = (this.featureCounts_[ci]![f] ?? 0) + (X[i]?.[f] ?? 0);
      }
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    const nClasses = this.classes_.length;
    const nF = X[0]?.length ?? 0;
    return new Int32Array(X.map((x) => {
      let best = 0, bestScore = -Number.POSITIVE_INFINITY;
      for (let k = 0; k < nClasses; k++) {
        const cnt = this.classCounts_[k] ?? 0;
        let score = Math.log((cnt + this.alpha) / (this.nSeen_ + nClasses * this.alpha));
        const totalF = (this.featureCounts_[k] ?? new Float64Array(0)).reduce((a, b) => a + b, 0) + nF * this.alpha;
        for (let f = 0; f < nF; f++) {
          const fc = (this.featureCounts_[k]?.[f] ?? 0) + this.alpha;
          score += (x[f] ?? 0) * Math.log(fc / Math.max(totalF, 1e-10));
        }
        if (score > bestScore) { bestScore = score; best = k; }
      }
      return this.classes_[best] ?? 0;
    }));
  }
}

export class CategoricalNBExt {
  private categories_: number[][] = [];
  private condLogProb_: Array<Float64Array[]> = [];
  private classPriors_: Float64Array = new Float64Array(0);
  private classes_: Int32Array = new Int32Array(0);

  constructor(private readonly alpha = 1.0) {}

  fit(X: Int32Array[], y: Int32Array): this {
    const classSet = new Set<number>();
    for (const c of y) classSet.add(c);
    this.classes_ = new Int32Array([...classSet].sort((a, b) => a - b));
    const nClasses = this.classes_.length;
    const nF = X[0]?.length ?? 0;
    const n = X.length;
    // Find categories per feature
    this.categories_ = Array.from({ length: nF }, () => []);
    for (const x of X) for (let f = 0; f < nF; f++) {
      const v = x[f] ?? 0;
      if (!this.categories_[f]!.includes(v)) this.categories_[f]!.push(v);
    }
    for (const cats of this.categories_) cats.sort((a, b) => a - b);
    // Compute class priors
    const classCounts = new Int32Array(nClasses);
    for (const c of y) { const ci = this.classes_.indexOf(c); if (ci >= 0) classCounts[ci] = (classCounts[ci] ?? 0) + 1; }
    this.classPriors_ = new Float64Array(nClasses);
    for (let k = 0; k < nClasses; k++) this.classPriors_[k] = Math.log((classCounts[k] ?? 0 + this.alpha) / (n + nClasses * this.alpha));
    // Compute conditional log probs
    this.condLogProb_ = Array.from({ length: nClasses }, (_, k) => {
      return Array.from({ length: nF }, (__, f) => {
        const cats = this.categories_[f]!;
        const counts = new Float64Array(cats.length).fill(this.alpha);
        for (let i = 0; i < n; i++) {
          if (this.classes_.indexOf(y[i]!) !== k) continue;
          const catIdx = cats.indexOf(X[i]?.[f] ?? 0);
          if (catIdx >= 0) counts[catIdx] = (counts[catIdx] ?? 0) + 1;
        }
        const total = counts.reduce((a, b) => a + b, 0);
        return new Float64Array(counts.map((c) => Math.log(c / Math.max(total, 1e-10))));
      });
    });
    return this;
  }

  predict(X: Int32Array[]): Int32Array {
    const nClasses = this.classes_.length;
    return new Int32Array(X.map((x) => {
      let best = 0, bestScore = -Number.POSITIVE_INFINITY;
      for (let k = 0; k < nClasses; k++) {
        let score = this.classPriors_[k] ?? 0;
        for (let f = 0; f < x.length; f++) {
          const cats = this.categories_[f]!;
          const catIdx = cats.indexOf(x[f] ?? 0);
          if (catIdx >= 0) score += (this.condLogProb_[k]?.[f]?.[catIdx] ?? 0);
        }
        if (score > bestScore) { bestScore = score; best = k; }
      }
      return this.classes_[best] ?? 0;
    }));
  }
}

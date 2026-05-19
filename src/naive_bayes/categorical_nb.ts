/**
 * Categorical and Complement Naive Bayes classifiers.
 * Mirrors sklearn.naive_bayes.CategoricalNB and ComplementNB.
 */

import { checkIsFitted } from "../base.js";
import { NotFittedError } from "../exceptions.js";

export interface CategoricalNBOptions {
  alpha?: number;
  fitPrior?: boolean;
  classPrior?: Float64Array | null;
  minCategories?: number | null;
}

/**
 * Naive Bayes classifier for categorical features.
 * Each feature is assumed to follow a categorical distribution.
 */
export class CategoricalNB {
  alpha: number;
  fitPrior: boolean;
  classPrior: Float64Array | null;
  minCategories: number | null;

  private classCounts_: Float64Array | null = null;
  private classLogPrior_: Float64Array | null = null;
  private categoryLogProb_: Float64Array[][] | null = null;
  private nCategories_: Int32Array | null = null;
  private classes_: Int32Array | null = null;
  private nFeatures_: number | null = null;

  constructor(options: CategoricalNBOptions = {}) {
    this.alpha = options.alpha ?? 1.0;
    this.fitPrior = options.fitPrior ?? true;
    this.classPrior = options.classPrior ?? null;
    this.minCategories = options.minCategories ?? null;
  }

  fit(X: Int32Array[], y: Int32Array): this {
    const nSamples = X.length;
    const nFeatures = X[0]?.length ?? 0;
    this.nFeatures_ = nFeatures;

    // Find classes
    const classSet = new Set<number>();
    for (let i = 0; i < nSamples; i++) {
      classSet.add(y[i] ?? 0);
    }
    const sortedClasses = Array.from(classSet).sort((a, b) => a - b);
    this.classes_ = new Int32Array(sortedClasses);
    const nClasses = sortedClasses.length;
    const classIndex = new Map<number, number>();
    sortedClasses.forEach((c, i) => classIndex.set(c, i));

    // Count samples per class
    this.classCounts_ = new Float64Array(nClasses);
    for (let i = 0; i < nSamples; i++) {
      const ci = classIndex.get(y[i] ?? 0) ?? 0;
      this.classCounts_[ci]! += 1;
    }

    // Compute log priors
    this.classLogPrior_ = new Float64Array(nClasses);
    if (this.classPrior !== null) {
      for (let c = 0; c < nClasses; c++) {
        this.classLogPrior_[c] = Math.log(this.classPrior[c] ?? (1 / nClasses));
      }
    } else if (this.fitPrior) {
      for (let c = 0; c < nClasses; c++) {
        this.classLogPrior_[c] = Math.log((this.classCounts_[c] ?? 1) / nSamples);
      }
    } else {
      const logUniform = Math.log(1 / nClasses);
      this.classLogPrior_.fill(logUniform);
    }

    // Find number of categories per feature
    const nCats = new Array<number>(nFeatures).fill(0);
    for (let j = 0; j < nFeatures; j++) {
      let maxCat = 0;
      for (let i = 0; i < nSamples; i++) {
        const val = X[i]?.[j] ?? 0;
        if (val > maxCat) maxCat = val;
      }
      const minCats = this.minCategories ?? 0;
      nCats[j] = Math.max(maxCat + 1, minCats);
    }
    this.nCategories_ = new Int32Array(nCats);

    // Count feature-category occurrences per class
    this.categoryLogProb_ = [];
    for (let c = 0; c < nClasses; c++) {
      const classProbs: Float64Array[] = [];
      for (let j = 0; j < nFeatures; j++) {
        classProbs.push(new Float64Array(nCats[j] ?? 1));
      }
      this.categoryLogProb_.push(classProbs);
    }

    for (let i = 0; i < nSamples; i++) {
      const ci = classIndex.get(y[i] ?? 0) ?? 0;
      for (let j = 0; j < nFeatures; j++) {
        const cat = X[i]?.[j] ?? 0;
        const classProbs = this.categoryLogProb_[ci];
        if (classProbs !== undefined && classProbs[j] !== undefined) {
          classProbs[j]![cat] = (classProbs[j]![cat] ?? 0) + 1;
        }
      }
    }

    // Smooth and log-normalize
    for (let c = 0; c < nClasses; c++) {
      for (let j = 0; j < nFeatures; j++) {
        const counts = this.categoryLogProb_![c]?.[j];
        if (counts === undefined) continue;
        const total = (this.classCounts_[c] ?? 0) + this.alpha * (nCats[j] ?? 1);
        for (let k = 0; k < counts.length; k++) {
          counts[k] = Math.log(((counts[k] ?? 0) + this.alpha) / total);
        }
      }
    }

    return this;
  }

  predictLogProba(X: Int32Array[]): Float64Array[] {
    checkIsFitted(this, ["classes_"]);
    const nSamples = X.length;
    const nClasses = this.classes_!.length;
    const result: Float64Array[] = [];

    for (let i = 0; i < nSamples; i++) {
      const logProba = new Float64Array(nClasses);
      for (let c = 0; c < nClasses; c++) {
        logProba[c] = this.classLogPrior_![c] ?? 0;
        const nFeatures = this.nFeatures_ ?? 0;
        for (let j = 0; j < nFeatures; j++) {
          const cat = X[i]?.[j] ?? 0;
          const lp = this.categoryLogProb_![c]?.[j]?.[cat] ?? -Infinity;
          logProba[c] = (logProba[c] ?? 0) + lp;
        }
      }
      result.push(logProba);
    }
    return result;
  }

  predict(X: Int32Array[]): Int32Array {
    const logProba = this.predictLogProba(X);
    const classes = this.classes_!;
    return new Int32Array(logProba.map(lp => {
      let maxIdx = 0;
      let maxVal = lp[0] ?? -Infinity;
      for (let c = 1; c < lp.length; c++) {
        if ((lp[c] ?? -Infinity) > maxVal) {
          maxVal = lp[c]!;
          maxIdx = c;
        }
      }
      return classes[maxIdx] ?? 0;
    }));
  }

  score(X: Int32Array[], y: Int32Array): number {
    const yPred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      if (yPred[i] === y[i]) correct++;
    }
    return correct / y.length;
  }
}

export interface ComplementNBOptions {
  alpha?: number;
  fitPrior?: boolean;
  classPrior?: Float64Array | null;
  norm?: boolean;
}

/**
 * Complement Naive Bayes classifier.
 * Particularly suited for imbalanced datasets.
 */
export class ComplementNB {
  alpha: number;
  fitPrior: boolean;
  classPrior: Float64Array | null;
  norm: boolean;

  private classCounts_: Float64Array | null = null;
  private classLogPrior_: Float64Array | null = null;
  private featureLogProb_: Float64Array[] | null = null;
  private classes_: Int32Array | null = null;
  private nFeatures_: number | null = null;

  constructor(options: ComplementNBOptions = {}) {
    this.alpha = options.alpha ?? 1.0;
    this.fitPrior = options.fitPrior ?? true;
    this.classPrior = options.classPrior ?? null;
    this.norm = options.norm ?? false;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const nSamples = X.length;
    const nFeatures = X[0]?.length ?? 0;
    this.nFeatures_ = nFeatures;

    const classSet = new Set<number>();
    for (let i = 0; i < nSamples; i++) classSet.add(y[i] ?? 0);
    const sortedClasses = Array.from(classSet).sort((a, b) => a - b);
    this.classes_ = new Int32Array(sortedClasses);
    const nClasses = sortedClasses.length;
    const classIndex = new Map<number, number>();
    sortedClasses.forEach((c, i) => classIndex.set(c, i));

    this.classCounts_ = new Float64Array(nClasses);
    // Feature sums per class
    const featureSum = Array.from({ length: nClasses }, () => new Float64Array(nFeatures));
    const totalSum = new Float64Array(nClasses);

    for (let i = 0; i < nSamples; i++) {
      const ci = classIndex.get(y[i] ?? 0) ?? 0;
      this.classCounts_[ci] = (this.classCounts_[ci] ?? 0) + 1;
      for (let j = 0; j < nFeatures; j++) {
        const val = X[i]?.[j] ?? 0;
        featureSum[ci]![j] = (featureSum[ci]![j] ?? 0) + val;
        totalSum[ci] = (totalSum[ci] ?? 0) + val;
      }
    }

    // Class log prior
    this.classLogPrior_ = new Float64Array(nClasses);
    if (this.classPrior !== null) {
      for (let c = 0; c < nClasses; c++) {
        this.classLogPrior_[c] = Math.log(this.classPrior[c] ?? (1 / nClasses));
      }
    } else if (this.fitPrior) {
      for (let c = 0; c < nClasses; c++) {
        this.classLogPrior_[c] = Math.log((this.classCounts_[c] ?? 1) / nSamples);
      }
    } else {
      this.classLogPrior_.fill(Math.log(1 / nClasses));
    }

    // Complement feature log prob: use sum of all OTHER classes
    this.featureLogProb_ = [];
    for (let c = 0; c < nClasses; c++) {
      const compFeatureProb = new Float64Array(nFeatures);
      let compTotal = 0;
      for (let c2 = 0; c2 < nClasses; c2++) {
        if (c2 !== c) {
          compTotal += (totalSum[c2] ?? 0) + this.alpha * nFeatures;
          for (let j = 0; j < nFeatures; j++) {
            compFeatureProb[j] = (compFeatureProb[j] ?? 0) + (featureSum[c2]?.[j] ?? 0) + this.alpha;
          }
        }
      }
      for (let j = 0; j < nFeatures; j++) {
        compFeatureProb[j] = Math.log((compFeatureProb[j] ?? this.alpha) / (compTotal || 1));
      }
      if (this.norm) {
        let norm = 0;
        for (let j = 0; j < nFeatures; j++) norm += Math.abs(compFeatureProb[j] ?? 0);
        if (norm > 0) {
          for (let j = 0; j < nFeatures; j++) compFeatureProb[j] = (compFeatureProb[j] ?? 0) / norm;
        }
      }
      this.featureLogProb_.push(compFeatureProb);
    }

    return this;
  }

  predictLogProba(X: Float64Array[]): Float64Array[] {
    checkIsFitted(this, ["classes_"]);
    const nClasses = this.classes_!.length;
    const nFeatures = this.nFeatures_ ?? 0;

    return X.map(x => {
      const logProba = new Float64Array(nClasses);
      for (let c = 0; c < nClasses; c++) {
        // Complement NB: subtract complement log prob
        let score = this.classLogPrior_![c] ?? 0;
        for (let j = 0; j < nFeatures; j++) {
          score -= (x[j] ?? 0) * (this.featureLogProb_![c]?.[j] ?? 0);
        }
        logProba[c] = score;
      }
      return logProba;
    });
  }

  predict(X: Float64Array[]): Int32Array {
    const logProba = this.predictLogProba(X);
    const classes = this.classes_!;
    return new Int32Array(logProba.map(lp => {
      let maxIdx = 0;
      let maxVal = lp[0] ?? -Infinity;
      for (let c = 1; c < lp.length; c++) {
        if ((lp[c] ?? -Infinity) > maxVal) { maxVal = lp[c]!; maxIdx = c; }
      }
      return classes[maxIdx] ?? 0;
    }));
  }

  score(X: Float64Array[], y: Int32Array): number {
    const yPred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      if (yPred[i] === y[i]) correct++;
    }
    return correct / y.length;
  }
}

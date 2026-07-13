/**
 * Extended multiclass classification utilities.
 * Port of sklearn.multiclass extensions.
 */

import { NotFittedError } from "../exceptions.js";

type Estimator = {
  fit(X: Float64Array[], y: Int32Array): Estimator;
  predict(X: Float64Array[]): Int32Array;
};

/**
 * Error-Correcting Output Codes (ECOC) classifier.
 */
export class ErrorCorrectingOutputCodes {
  private nClasses: number;
  private codeLength: number;
  private codeBook_: Int32Array[] = [];
  private classifiers_: Estimator[] = [];
  private fitted = false;

  constructor(
    private baseEstimator: new () => Estimator,
    options: { nClasses?: number; codeLength?: number } = {}
  ) {
    this.nClasses = options.nClasses ?? 3;
    this.codeLength = options.codeLength ?? 15;
  }

  private generateCodeBook(): Int32Array[] {
    // Random code book
    return Array.from({ length: this.nClasses }, () => {
      const code = new Int32Array(this.codeLength);
      for (let j = 0; j < this.codeLength; j++) {
        code[j] = Math.random() < 0.5 ? -1 : 1;
      }
      return code;
    });
  }

  fit(X: Float64Array[], y: Int32Array): this {
    this.codeBook_ = this.generateCodeBook();
    this.classifiers_ = [];

    for (let j = 0; j < this.codeLength; j++) {
      // Binary labels: +1 or -1 based on code book
      const binaryY = Int32Array.from(y, label => this.codeBook_[label]?.[j] ?? 0 > 0 ? 1 : 0);
      const clf = new this.baseEstimator();
      clf.fit(X, binaryY);
      this.classifiers_.push(clf);
    }

    this.fitted = true;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted) throw new NotFittedError("ErrorCorrectingOutputCodes not fitted");

    // Get binary predictions for each classifier
    const codes: Int32Array[] = this.classifiers_.map(clf => clf.predict(X));

    return Int32Array.from(X, (_, i) => {
      let bestClass = 0; let bestDist = Number.POSITIVE_INFINITY;
      for (let c = 0; c < this.nClasses; c++) {
        let hamming = 0;
        for (let j = 0; j < this.codeLength; j++) {
          const pred = codes[j]?.[i] ?? 0;
          const expected = (this.codeBook_[c]?.[j] ?? 0) > 0 ? 1 : 0;
          hamming += pred !== expected ? 1 : 0;
        }
        if (hamming < bestDist) { bestDist = hamming; bestClass = c; }
      }
      return bestClass;
    });
  }
}

/**
 * Nested One-vs-One classifier with tournament structure.
 */
export class TournamentClassifier {
  private nClasses: number;
  private pairwiseClfs_: Map<string, Estimator> = new Map();
  private fitted = false;

  constructor(
    private baseEstimator: new () => Estimator,
    options: { nClasses?: number } = {}
  ) {
    this.nClasses = options.nClasses ?? 3;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    // Train one classifier per pair
    for (let i = 0; i < this.nClasses; i++) {
      for (let j = i + 1; j < this.nClasses; j++) {
        const mask = Array.from(y, (label, idx) => label === i || label === j ? idx : -1).filter(idx => idx >= 0);
        const subX = mask.map(idx => X[idx]!);
        const subY = Int32Array.from(mask, idx => y[idx] === i ? 0 : 1);
        const clf = new this.baseEstimator();
        clf.fit(subX, subY);
        this.pairwiseClfs_.set(`${i}_${j}`, clf);
      }
    }
    this.fitted = true;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted) throw new NotFittedError("TournamentClassifier not fitted");
    return Int32Array.from(X, (x, _) => {
      const votes = new Int32Array(this.nClasses);
      for (let i = 0; i < this.nClasses; i++) {
        for (let j = i + 1; j < this.nClasses; j++) {
          const clf = this.pairwiseClfs_.get(`${i}_${j}`);
          if (!clf) continue;
          const pred = clf.predict([x]);
          const winner = (pred[0] ?? 0) === 0 ? i : j;
          votes[winner]!++;
        }
      }
      let best = 0; let bestVotes = -1;
      for (let c = 0; c < this.nClasses; c++) {
        if ((votes[c] ?? 0) > bestVotes) { bestVotes = votes[c] ?? 0; best = c; }
      }
      return best;
    });
  }
}

/**
 * Confusion matrix utilities for multiclass problems.
 */
export function multiclassConfusionMatrix(
  yTrue: Int32Array,
  yPred: Int32Array,
  nClasses: number,
): Int32Array[] {
  const cm = Array.from({ length: nClasses }, () => new Int32Array(nClasses));
  for (let i = 0; i < yTrue.length; i++) {
    const t = yTrue[i] ?? 0;
    const p = yPred[i] ?? 0;
    cm[t]![p]!++;
  }
  return cm;
}

/**
 * Multiclass ROC AUC using OvR approach.
 */
export function multiclassRocAuc(
  yTrue: Int32Array,
  yScore: Float64Array[],
  nClasses: number,
  average: "macro" | "weighted" = "macro",
): number {
  const aucs: number[] = [];
  const classCounts = new Int32Array(nClasses);
  for (let i = 0; i < yTrue.length; i++) classCounts[yTrue[i] ?? 0]! += 1;

  for (let c = 0; c < nClasses; c++) {
    const scores = Float64Array.from(yScore, row => row[c] ?? 0);
    const binary = Int32Array.from(yTrue, label => label === c ? 1 : 0);

    // Compute AUC via trapezoidal rule
    const pairs = Array.from({ length: scores.length }, (_, i) => ({ score: scores[i] ?? 0, label: binary[i] ?? 0 }));
    pairs.sort((a, b) => b.score - a.score);

    let tp = 0; let fp = 0;
    const nPos = (classCounts[c] ?? 0);
    const nNeg = yTrue.length - nPos;
    let auc = 0;
    let prevTp = 0; let prevFp = 0;

    for (const { label } of pairs) {
      if (label === 1) tp++; else fp++;
      auc += (fp - prevFp) * (tp + prevTp) / 2;
      prevTp = tp; prevFp = fp;
    }

    aucs.push(nPos > 0 && nNeg > 0 ? auc / (nPos * nNeg) : 0.5);
  }

  if (average === "weighted") {
    const total = yTrue.length;
    return aucs.reduce((s, auc, c) => s + auc * (classCounts[c] ?? 0) / total, 0);
  }
  return aucs.reduce((s, v) => s + v, 0) / nClasses;
}

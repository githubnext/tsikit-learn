/**
 * OutputCodeClassifier — Error-Correcting Output Codes multiclass strategy.
 * Mirrors sklearn.multiclass.OutputCodeClassifier.
 */

import { NotFittedError } from "../exceptions.js";

export interface BinaryClassifierOCC {
  fit(X: Float64Array[], y: Int32Array | Float64Array): this;
  predict(X: Float64Array[]): Int32Array | Float64Array;
}

export interface OutputCodeClassifierOptions {
  estimator: BinaryClassifierOCC;
  code_size?: number;
  random_state?: number;
}

/**
 * OutputCodeClassifier — reduces multiclass classification to a set of binary
 * problems using random output codes. The number of binary classifiers is
 * `ceil(n_classes * code_size)`. At prediction time, the output vector is
 * compared to each class row in the code book and the nearest class wins.
 */
export class OutputCodeClassifier {
  estimator: BinaryClassifierOCC;
  code_size: number;
  random_state: number;

  estimators_: BinaryClassifierOCC[] | null = null;
  classes_: Int32Array | null = null;
  code_book_: Int32Array[] | null = null;

  constructor(options: OutputCodeClassifierOptions) {
    this.estimator = options.estimator;
    this.code_size = options.code_size ?? 1.5;
    this.random_state = options.random_state ?? 42;
  }

  /** Simple seeded LCG random bit generator. */
  private _rng(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
      s = (Math.imul(1664525, s) + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }

  fit(X: Float64Array[], y: Int32Array | Float64Array): this {
    const n = X.length;
    const classSet = new Set<number>();
    for (let i = 0; i < n; i++) classSet.add(y[i] ?? 0);
    const classes = Int32Array.from([...classSet].sort((a, b) => a - b));
    this.classes_ = classes;
    const nClasses = classes.length;
    const nCodes = Math.ceil(nClasses * this.code_size);

    // Generate random code book [nClasses x nCodes] with 0/1 entries
    const rand = this._rng(this.random_state);
    const codeBook: Int32Array[] = Array.from({ length: nClasses }, () => {
      const row = new Int32Array(nCodes);
      for (let j = 0; j < nCodes; j++) row[j]! = rand() < 0.5 ? 0 : 1;
      return row;
    });
    this.code_book_ = codeBook;

    // Train one binary classifier per code column
    const classIndex = new Map<number, number>();
    for (let ci = 0; ci < nClasses; ci++) classIndex.set(classes[ci]!, ci);

    this.estimators_ = Array.from({ length: nCodes }, (_, col) => {
      // Binary labels: 0 or 1 based on code book column
      const yBin = Float64Array.from({ length: n }, (_, i) => {
        const ci = classIndex.get(y[i] ?? 0) ?? 0;
        return codeBook[ci]![col]! ?? 0;
      });
      const clf = Object.assign(
        Object.create(Object.getPrototypeOf(this.estimator)) as BinaryClassifierOCC,
        this.estimator,
      );
      return clf.fit(X, yBin);
    });

    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.estimators_ || !this.classes_ || !this.code_book_)
      throw new NotFittedError("OutputCodeClassifier is not fitted");

    const nCodes = this.estimators_.length;
    const nClasses = this.classes_.length;

    // Collect binary predictions [nCodes]
    const binPreds: (Int32Array | Float64Array)[] = this.estimators_.map((clf) =>
      clf.predict(X),
    );

    return Int32Array.from({ length: X.length }, (_, i) => {
      // Build output vector for sample i
      const outVec = new Float64Array(nCodes);
      for (let col = 0; col < nCodes; col++) outVec[col]! = binPreds[col]![i]! ?? 0;

      // Find nearest class by Hamming distance
      let bestClass = 0;
      let bestDist = Infinity;
      for (let ci = 0; ci < nClasses; ci++) {
        let dist = 0;
        for (let col = 0; col < nCodes; col++) {
          dist += Math.abs((outVec[col]! ?? 0) - (this.code_book_![ci]![col]! ?? 0));
        }
        if (dist < bestDist) {
          bestDist = dist;
          bestClass = ci;
        }
      }
      return this.classes_![bestClass]! ?? 0;
    });
  }

  score(X: Float64Array[], y: Int32Array | Float64Array): number {
    const preds = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) if (preds[i] === y[i]) correct++;
    return correct / y.length;
  }
}

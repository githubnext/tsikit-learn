/**
 * SVM multiclass strategies.
 * Mirrors scikit-learn's svm multiclass support via OvO and OvR decompositions.
 */

export interface BinarySVM {
  fit(X: Float64Array[], y: Int32Array): this;
  predict(X: Float64Array[]): Int32Array;
  decisionFunction?(X: Float64Array[]): Float64Array;
}

/**
 * One-vs-One multiclass SVM wrapper.
 * Trains K*(K-1)/2 binary classifiers, uses voting for prediction.
 */
export class OvOSVM {
  private _classifiers: Array<{
    clf: BinarySVM;
    class0: number;
    class1: number;
  }> = [];
  private _classes: Int32Array | null = null;

  constructor(private readonly _baseClf: () => BinarySVM) {}

  fit(X: Float64Array[], y: Int32Array): this {
    const classes = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
    this._classes = Int32Array.from(classes);
    this._classifiers = [];

    for (let i = 0; i < classes.length; i++) {
      for (let j = i + 1; j < classes.length; j++) {
        const c0 = classes[i]!;
        const c1 = classes[j]!;
        const mask = Array.from(y)
          .map((label, idx) => ({ idx, label }))
          .filter(({ label }) => label === c0 || label === c1);
        const XBin = mask.map(({ idx }) => X[idx]!);
        const yBin = Int32Array.from(mask, ({ label }) =>
          label === c0 ? 0 : 1,
        );
        const clf = this._baseClf();
        clf.fit(XBin, yBin);
        this._classifiers.push({ clf, class0: c0, class1: c1 });
      }
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (this._classes === null) throw new Error("OvOSVM must be fitted first");
    const classes = this._classes;
    const votes = X.map(() => new Map<number, number>());

    for (const { clf, class0, class1 } of this._classifiers) {
      const preds = clf.predict(X);
      for (let i = 0; i < X.length; i++) {
        const winner = preds[i] === 0 ? class0 : class1;
        votes[i]!.set(winner, (votes[i]!.get(winner) ?? 0) + 1);
      }
    }

    return Int32Array.from(votes, (v) => {
      let best = classes[0]!;
      let bestVotes = -1;
      for (const [cls, count] of v) {
        if (count > bestVotes) {
          bestVotes = count;
          best = cls;
        }
      }
      return best;
    });
  }
}

/**
 * One-vs-Rest multiclass SVM wrapper.
 */
export class OvRSVM {
  private _classifiers: Array<{ clf: BinarySVM; cls: number }> = [];
  private _classes: Int32Array | null = null;

  constructor(private readonly _baseClf: () => BinarySVM) {}

  fit(X: Float64Array[], y: Int32Array): this {
    const classes = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
    this._classes = Int32Array.from(classes);
    this._classifiers = [];

    for (const cls of classes) {
      const yBin = Int32Array.from(y, (label) => (label === cls ? 1 : 0));
      const clf = this._baseClf();
      clf.fit(X, yBin);
      this._classifiers.push({ clf, cls });
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (this._classes === null) throw new Error("OvRSVM must be fitted first");
    const scores: Float64Array[] = this._classifiers.map(({ clf }) => {
      if (typeof clf.decisionFunction === "function") {
        return clf.decisionFunction(X);
      }
      // Fallback: use predict (0 or 1)
      const preds = clf.predict(X);
      return Float64Array.from(preds);
    });

    return Int32Array.from({ length: X.length }, (_, i) => {
      let best = this._classes![0]!;
      let bestScore = Number.NEGATIVE_INFINITY;
      for (let c = 0; c < this._classifiers.length; c++) {
        const s = scores[c]?.[i] ?? 0;
        if (s > bestScore) {
          bestScore = s;
          best = this._classifiers[c]!.cls;
        }
      }
      return best;
    });
  }
}

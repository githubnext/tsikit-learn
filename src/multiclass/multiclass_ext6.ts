/**
 * Output code classifier, ECOC, and error correcting output codes.
 */

export interface BinaryClassifier {
  fit(X: Float64Array[], y: Int32Array): this;
  predict(X: Float64Array[]): Int32Array;
  predictProba?(X: Float64Array[]): Float64Array[];
}

export class OutputCodeClassifier {
  private estimators_: BinaryClassifier[] = [];
  private codeBook_!: Int32Array[];
  private classes_!: Int32Array;
  private fitted_ = false;

  constructor(
    private estimatorFactory: () => BinaryClassifier,
    private codeSize = 1.5
  ) {}

  fit(X: Float64Array[], y: Int32Array): this {
    const classSet = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
    this.classes_ = new Int32Array(classSet);
    const nClasses = classSet.length;
    const nCode = Math.max(nClasses, Math.ceil(nClasses * this.codeSize));

    // Random code book: {-1, 1}
    this.codeBook_ = Array.from({ length: nClasses }, () =>
      new Int32Array(nCode).map(() => Math.random() < 0.5 ? 1 : -1)
    );

    // Ensure each code column has both +1 and -1
    for (let col = 0; col < nCode; col++) {
      let hasPos = false, hasNeg = false;
      for (let ci = 0; ci < nClasses; ci++) {
        if ((this.codeBook_[ci]![col] ?? 0) === 1) hasPos = true;
        else hasNeg = true;
      }
      if (!hasPos) this.codeBook_[0]![col] = 1;
      if (!hasNeg) this.codeBook_[1 % nClasses]![col] = -1;
    }

    // Train one binary classifier per code column
    for (let col = 0; col < nCode; col++) {
      const yBin = new Int32Array(y.length).map((_, i) => {
        const classIdx = classSet.indexOf(y[i] ?? 0);
        return classIdx >= 0 ? (this.codeBook_[classIdx]![col] ?? -1) : -1;
      });
      // Only train if both classes are present
      const hasBoth = Array.from(yBin).some(v => v === 1) && Array.from(yBin).some(v => v === -1);
      const est = this.estimatorFactory();
      if (hasBoth) est.fit(X, yBin);
      else {
        // Create dummy estimator
        const majority = Array.from(yBin).filter(v => v === 1).length >= yBin.length / 2 ? 1 : -1;
        const dummy = this.estimatorFactory();
        const dummyY = new Int32Array(2).fill(majority);
        dummy.fit([X[0] ?? new Float64Array(), X[0] ?? new Float64Array()], dummyY);
        this.estimators_.push(dummy);
        continue;
      }
      this.estimators_.push(est);
    }
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted_) throw new Error('Not fitted');
    const nCode = this.estimators_.length;
    const nClasses = this.classes_.length;
    const predictions = this.estimators_.map(est => est.predict(X));
    return new Int32Array(X.length).map((_, i) => {
      // For each class, compute Hamming distance to its code
      let bestClass = this.classes_[0]!, bestDist = Number.POSITIVE_INFINITY;
      for (let ci = 0; ci < nClasses; ci++) {
        let dist = 0;
        for (let col = 0; col < nCode; col++) {
          const pred = predictions[col]![i] ?? 0;
          const code = this.codeBook_[ci]![col] ?? 0;
          if (pred !== code) dist++;
        }
        if (dist < bestDist) { bestDist = dist; bestClass = this.classes_[ci]!; }
      }
      return bestClass;
    });
  }
}

export class ECOC extends OutputCodeClassifier {
  constructor(estimatorFactory: () => BinaryClassifier, codeType: 'dense' | 'sparse' = 'dense') {
    super(estimatorFactory, codeType === 'dense' ? 2.0 : 0.5);
  }
}

export function oneVsAllStrategy(
  estimatorFactory: () => BinaryClassifier,
  X: Float64Array[],
  y: Int32Array
): { classifiers: Map<number, BinaryClassifier>; classes: Int32Array } {
  const classSet = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
  const classifiers = new Map<number, BinaryClassifier>();
  for (const c of classSet) {
    const yBin = new Int32Array(y.length).map((_, i) => y[i] === c ? 1 : 0);
    const est = estimatorFactory();
    est.fit(X, yBin);
    classifiers.set(c, est);
  }
  return { classifiers, classes: new Int32Array(classSet) };
}

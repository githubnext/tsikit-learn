/**
 * ECOC (Error Correcting Output Codes) multi-class classifier.
 */

export interface BinaryEstimator {
  fit(X: Float64Array[], y: Int32Array): this;
  decisionFunction?(X: Float64Array[]): Float64Array;
  predict(X: Float64Array[]): Int32Array;
}

export function hadamardCodebook(nClasses: number): Int32Array[] {
  // Generate a simple Hadamard-like codebook for error correcting output codes
  const nBits = Math.max(nClasses, 4);
  const codebook: Int32Array[] = Array.from({ length: nClasses }, (_, i) =>
    Int32Array.from({ length: nBits }, (_, b) => ((i >> (b % Math.ceil(Math.log2(nClasses + 1)))) & 1) * 2 - 1)
  );
  return codebook;
}

export function exhaustiveCodebook(nClasses: number): Int32Array[] {
  const nBits = Math.min(Math.pow(2, nClasses - 1) - 1, 64);
  const codebook: Int32Array[] = [];
  for (let c = 0; c < nClasses; c++) {
    const code = new Int32Array(nBits);
    for (let b = 0; b < nBits; b++) {
      code[b] = ((b + 1) >> c) & 1 ? 1 : -1;
    }
    codebook.push(code);
  }
  return codebook;
}

export class ECOCClassifier {
  estimatorFactory: () => BinaryEstimator;
  coding: "ovr" | "hadamard" | "exhaustive";
  nClasses: number;
  private _codebook: Int32Array[] = [];
  private _estimators: BinaryEstimator[] = [];

  constructor(estimatorFactory: () => BinaryEstimator, coding: "ovr" | "hadamard" | "exhaustive" = "hadamard", nClasses = 3) {
    this.estimatorFactory = estimatorFactory;
    this.coding = coding;
    this.nClasses = nClasses;
  }

  private _buildCodebook(): Int32Array[] {
    if (this.coding === "ovr") {
      return Array.from({ length: this.nClasses }, (_, c) =>
        Int32Array.from({ length: this.nClasses }, (_, k) => k === c ? 1 : -1)
      );
    } else if (this.coding === "hadamard") {
      return hadamardCodebook(this.nClasses);
    } else {
      return exhaustiveCodebook(this.nClasses);
    }
  }

  fit(X: Float64Array[], y: Int32Array): this {
    this._codebook = this._buildCodebook();
    const nBits = this._codebook[0]?.length ?? 0;
    this._estimators = Array.from({ length: nBits }, (_, b) => {
      const binaryY = Int32Array.from(y, (label) => {
        const code = (this._codebook[label] as Int32Array)[b] ?? 0;
        return code > 0 ? 1 : 0;
      });
      const est = this.estimatorFactory();
      est.fit(X, binaryY);
      return est;
    });
    return this;
  }

  decisionFunction(X: Float64Array[]): Float64Array[] {
    const nBits = this._estimators.length;
    const n = X.length;
    const scores = Array.from({ length: n }, () => new Float64Array(this.nClasses));

    const bitPredictions: Float64Array[] = this._estimators.map((est) => {
      if (est.decisionFunction) return est.decisionFunction(X);
      const preds = est.predict(X);
      return Float64Array.from(preds, (p) => p * 2 - 1);
    });

    for (let i = 0; i < n; i++) {
      for (let c = 0; c < this.nClasses; c++) {
        let hamming = 0;
        for (let b = 0; b < nBits; b++) {
          const predSign = (bitPredictions[b]?.[i] ?? 0) > 0 ? 1 : -1;
          const codeSign = (this._codebook[c] as Int32Array)[b] ?? 0;
          hamming += predSign !== codeSign ? 1 : 0;
        }
        (scores[i] as Float64Array)[c] = -hamming;
      }
    }
    return scores;
  }

  predict(X: Float64Array[]): Int32Array {
    const scores = this.decisionFunction(X);
    return Int32Array.from(scores, (row) => {
      let best = 0, bestScore = -Number.POSITIVE_INFINITY;
      for (let c = 0; c < row.length; c++) {
        if ((row[c] ?? -Number.POSITIVE_INFINITY) > bestScore) { bestScore = row[c] ?? -Number.POSITIVE_INFINITY; best = c; }
      }
      return best;
    });
  }

  score(X: Float64Array[], y: Int32Array): number {
    const preds = this.predict(X);
    return preds.filter((p, i) => p === (y[i] ?? -1)).length / Math.max(preds.length, 1);
  }
}

export class OneVsOneClassifier {
  estimatorFactory: () => BinaryEstimator;
  nClasses: number;
  private _classifiers: Array<{ i: number; j: number; est: BinaryEstimator }> = [];

  constructor(estimatorFactory: () => BinaryEstimator, nClasses = 3) {
    this.estimatorFactory = estimatorFactory;
    this.nClasses = nClasses;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    this._classifiers = [];
    for (let i = 0; i < this.nClasses; i++) {
      for (let j = i + 1; j < this.nClasses; j++) {
        const mask = Array.from(y).map((label) => label === i || label === j);
        const Xij = X.filter((_, k) => mask[k]);
        const yij = Int32Array.from(Array.from(y).filter((_, k) => mask[k]).map((label) => label === i ? 0 : 1));
        const est = this.estimatorFactory();
        est.fit(Xij, yij);
        this._classifiers.push({ i, j, est });
      }
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    const n = X.length;
    const votes = Array.from({ length: n }, () => new Int32Array(this.nClasses));
    for (const { i, j, est } of this._classifiers) {
      const preds = est.predict(X);
      for (let k = 0; k < n; k++) {
        if ((preds[k] ?? 0) === 0) (votes[k]! as Int32Array)[i]!++;
        else (votes[k]! as Int32Array)[j]!++;
      }
    }
    return Int32Array.from(votes, (row) => {
      let best = 0, bestVotes = -1;
      for (let c = 0; c < row.length; c++) if ((row[c] ?? 0) > bestVotes) { bestVotes = row[c] ?? 0; best = c; }
      return best;
    });
  }

  score(X: Float64Array[], y: Int32Array): number {
    const preds = this.predict(X);
    return preds.filter((p, i) => p === (y[i] ?? -1)).length / Math.max(preds.length, 1);
  }
}

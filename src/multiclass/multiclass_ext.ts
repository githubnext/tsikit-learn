/**
 * Multiclass extensions: extended OvO, ECOC, pairwise coupling.
 */

export class ExtendedOneVsOneClassifier {
  private classifiers: Map<string, { fit: (X: Float64Array[], y: Int32Array) => void; predict: (X: Float64Array[]) => Int32Array }> = new Map();
  private classes: Int32Array = new Int32Array(0);

  constructor(
    private readonly baseClassifierFactory: () => { fit: (X: Float64Array[], y: Int32Array) => void; predict: (X: Float64Array[]) => Int32Array }
  ) {}

  fit(X: Float64Array[], y: Int32Array): this {
    const classSet = new Set<number>();
    for (const c of y) classSet.add(c);
    this.classes = new Int32Array([...classSet].sort((a, b) => a - b));
    for (let i = 0; i < this.classes.length; i++) {
      for (let j = i + 1; j < this.classes.length; j++) {
        const ci = this.classes[i]!;
        const cj = this.classes[j]!;
        const mask = y.map((v) => v === ci || v === cj ? 1 : 0);
        const Xi: Float64Array[] = [];
        const yi: number[] = [];
        for (let k = 0; k < y.length; k++) {
          if (mask[k] === 1) {
            Xi.push(X[k]!);
            yi.push(y[k] === ci ? 1 : -1);
          }
        }
        const clf = this.baseClassifierFactory();
        clf.fit(Xi, new Int32Array(yi));
        this.classifiers.set(`${ci}-${cj}`, clf);
      }
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    const votes = X.map(() => new Map<number, number>());
    for (let i = 0; i < this.classes.length; i++) {
      for (let j = i + 1; j < this.classes.length; j++) {
        const ci = this.classes[i]!;
        const cj = this.classes[j]!;
        const clf = this.classifiers.get(`${ci}-${cj}`);
        if (!clf) continue;
        const preds = clf.predict(X);
        for (let k = 0; k < preds.length; k++) {
          const winner = preds[k] === 1 ? ci : cj;
          const v = votes[k];
          if (v !== undefined) v.set(winner, (v.get(winner) ?? 0) + 1);
        }
      }
    }
    return new Int32Array(votes.map((v) => {
      let best = this.classes[0] ?? 0;
      let bestVotes = 0;
      for (const [c, cnt] of v) {
        if (cnt > bestVotes) { bestVotes = cnt; best = c; }
      }
      return best;
    }));
  }
}

export class ErrorCorrectionOutputCode {
  private codeMatrix: Int32Array[] = [];
  private classifiers: Array<{ fit: (X: Float64Array[], y: Int32Array) => void; predict: (X: Float64Array[]) => Int32Array }> = [];
  private classes: Int32Array = new Int32Array(0);

  constructor(
    private readonly baseClassifierFactory: () => { fit: (X: Float64Array[], y: Int32Array) => void; predict: (X: Float64Array[]) => Int32Array },
    private readonly codeSize = 1.5
  ) {}

  fit(X: Float64Array[], y: Int32Array): this {
    const classSet = new Set<number>();
    for (const c of y) classSet.add(c);
    this.classes = new Int32Array([...classSet].sort((a, b) => a - b));
    const nClasses = this.classes.length;
    const nBits = Math.max(10, Math.ceil(this.codeSize * Math.ceil(Math.log2(nClasses))));
    // Generate random code matrix
    this.codeMatrix = Array.from({ length: nClasses }, () => {
      const code = new Int32Array(nBits);
      for (let b = 0; b < nBits; b++) code[b] = Math.random() < 0.5 ? 1 : -1;
      return code;
    });
    // Train one classifier per bit
    this.classifiers = [];
    for (let b = 0; b < nBits; b++) {
      const binaryY = new Int32Array(y.length);
      for (let k = 0; k < y.length; k++) {
        const ci = this.classes.indexOf(y[k]!);
        binaryY[k] = this.codeMatrix[ci]?.[b] ?? 1;
      }
      const clf = this.baseClassifierFactory();
      clf.fit(X, binaryY);
      this.classifiers.push(clf);
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    const nBits = this.classifiers.length;
    const codes = Array.from({ length: X.length }, () => new Float64Array(nBits));
    for (let b = 0; b < nBits; b++) {
      const preds = this.classifiers[b]?.predict(X) ?? new Int32Array(X.length);
      for (let k = 0; k < X.length; k++) {
        const c = codes[k];
        if (c !== undefined) c[b] = preds[k] ?? 1;
      }
    }
    return new Int32Array(codes.map((code) => {
      let best = this.classes[0] ?? 0;
      let bestDist = Number.POSITIVE_INFINITY;
      for (let ci = 0; ci < this.classes.length; ci++) {
        const classCode = this.codeMatrix[ci];
        if (!classCode) continue;
        let dist = 0;
        for (let b = 0; b < nBits; b++) dist += Math.abs((code[b] ?? 0) - (classCode[b] ?? 0));
        if (dist < bestDist) { bestDist = dist; best = this.classes[ci] ?? 0; }
      }
      return best;
    }));
  }
}

export class PairwiseCouplingClassifier {
  private pairwiseProbs: Map<string, Float64Array> = new Map();
  private classes: Int32Array = new Int32Array(0);

  setClasses(classes: Int32Array): void {
    this.classes = classes;
  }

  setPairwiseProb(ci: number, cj: number, probs: Float64Array): void {
    this.pairwiseProbs.set(`${ci}-${cj}`, probs);
  }

  predict(nSamples: number): Int32Array {
    const result = new Int32Array(nSamples);
    for (let k = 0; k < nSamples; k++) {
      let best = this.classes[0] ?? 0;
      let bestScore = -1;
      for (let i = 0; i < this.classes.length; i++) {
        let score = 0;
        const ci = this.classes[i]!;
        for (let j = 0; j < this.classes.length; j++) {
          if (i === j) continue;
          const cj = this.classes[j]!;
          const probs = this.pairwiseProbs.get(`${ci}-${cj}`) ?? this.pairwiseProbs.get(`${cj}-${ci}`);
          if (probs) score += probs[k] ?? 0.5;
        }
        if (score > bestScore) { bestScore = score; best = ci; }
      }
      result[k] = best;
    }
    return result;
  }
}

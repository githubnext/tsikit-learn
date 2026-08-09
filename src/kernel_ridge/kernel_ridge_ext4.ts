/**
 * Kernel Ridge with Cross-Validation.
 */

export function kernelRidgeCv(
  X: Float64Array[], y: Float64Array,
  alphas: Float64Array, kernel: (x1: Float64Array, x2: Float64Array) => number,
  cv = 5
): number {
  const n = X.length;
  let bestAlpha = alphas[0] ?? 1.0;
  let bestScore = -Number.POSITIVE_INFINITY;
  for (let ai = 0; ai < alphas.length; ai++) {
    const alpha = alphas[ai] ?? 1.0;
    let totalScore = 0;
    for (let fold = 0; fold < cv; fold++) {
      const testIdx: number[] = [], trainIdx: number[] = [];
      for (let i = 0; i < n; i++) {
        if (i % cv === fold) testIdx.push(i);
        else trainIdx.push(i);
      }
      const Xtr = trainIdx.map(i => X[i]!);
      const ytr = new Float64Array(trainIdx.map(i => y[i] ?? 0));
      const Xte = testIdx.map(i => X[i]!);
      const yte = testIdx.map(i => y[i] ?? 0);
      // Build kernel matrix
      const nTr = Xtr.length;
      const K = Array.from({ length: nTr }, (_, i) =>
        new Float64Array(nTr).map((_, j) => kernel(Xtr[i]!, Xtr[j]!))
      );
      // Solve (K + alpha*I) alpha_ = y
      for (let i = 0; i < nTr; i++) K[i]![i]! += alpha;
      const alpha_ = solveLinear(K, ytr);
      // Predict
      let sse = 0, sst = 0;
      const ymean = yte.reduce((a, b) => a + b, 0) / yte.length;
      for (let i = 0; i < testIdx.length; i++) {
        let pred = 0;
        for (let j = 0; j < nTr; j++) pred += (alpha_[j] ?? 0) * kernel(Xte[i]!, Xtr[j]!);
        sse += (yte[i]! - pred) ** 2;
        sst += (yte[i]! - ymean) ** 2;
      }
      totalScore += 1 - sse / (sst || 1);
    }
    const score = totalScore / cv;
    if (score > bestScore) { bestScore = score; bestAlpha = alpha; }
  }
  return bestAlpha;
}

function solveLinear(A: Float64Array[], b: Float64Array): Float64Array {
  const n = A.length;
  const M = A.map(row => Float64Array.from(row));
  const x = Float64Array.from(b);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++)
      if (Math.abs(M[row]![col] ?? 0) > Math.abs(M[maxRow]![col] ?? 0)) maxRow = row;
    const tmp = M[col]; M[col] = M[maxRow]!; M[maxRow] = tmp!;
    const xtmp = x[col] ?? 0; x[col] = x[maxRow] ?? 0; x[maxRow] = xtmp;
    for (let row = col + 1; row < n; row++) {
      const f = (M[row]![col] ?? 0) / (M[col]![col] ?? 1);
      for (let k = col; k < n; k++) M[row]![k]! -= f * (M[col]![k] ?? 0);
      x[row]! -= f * (x[col] ?? 0);
    }
  }
  for (let i = n - 1; i >= 0; i--) {
    x[i]! = (x[i] ?? 0);
    for (let j = i + 1; j < n; j++) x[i]! -= (M[i]![j] ?? 0) * (x[j] ?? 0);
    x[i]! /= M[i]![i] ?? 1;
  }
  return x;
}

export class KernelRidgeCV {
  private bestAlpha_!: number;
  private alpha_!: Float64Array;
  private XTrain_!: Float64Array[];
  private fitted_ = false;

  constructor(
    private alphas: Float64Array = new Float64Array([0.1, 1.0, 10.0]),
    private gamma = 1.0,
    private cv = 5
  ) {}

  private rbf(x1: Float64Array, x2: Float64Array): number {
    return Math.exp(-this.gamma * x1.reduce((s, v, i) => s + (v - (x2[i] ?? 0)) ** 2, 0));
  }

  fit(X: Float64Array[], y: Float64Array): this {
    this.bestAlpha_ = kernelRidgeCv(X, y, this.alphas, this.rbf.bind(this), this.cv);
    const n = X.length;
    const K = Array.from({ length: n }, (_, i) =>
      new Float64Array(n).map((_, j) => this.rbf(X[i]!, X[j]!))
    );
    for (let i = 0; i < n; i++) K[i]![i]! += this.bestAlpha_;
    this.alpha_ = solveLinear(K, y);
    this.XTrain_ = X;
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error("Not fitted");
    return new Float64Array(X.map(x =>
      this.XTrain_.reduce((s, xt, j) => s + (this.alpha_[j] ?? 0) * this.rbf(x, xt), 0)
    ));
  }

  get bestAlpha(): number { return this.bestAlpha_; }
}

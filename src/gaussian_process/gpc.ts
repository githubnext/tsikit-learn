/**
 * Gaussian Process Classifier.
 */

export class GaussianProcessClassifier {
  private alpha_: Float64Array = new Float64Array(0);
  private XTrain_: Float64Array[] = [];
  private classes_: Int32Array = new Int32Array(0);
  private fitted = false;

  constructor(
    private readonly lengthScale = 1.0,
    private readonly noiseLevel = 1e-8,
    private readonly maxIter = 100,
  ) {}

  private _rbfKernel(X1: Float64Array[], X2: Float64Array[]): Float64Array[] {
    const K: Float64Array[] = Array.from(
      { length: X1.length },
      () => new Float64Array(X2.length),
    );
    for (let i = 0; i < X1.length; i++) {
      for (let j = 0; j < X2.length; j++) {
        let d = 0;
        const xi = X1[i]!;
        const xj = X2[j]!;
        for (let f = 0; f < xi.length; f++)
          d += ((xi[f] ?? 0) - (xj[f] ?? 0)) ** 2;
        K[i]![j] = Math.exp(-d / (2 * this.lengthScale ** 2));
      }
    }
    return K;
  }

  private _sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  fit(X: Float64Array[], y: Int32Array): this {
    this.XTrain_ = X;
    const classSet = new Set<number>();
    for (const c of y) classSet.add(c);
    this.classes_ = new Int32Array([...classSet].sort((a, b) => a - b));
    const n = X.length;
    const yBinary = new Float64Array(n);
    for (let i = 0; i < n; i++) yBinary[i] = y[i] === this.classes_[1] ? 1 : 0;
    // Laplace approximation
    const K = this._rbfKernel(X, X);
    const f = new Float64Array(n);
    for (let iter = 0; iter < this.maxIter; iter++) {
      const pi = f.map((v) => this._sigmoid(v));
      const W = pi.map((p) => p * (1 - p));
      const grad = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let Kf = 0;
        for (let j = 0; j < n; j++) Kf += (K[i]![j] ?? 0) * (f[j] ?? 0);
        grad[i] = (yBinary[i] ?? 0) - (pi[i] ?? 0) - Kf / (this.noiseLevel + 1);
      }
      for (let i = 0; i < n; i++) {
        const lr = 0.1 / (1 + iter * 0.01);
        f[i] = (f[i] ?? 0) + lr * (grad[i] ?? 0);
      }
      void W;
    }
    this.alpha_ = f;
    this.fitted = true;
    return this;
  }

  predictProba(X: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new Error("Not fitted");
    const K = this._rbfKernel(X, this.XTrain_);
    return K.map((kRow) => {
      let fStar = 0;
      for (let j = 0; j < this.XTrain_.length; j++)
        fStar += (kRow[j] ?? 0) * (this.alpha_[j] ?? 0);
      const p1 = this._sigmoid(fStar);
      return new Float64Array([1 - p1, p1]);
    });
  }

  predict(X: Float64Array[]): Int32Array {
    const proba = this.predictProba(X);
    return new Int32Array(
      proba.map(
        (p) => ((p[1] ?? 0) >= 0.5 ? this.classes_[1] : this.classes_[0]) ?? 0,
      ),
    );
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) correct++;
    return correct / Math.max(y.length, 1);
  }
}

/**
 * Self-Organizing Map (SOM) clustering.
 */

export class SelfOrganizingMap {
  private weights_!: Float64Array[];
  private fitted_ = false;

  constructor(
    private nX = 5,
    private nY = 5,
    private maxIter = 100,
    private learningRate = 0.5,
    private sigma = 1.0
  ) {}

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 1;
    const nNodes = this.nX * this.nY;
    this.weights_ = Array.from({ length: nNodes }, () =>
      new Float64Array(p).map(() => Math.random())
    );
    for (let iter = 0; iter < this.maxIter; iter++) {
      const lr = this.learningRate * Math.exp(-iter / this.maxIter);
      const sigma = this.sigma * Math.exp(-iter / this.maxIter);
      const xi = X[Math.floor(Math.random() * n)]!;
      const bmu = this._bestMatchingUnit(xi);
      const bx = bmu % this.nX, by = Math.floor(bmu / this.nX);
      for (let k = 0; k < nNodes; k++) {
        const kx = k % this.nX, ky = Math.floor(k / this.nX);
        const dist2 = (kx - bx) ** 2 + (ky - by) ** 2;
        const h = Math.exp(-dist2 / (2 * sigma * sigma));
        for (let j = 0; j < p; j++) {
          this.weights_[k]![j] = (this.weights_[k]![j] ?? 0) + lr * h * ((xi[j] ?? 0) - (this.weights_[k]![j] ?? 0));
        }
      }
    }
    this.fitted_ = true;
    return this;
  }

  private _bestMatchingUnit(x: Float64Array): number {
    let bestIdx = 0, bestDist = Number.POSITIVE_INFINITY;
    for (let k = 0; k < this.weights_.length; k++) {
      const w = this.weights_[k]!;
      const d = x.reduce((s, v, j) => s + (v - (w[j] ?? 0)) ** 2, 0);
      if (d < bestDist) { bestDist = d; bestIdx = k; }
    }
    return bestIdx;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted_) throw new Error('Not fitted');
    return new Int32Array(X.map(x => this._bestMatchingUnit(x)));
  }
}

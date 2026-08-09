/**
 * SVR (Support Vector Regression) and Structured SVM extensions.
 */

export class EpsilonSVR {
  private supportVectors_!: Float64Array[];
  private alphas_!: Float64Array;
  private bias_ = 0;
  private fitted_ = false;

  constructor(private C = 1.0, private epsilon = 0.1, private gamma: number | null = null, private maxIter = 200) {}

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length, p = X[0]?.length ?? 1;
    const g = this.gamma ?? 1 / p;
    const K = X.map(xi => new Float64Array(X.map(xj => Math.exp(-g * xi.reduce((s, v, j) => s + (v - (xj[j] ?? 0)) ** 2, 0)))));

    // SMO for SVR: dual variables alpha+ and alpha-
    const alphaPlus = new Float64Array(n);
    const alphaMinus = new Float64Array(n);
    let bias = 0;

    for (let iter = 0; iter < this.maxIter; iter++) {
      for (let i = 0; i < n; i++) {
        const fxi = K[i]!.reduce((s, Kij, j) => s + ((alphaPlus[j] ?? 0) - (alphaMinus[j] ?? 0)) * Kij, 0) + bias;
        const ri = fxi - (y[i] ?? 0);
        // Update alpha+
        const alphaOld = alphaPlus[i] ?? 0;
        if (ri > this.epsilon && alphaOld > 0) alphaPlus[i] = Math.max(0, alphaOld - (ri - this.epsilon) / ((K[i]![i] ?? 1) + 1));
        else if (ri < -this.epsilon && alphaOld < this.C) alphaPlus[i] = Math.min(this.C, alphaOld + (-this.epsilon - ri) / ((K[i]![i] ?? 1) + 1));
        // Update alpha-
        const alphaMinusOld = alphaMinus[i] ?? 0;
        if (-ri > this.epsilon && alphaMinusOld > 0) alphaMinus[i] = Math.max(0, alphaMinusOld - (-ri - this.epsilon) / ((K[i]![i] ?? 1) + 1));
        else if (-ri < -this.epsilon && alphaMinusOld < this.C) alphaMinus[i] = Math.min(this.C, alphaMinusOld + (-this.epsilon + ri) / ((K[i]![i] ?? 1) + 1));
      }
      // Update bias
      bias = 0;
      let count = 0;
      for (let i = 0; i < n; i++) {
        const ap = alphaPlus[i] ?? 0, am = alphaMinus[i] ?? 0;
        if (ap > 1e-5 && ap < this.C - 1e-5) {
          bias += (y[i] ?? 0) - this.epsilon - K[i]!.reduce((s, Kij, j) => s + ((alphaPlus[j] ?? 0) - (alphaMinus[j] ?? 0)) * Kij, 0);
          count++;
        } else if (am > 1e-5 && am < this.C - 1e-5) {
          bias += (y[i] ?? 0) + this.epsilon - K[i]!.reduce((s, Kij, j) => s + ((alphaPlus[j] ?? 0) - (alphaMinus[j] ?? 0)) * Kij, 0);
          count++;
        }
      }
      bias = count > 0 ? bias / count : 0;
    }

    const svIdx = Array.from({ length: n }, (_, i) => i).filter(i => Math.abs((alphaPlus[i] ?? 0) - (alphaMinus[i] ?? 0)) > 1e-5);
    this.supportVectors_ = svIdx.map(i => X[i]!);
    this.alphas_ = new Float64Array(svIdx.map(i => (alphaPlus[i] ?? 0) - (alphaMinus[i] ?? 0)));
    this.bias_ = bias;
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    const p = X[0]?.length ?? 1;
    const g = this.gamma ?? 1 / p;
    return new Float64Array(X.map(x =>
      this.supportVectors_.reduce((s, sv, k) => {
        const K_ = Math.exp(-g * x.reduce((ss, v, j) => ss + (v - (sv[j] ?? 0)) ** 2, 0));
        return s + (this.alphas_[k] ?? 0) * K_;
      }, this.bias_)
    ));
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    const yMean = Array.from(y).reduce((s, v) => s + v, 0) / y.length;
    const ssRes = pred.reduce((s, p, i) => s + ((y[i] ?? 0) - p) ** 2, 0);
    const ssTot = Array.from(y).reduce((s, v) => s + (v - yMean) ** 2, 0);
    return 1 - ssRes / (ssTot + 1e-10);
  }

  get supportVectors(): Float64Array[] { return this.supportVectors_; }
  get nSupportVectors(): number { return this.supportVectors_.length; }
}

export class OneclassSVM {
  private supportVectors_!: Float64Array[];
  private alphas_!: Float64Array;
  private rho_ = 0;
  private fitted_ = false;

  constructor(private nu = 0.5, private gamma: number | null = null, private maxIter = 200) {}

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 1;
    const g = this.gamma ?? 1 / p;
    const K = X.map(xi => new Float64Array(X.map(xj => Math.exp(-g * xi.reduce((s, v, j) => s + (v - (xj[j] ?? 0)) ** 2, 0)))));
    const nSV = Math.max(1, Math.floor(this.nu * n));
    const alphas = new Float64Array(n);
    for (let i = 0; i < nSV; i++) alphas[i] = 1 / nSV;
    // Project onto simplex with nu constraint
    const q = this.nu;
    let sumA = alphas.reduce((s, v) => s + v, 0);
    if (sumA > q) for (let i = 0; i < n; i++) alphas[i] = (alphas[i] ?? 0) * q / sumA;
    this.rho_ = K.reduce((s, row, i) => s + row.reduce((ss, v, j) => ss + (alphas[i] ?? 0) * (alphas[j] ?? 0) * v, 0), 0) * 0.5;
    const svIdx = Array.from({ length: n }, (_, i) => i).filter(i => (alphas[i] ?? 0) > 1e-5);
    this.supportVectors_ = svIdx.map(i => X[i]!);
    this.alphas_ = new Float64Array(svIdx.map(i => alphas[i] ?? 0));
    this.fitted_ = true;
    return this;
  }

  decisionFunction(X: Float64Array[]): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    const p = X[0]?.length ?? 1;
    const g = this.gamma ?? 1 / p;
    return new Float64Array(X.map(x => this.supportVectors_.reduce((s, sv, k) => {
      const K_ = Math.exp(-g * x.reduce((ss, v, j) => ss + (v - (sv[j] ?? 0)) ** 2, 0));
      return s + (this.alphas_[k] ?? 0) * K_;
    }, -this.rho_)));
  }

  predict(X: Float64Array[]): Int32Array {
    const dec = this.decisionFunction(X);
    return new Int32Array(dec.map(v => v >= 0 ? 1 : -1));
  }
}

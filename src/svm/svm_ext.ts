/**
 * SVM extensions: OneClassSVM, NuSVR extensions, sequential minimal optimization utilities.
 */

export class OneClassSVM {
  private nu: number;
  private kernel: string;
  private gamma: number | string;
  private supportVectors: Float64Array | null = null;
  private dualCoef: Float64Array | null = null;
  private intercept = 0;
  private nFeatures = 0;

  constructor(params: { nu?: number; kernel?: string; gamma?: number | string } = {}) {
    this.nu = params.nu ?? 0.5;
    this.kernel = params.kernel ?? "rbf";
    this.gamma = params.gamma ?? "scale";
  }

  fit(X: Float64Array[], _y?: unknown): this {
    this.nFeatures = X[0]?.length ?? 0;
    const n = X.length;
    // Simplified: store subset of support vectors
    const sv: number[] = [];
    for (let i = 0; i < Math.min(Math.ceil(this.nu * n), n); i++) {
      const row = X[i];
      if (row !== undefined) {
        for (const v of row) sv.push(v);
      }
    }
    this.supportVectors = new Float64Array(sv);
    this.dualCoef = new Float64Array(Math.ceil(this.nu * n)).fill(1 / Math.ceil(this.nu * n));
    this.intercept = 0;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (this.supportVectors === null) throw new Error("Not fitted");
    const result = new Int32Array(X.length);
    for (let i = 0; i < X.length; i++) {
      result[i] = this._decisionFunction(X[i] ?? new Float64Array(0)) >= 0 ? 1 : -1;
    }
    return result;
  }

  private _decisionFunction(x: Float64Array): number {
    if (this.supportVectors === null || this.dualCoef === null) return 0;
    const nSV = this.dualCoef.length;
    let score = this.intercept;
    for (let j = 0; j < nSV; j++) {
      const sv = this.supportVectors.subarray(j * this.nFeatures, (j + 1) * this.nFeatures);
      const k = this._rbfKernel(x, sv);
      score += (this.dualCoef[j] ?? 0) * k;
    }
    return score;
  }

  private _rbfKernel(a: Float64Array, b: Float64Array): number {
    const g = typeof this.gamma === "number" ? this.gamma : 1 / Math.max(this.nFeatures, 1);
    let d = 0;
    for (let i = 0; i < a.length; i++) d += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
    return Math.exp(-g * d);
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) correct++;
    return correct / Math.max(y.length, 1);
  }
}

export class SMOSolver {
  private alpha: Float64Array;
  private b = 0;
  private X: Float64Array[];
  private y: Float64Array;
  private C: number;
  private tol: number;
  private maxIter: number;

  constructor(params: { C?: number; tol?: number; maxIter?: number } = {}) {
    this.C = params.C ?? 1.0;
    this.tol = params.tol ?? 1e-3;
    this.maxIter = params.maxIter ?? 100;
    this.alpha = new Float64Array(0);
    this.X = [];
    this.y = new Float64Array(0);
  }

  fit(X: Float64Array[], y: Float64Array): this {
    this.X = X;
    this.y = y;
    const n = X.length;
    this.alpha = new Float64Array(n);
    this.b = 0;

    let iter = 0;
    while (iter < this.maxIter) {
      let changed = 0;
      for (let i = 0; i < n; i++) {
        const ei = this._error(i);
        if ((y[i]! * ei < -this.tol && this.alpha[i]! < this.C) ||
            (y[i]! * ei > this.tol && this.alpha[i]! > 0)) {
          const j = (i + 1) % n;
          const ej = this._error(j);
          const ai0 = this.alpha[i]!;
          const aj0 = this.alpha[j]!;
          const L = y[i] === y[j] ? Math.max(0, aj0 + ai0 - this.C) : Math.max(0, aj0 - ai0);
          const H = y[i] === y[j] ? Math.min(this.C, aj0 + ai0) : Math.min(this.C, this.C + aj0 - ai0);
          if (L >= H) continue;
          const kij = this._kernel(X[i]!, X[j]!);
          const kii = this._kernel(X[i]!, X[i]!);
          const kjj = this._kernel(X[j]!, X[j]!);
          const eta = 2 * kij - kii - kjj;
          if (eta >= 0) continue;
          let ajNew = aj0 - (y[j]! * (ei - ej)) / eta;
          ajNew = Math.max(L, Math.min(H, ajNew));
          if (Math.abs(ajNew - aj0) < 1e-5) continue;
          const aiNew = ai0 + (y[i]! * y[j]!) * (aj0 - ajNew);
          this.alpha[i] = aiNew;
          this.alpha[j] = ajNew;
          this.b -= ei + y[i]! * (aiNew - ai0) * kii + y[j]! * (ajNew - aj0) * kij;
          changed++;
        }
      }
      if (changed === 0) break;
      iter++;
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    return new Float64Array(X.map((x) => this._decision(x)));
  }

  private _error(i: number): number {
    return this._decision(this.X[i]!) - (this.y[i] ?? 0);
  }

  private _decision(x: Float64Array): number {
    let sum = this.b;
    for (let i = 0; i < this.X.length; i++) {
      sum += (this.alpha[i] ?? 0) * (this.y[i] ?? 0) * this._kernel(this.X[i]!, x);
    }
    return sum;
  }

  private _kernel(a: Float64Array, b: Float64Array): number {
    let d = 0;
    for (let i = 0; i < a.length; i++) d += (a[i] ?? 0) * (b[i] ?? 0);
    return d;
  }
}

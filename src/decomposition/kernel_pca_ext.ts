/**
 * Extended Kernel PCA utilities.
 * Mirrors scikit-learn's decomposition.KernelPCA with additional kernel support.
 */

export type KernelType = "rbf" | "polynomial" | "sigmoid" | "cosine" | "linear" | "laplacian";

export interface KernelPCAExtOptions {
  nComponents?: number;
  kernel?: KernelType;
  gamma?: number;
  degree?: number;
  coef0?: number;
  fitInverseTransform?: boolean;
  alpha?: number;
}

export class KernelPCAExt {
  readonly nComponents: number;
  readonly kernel: KernelType;
  readonly gamma: number | null;
  readonly degree: number;
  readonly coef0: number;
  readonly fitInverseTransform: boolean;
  readonly alpha: number;

  private _alphas: Float64Array[] | null = null;
  private _lambdas: Float64Array | null = null;
  private _XFit: Float64Array[] | null = null;

  constructor(options: KernelPCAExtOptions = {}) {
    this.nComponents = options.nComponents ?? 2;
    this.kernel = options.kernel ?? "rbf";
    this.gamma = options.gamma ?? null;
    this.degree = options.degree ?? 3;
    this.coef0 = options.coef0 ?? 1;
    this.fitInverseTransform = options.fitInverseTransform ?? false;
    this.alpha = options.alpha ?? 1.0;
  }

  private _computeKernel(X: Float64Array[], Y: Float64Array[]): Float64Array[] {
    const n = X.length;
    const m = Y.length;
    const gamma = this.gamma ?? (X[0] !== undefined ? 1 / X[0].length : 1);
    const K: Float64Array[] = Array.from({ length: n }, () => new Float64Array(m));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        const xi = X[i]!;
        const yj = Y[j]!;
        let val = 0;
        switch (this.kernel) {
          case "rbf": {
            let d = 0;
            for (let k = 0; k < xi.length; k++) d += ((xi[k] ?? 0) - (yj[k] ?? 0)) ** 2;
            val = Math.exp(-gamma * d);
            break;
          }
          case "polynomial": {
            let dot = 0;
            for (let k = 0; k < xi.length; k++) dot += (xi[k] ?? 0) * (yj[k] ?? 0);
            val = (gamma * dot + this.coef0) ** this.degree;
            break;
          }
          case "sigmoid": {
            let dot = 0;
            for (let k = 0; k < xi.length; k++) dot += (xi[k] ?? 0) * (yj[k] ?? 0);
            val = Math.tanh(gamma * dot + this.coef0);
            break;
          }
          case "cosine": {
            let dot = 0, ni = 0, nj = 0;
            for (let k = 0; k < xi.length; k++) {
              dot += (xi[k] ?? 0) * (yj[k] ?? 0);
              ni += (xi[k] ?? 0) ** 2;
              nj += (yj[k] ?? 0) ** 2;
            }
            val = dot / (Math.sqrt(ni * nj) + 1e-10);
            break;
          }
          case "laplacian": {
            let d = 0;
            for (let k = 0; k < xi.length; k++) d += Math.abs((xi[k] ?? 0) - (yj[k] ?? 0));
            val = Math.exp(-gamma * d);
            break;
          }
          default: {
            // linear
            let dot = 0;
            for (let k = 0; k < xi.length; k++) dot += (xi[k] ?? 0) * (yj[k] ?? 0);
            val = dot;
          }
        }
        K[i]![j] = val;
      }
    }
    return K;
  }

  private _centerKernel(K: Float64Array[]): Float64Array[] {
    const n = K.length;
    const m = K[0]?.length ?? 0;
    const rowMeans = K.map((row) => Array.from(row).reduce((s, v) => s + v, 0) / m);
    const colMeans = Array.from({ length: m }, (_, j) =>
      K.reduce((s, row) => s + (row[j] ?? 0), 0) / n,
    );
    const totalMean = rowMeans.reduce((s, v) => s + v, 0) / n;
    return K.map((row, i) =>
      Float64Array.from(row, (v, j) => v - (rowMeans[i] ?? 0) - (colMeans[j] ?? 0) + totalMean),
    );
  }

  fit(X: Float64Array[]): this {
    this._XFit = X;
    const K = this._centerKernel(this._computeKernel(X, X));
    // Power iteration for top eigenvalues (simplified)
    const n = K.length;
    const nc = Math.min(this.nComponents, n);
    const alphas: Float64Array[] = [];
    const lambdas: number[] = [];

    for (let c = 0; c < nc; c++) {
      let v = new Float64Array(n).fill(1 / Math.sqrt(n));
      let lambda = 0;
      for (let iter = 0; iter < 100; iter++) {
        const Kv = K.map((row) =>
          row.reduce((s, val, j) => s + val * (v[j] ?? 0), 0),
        );
        lambda = Math.sqrt(Kv.reduce((s, x) => s + x * x, 0));
        if (lambda < 1e-10) break;
        v = Float64Array.from(Kv, (x) => x / lambda);
      }
      alphas.push(v);
      lambdas.push(lambda);

      // Deflate
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          K[i]![j] = (K[i]![j] ?? 0) - lambda * (v[i] ?? 0) * (v[j] ?? 0);
        }
      }
    }

    this._alphas = alphas;
    this._lambdas = new Float64Array(lambdas);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this._XFit === null || this._alphas === null || this._lambdas === null) {
      throw new Error("KernelPCAExt must be fitted before transform");
    }
    const K = this._centerKernel(this._computeKernel(X, this._XFit));
    const nc = this._alphas.length;
    return K.map((row) => {
      const result = new Float64Array(nc);
      for (let c = 0; c < nc; c++) {
        const alpha = this._alphas![c]!;
        const lambda = this._lambdas![c] ?? 1;
        result[c] = row.reduce((s, v, j) => s + v * (alpha[j] ?? 0), 0) / Math.sqrt(lambda);
      }
      return result;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

/**
 * Kernel SVM extensions: polynomial kernel, sigmoid kernel, and multi-class SVM.
 */

export function polynomialKernel(X1: Float64Array[], X2: Float64Array[], degree = 3, coef0 = 1, gamma: number | null = null): Float64Array[] {
  const n1 = X1.length, n2 = X2.length;
  const p = X1[0]?.length ?? 1;
  const g = gamma ?? 1 / p;
  return Array.from({ length: n1 }, (_, i) =>
    new Float64Array(n2).map((_, j) =>
      (g * X1[i]!.reduce((s, v, k) => s + v * (X2[j]![k] ?? 0), 0) + coef0) ** degree
    )
  );
}

export function sigmoidKernel(X1: Float64Array[], X2: Float64Array[], gamma: number | null = null, coef0 = 0): Float64Array[] {
  const n1 = X1.length, n2 = X2.length;
  const p = X1[0]?.length ?? 1;
  const g = gamma ?? 1 / p;
  return Array.from({ length: n1 }, (_, i) =>
    new Float64Array(n2).map((_, j) =>
      Math.tanh(g * X1[i]!.reduce((s, v, k) => s + v * (X2[j]![k] ?? 0), 0) + coef0)
    )
  );
}

export function laplacianKernel(X1: Float64Array[], X2: Float64Array[], gamma: number | null = null): Float64Array[] {
  const n1 = X1.length, n2 = X2.length;
  const p = X1[0]?.length ?? 1;
  const g = gamma ?? 1 / p;
  return Array.from({ length: n1 }, (_, i) =>
    new Float64Array(n2).map((_, j) => {
      const dist = X1[i]!.reduce((s, v, k) => s + Math.abs(v - (X2[j]![k] ?? 0)), 0);
      return Math.exp(-g * dist);
    })
  );
}

export class MultiClassSVM {
  private svms_: Array<{ supportVectors: Float64Array[]; alphas: Float64Array; bias: number; classes: [number, number] }> = [];
  private classes_!: number[];
  private fitted_ = false;

  constructor(private C = 1.0, private kernel: 'rbf' | 'linear' | 'poly' = 'rbf', private gamma: number | null = null, private maxIter = 200) {}

  fit(X: Float64Array[], y: Int32Array): this {
    this.classes_ = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
    this.svms_ = [];
    // One-vs-one decomposition
    for (let i = 0; i < this.classes_.length; i++) {
      for (let j = i + 1; j < this.classes_.length; j++) {
        const ci = this.classes_[i]!, cj = this.classes_[j]!;
        const idx = Array.from({ length: X.length }, (_, k) => k).filter(k => y[k] === ci || y[k] === cj);
        const Xbinary = idx.map(k => X[k]!);
        const yBinary = new Int32Array(idx.map(k => y[k] === ci ? 1 : -1));
        const svm = this._fitBinary(Xbinary, yBinary);
        this.svms_.push({ ...svm, classes: [ci, cj] });
      }
    }
    this.fitted_ = true;
    return this;
  }

  private _fitBinary(X: Float64Array[], y: Int32Array): { supportVectors: Float64Array[]; alphas: Float64Array; bias: number } {
    const n = X.length, p = X[0]?.length ?? 1;
    const g = this.gamma ?? 1 / p;
    const K = this._computeKernel(X, X, g);
    const alphas = new Float64Array(n);
    let bias = 0;
    // SMO-simplified
    for (let iter = 0; iter < this.maxIter; iter++) {
      for (let i = 0; i < n; i++) {
        const Ei = K[i]!.reduce((s, Kij, j) => s + (alphas[j] ?? 0) * (y[j] ?? 0) * Kij, 0) + bias - (y[i] ?? 0);
        if (((y[i] ?? 0) * Ei < -0.001 && (alphas[i] ?? 0) < this.C) || ((y[i] ?? 0) * Ei > 0.001 && (alphas[i] ?? 0) > 0)) {
          const j = (i + 1 + Math.floor(Math.random() * (n - 1))) % n;
          const Ej = K[j]!.reduce((s, Kjl, l) => s + (alphas[l] ?? 0) * (y[l] ?? 0) * Kjl, 0) + bias - (y[j] ?? 0);
          const alphaIOld = alphas[i] ?? 0, alphaJOld = alphas[j] ?? 0;
          const L = (y[i] ?? 0) !== (y[j] ?? 0) ? Math.max(0, alphaJOld - alphaIOld) : Math.max(0, alphaIOld + alphaJOld - this.C);
          const H = (y[i] ?? 0) !== (y[j] ?? 0) ? Math.min(this.C, this.C + alphaJOld - alphaIOld) : Math.min(this.C, alphaIOld + alphaJOld);
          const eta = 2 * (K[i]![j] ?? 0) - (K[i]![i] ?? 0) - (K[j]![j] ?? 0);
          if (Math.abs(eta) < 1e-10) continue;
          alphas[j] = Math.max(L, Math.min(H, alphaJOld - (y[j] ?? 0) * (Ei - Ej) / eta));
          alphas[i] = alphaIOld + (y[i] ?? 0) * (y[j] ?? 0) * (alphaJOld - (alphas[j] ?? 0));
          bias -= (Ei + (y[i] ?? 0) * (K[i]![i] ?? 0) * ((alphas[i] ?? 0) - alphaIOld) + (y[j] ?? 0) * (K[j]![i] ?? 0) * ((alphas[j] ?? 0) - alphaJOld));
        }
      }
    }
    const svIdx = Array.from({ length: n }, (_, i) => i).filter(i => (alphas[i] ?? 0) > 1e-5);
    return { supportVectors: svIdx.map(i => X[i]!), alphas: new Float64Array(svIdx.map(i => (alphas[i] ?? 0) * (y[i] ?? 0))), bias };
  }

  private _computeKernel(X1: Float64Array[], X2: Float64Array[], g: number): Float64Array[] {
    return X1.map(x1 => new Float64Array(X2.map(x2 => {
      if (this.kernel === 'linear') return x1.reduce((s, v, j) => s + v * (x2[j] ?? 0), 0);
      const d2 = x1.reduce((s, v, j) => s + (v - (x2[j] ?? 0)) ** 2, 0);
      return Math.exp(-g * d2);
    })));
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted_) throw new Error('Not fitted');
    const p = X[0]?.length ?? 1;
    const g = this.gamma ?? 1 / p;
    return new Int32Array(X.map(x => {
      const votes = new Map<number, number>(this.classes_.map(c => [c, 0]));
      for (const svm of this.svms_) {
        const dec = svm.supportVectors.reduce((s, sv, k) => {
          const K_ = this.kernel === 'linear' ? x.reduce((ss, v, j) => ss + v * (sv[j] ?? 0), 0) : Math.exp(-g * x.reduce((ss, v, j) => ss + (v - (sv[j] ?? 0)) ** 2, 0));
          return s + (svm.alphas[k] ?? 0) * K_;
        }, svm.bias);
        const winner = dec >= 0 ? svm.classes[0] : svm.classes[1];
        votes.set(winner, (votes.get(winner) ?? 0) + 1);
      }
      return [...votes.entries()].reduce((best, [c, v]) => v > best.v ? { c, v } : best, { c: 0, v: -1 }).c;
    }));
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    return pred.filter((v, i) => v === y[i]).length / pred.length;
  }
}

/**
 * Decomposition base and DictionaryLearning port.
 */

export abstract class DecompositionMixin {
  abstract fit(X: Float64Array[]): this;
  abstract transform(X: Float64Array[]): Float64Array[];

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  abstract get nComponents(): number;
}

export class DictionaryLearning extends DecompositionMixin {
  private _nComponents: number;
  alpha: number;
  maxIter: number;
  tol: number;
  fitAlgorithm: "lars" | "cd";
  transformAlgorithm: "omp" | "lasso_lars" | "lasso_cd" | "threshold";
  nNonzeroCoefs: number;
  components_: Float64Array[] | null = null;
  error_: number[] = [];
  nIter_: number = 0;

  constructor(
    nComponents = 8,
    alpha = 1,
    maxIter = 1000,
    tol = 1e-8,
    fitAlgorithm: "lars" | "cd" = "lars",
    transformAlgorithm: "omp" | "lasso_lars" | "lasso_cd" | "threshold" = "omp",
    nNonzeroCoefs = 1,
  ) {
    super();
    this._nComponents = nComponents;
    this.alpha = alpha;
    this.maxIter = maxIter;
    this.tol = tol;
    this.fitAlgorithm = fitAlgorithm;
    this.transformAlgorithm = transformAlgorithm;
    this.nNonzeroCoefs = nNonzeroCoefs;
  }

  get nComponents(): number {
    return this._nComponents;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const k = this._nComponents;

    // Initialize dictionary from random data samples
    this.components_ = Array.from({ length: k }, (_, i) => {
      const idx = Math.floor(Math.random() * n);
      const row = X[idx] as Float64Array;
      const norm = Math.sqrt(row.reduce((s, v) => s + v * v, 0)) || 1;
      return row.map((v) => v / norm);
    });

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Sparse coding step: find sparse codes for each sample
      const codes = X.map((x) => this._sparseCoding(x, this.components_ as Float64Array[]));

      // Dictionary update step
      const A = Array.from({ length: k }, () => new Float64Array(k));
      const B: Float64Array[] = Array.from({ length: p }, () => new Float64Array(k));
      for (let i = 0; i < n; i++) {
        const xi = X[i] as Float64Array;
        const ci = codes[i] as Float64Array;
        for (let j = 0; j < k; j++) {
          for (let l = 0; l < k; l++) (A[j]! as Float64Array)[l]! += (ci[j] ?? 0) * (ci[l] ?? 0) / n;
          for (let f = 0; f < p; f++) (B[f]! as Float64Array)[j]! += (xi[f] ?? 0) * (ci[j] ?? 0) / n;
        }
      }

      // Update each atom
      let maxChange = 0;
      for (let j = 0; j < k; j++) {
        const a_jj = (A[j] as Float64Array)[j] ?? 1e-8;
        const newAtom = new Float64Array(p);
        for (let f = 0; f < p; f++) {
          let s = (B[f] as Float64Array)[j] ?? 0;
          for (let l = 0; l < k; l++) {
            if (l !== j) s -= ((A[j] as Float64Array)[l] ?? 0) * ((this.components_![l] as Float64Array)[f] ?? 0);
          }
          newAtom[f] = s / a_jj;
        }
        const norm = Math.sqrt(newAtom.reduce((s, v) => s + v * v, 0)) || 1;
        const normalized = newAtom.map((v) => v / norm);
        const diff = normalized.reduce((s, v, f) => s + (v - ((this.components_![j] as Float64Array)[f] ?? 0)) ** 2, 0);
        maxChange = Math.max(maxChange, diff);
        this.components_![j] = normalized;
      }

      const err = X.reduce((s, x, i) => {
        const code = codes[i] as Float64Array;
        let residualNorm = 0;
        for (let f = 0; f < p; f++) {
          let approx = 0;
          for (let j = 0; j < k; j++) approx += (code[j] ?? 0) * ((this.components_![j] as Float64Array)[f] ?? 0);
          residualNorm += ((x[f] ?? 0) - approx) ** 2;
        }
        return s + residualNorm;
      }, 0) / n;
      this.error_.push(err);
      this.nIter_ = iter + 1;
      if (maxChange < this.tol) break;
    }
    return this;
  }

  private _sparseCoding(x: Float64Array, D: Float64Array[]): Float64Array {
    const k = D.length;
    const code = new Float64Array(k);
    if (this.transformAlgorithm === "threshold") {
      for (let j = 0; j < k; j++) {
        const corr = x.reduce((s, v, f) => s + (v * (D[j]?.[f] ?? 0)), 0);
        code[j] = Math.max(0, Math.abs(corr) - this.alpha) * Math.sign(corr);
      }
    } else {
      // OMP greedy
      const nNonzero = Math.min(this.nNonzeroCoefs, k);
      const residual = new Float64Array(x);
      const selected = new Set<number>();
      for (let step = 0; step < nNonzero; step++) {
        let bestJ = 0, bestCorr = -1;
        for (let j = 0; j < k; j++) {
          if (selected.has(j)) continue;
          const c = Math.abs(residual.reduce((s, v, f) => s + v * (D[j]?.[f] ?? 0), 0));
          if (c > bestCorr) { bestCorr = c; bestJ = j; }
        }
        selected.add(bestJ);
        // Update residual
        const atom = D[bestJ] as Float64Array;
        const corr = residual.reduce((s, v, f) => s + v * (atom[f] ?? 0), 0);
        code[bestJ] = corr;
        for (let f = 0; f < residual.length; f++) residual[f]! -= corr * (atom[f] ?? 0);
      }
    }
    return code;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_) throw new Error("Not fitted");
    return X.map((x) => this._sparseCoding(x, this.components_ as Float64Array[]));
  }
}

export class MiniBatchDictionaryLearning extends DictionaryLearning {
  batchSize: number;
  nIterNoChange: number;

  constructor(
    nComponents = 8,
    alpha = 1,
    maxIter = 1000,
    batchSize = 256,
    tol = 1e-8,
    nIterNoChange = 5,
  ) {
    super(nComponents, alpha, maxIter, tol);
    this.batchSize = batchSize;
    this.nIterNoChange = nIterNoChange;
  }
}

export class SparseCoder {
  dictionary: Float64Array[];
  transformAlgorithm: "omp" | "threshold";
  nNonzeroCoefs: number;
  transformAlpha: number;

  constructor(
    dictionary: Float64Array[],
    transformAlgorithm: "omp" | "threshold" = "omp",
    nNonzeroCoefs = 1,
    transformAlpha = 0.1,
  ) {
    this.dictionary = dictionary;
    this.transformAlgorithm = transformAlgorithm;
    this.nNonzeroCoefs = nNonzeroCoefs;
    this.transformAlpha = transformAlpha;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const k = this.dictionary.length;
    return X.map((x) => {
      const code = new Float64Array(k);
      if (this.transformAlgorithm === "threshold") {
        for (let j = 0; j < k; j++) {
          const corr = x.reduce((s, v, f) => s + v * (this.dictionary[j]?.[f] ?? 0), 0);
          code[j] = Math.max(0, Math.abs(corr) - this.transformAlpha) * Math.sign(corr);
        }
      } else {
        const nNonzero = Math.min(this.nNonzeroCoefs, k);
        const residual = new Float64Array(x);
        const selected = new Set<number>();
        for (let step = 0; step < nNonzero; step++) {
          let bestJ = 0, bestCorr = -1;
          for (let j = 0; j < k; j++) {
            if (selected.has(j)) continue;
            const c = Math.abs(residual.reduce((s, v, f) => s + v * (this.dictionary[j]?.[f] ?? 0), 0));
            if (c > bestCorr) { bestCorr = c; bestJ = j; }
          }
          selected.add(bestJ);
          const atom = this.dictionary[bestJ] as Float64Array;
          const corr = residual.reduce((s, v, f) => s + v * (atom[f] ?? 0), 0);
          code[bestJ] = corr;
          for (let f = 0; f < residual.length; f++) residual[f]! -= corr * (atom[f] ?? 0);
        }
      }
      return code;
    });
  }

  fit(_X: Float64Array[]): this {
    return this;
  }
}

/**
 * Dictionary Learning and Sparse Coding decomposition.
 */

export class DictionaryLearning {
  private components_!: Float64Array[];
  private fitted_ = false;

  constructor(
    private nComponents = 10,
    private alpha = 1.0,
    private maxIter = 100,
    private tol = 1e-8,
    private fitAlgorithm: 'lars' | 'cd' = 'cd'
  ) {}

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 1, k = this.nComponents;
    // Initialize dictionary
    this.components_ = Array.from({ length: k }, (_, i) =>
      new Float64Array(X[i % n]!).map((v, j) => v / (Math.sqrt(X[i % n]!.reduce((s, vv) => s + vv * vv, 0)) + 1e-10))
    );

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Sparse coding step: solve min_{code} ||X - code @ D||^2 + alpha * ||code||_1
      const codes = this._sparseCoding(X);
      // Dictionary update step
      const prevComps = this.components_.map(c => new Float64Array(c));
      for (let k_ = 0; k_ < k; k_++) {
        const rk = Array.from({ length: n }, (_, r) => {
          let res = new Float64Array(X[r]!);
          for (let l = 0; l < k; l++) {
            if (l === k_) continue;
            for (let j = 0; j < p; j++) res[j] = (res[j] ?? 0) - (codes[r]![l] ?? 0) * (this.components_[l]![j] ?? 0);
          }
          return res;
        });
        const denom = n > 0 ? codes.reduce((s, c) => s + (c[k_] ?? 0) ** 2, 0) : 1;
        if (denom < 1e-10) continue;
        const newComp = new Float64Array(p);
        for (let r = 0; r < n; r++) for (let j = 0; j < p; j++) newComp[j] = (newComp[j] ?? 0) + (codes[r]![k_] ?? 0) * (rk[r]![j] ?? 0);
        // Normalize
        const norm = Math.sqrt(newComp.reduce((s, v) => s + v * v, 0)) + 1e-10;
        for (let j = 0; j < p; j++) this.components_[k_]![j] = (newComp[j] ?? 0) / norm;
      }
      // Check convergence
      const diff = this.components_.reduce((s, c, i) => s + c.reduce((ss, v, j) => ss + (v - (prevComps[i]![j] ?? 0)) ** 2, 0), 0);
      if (diff < this.tol) break;
    }
    this.fitted_ = true;
    return this;
  }

  private _sparseCoding(X: Float64Array[]): Float64Array[] {
    const n = X.length, k = this.nComponents;
    return X.map(x => {
      const code = new Float64Array(k);
      // Coordinate descent for lasso
      for (let iter = 0; iter < 100; iter++) {
        let changed = false;
        for (let i = 0; i < k; i++) {
          const ri = x.reduce((s, v, j) => s + v * (this.components_[i]![j] ?? 0), 0)
            - code.reduce((s, c, l) => s + c * this.components_[i]!.reduce((ss, v, j) => ss + v * (this.components_[l]![j] ?? 0), 0), 0)
            + (code[i] ?? 0);
          const newC = Math.sign(ri) * Math.max(Math.abs(ri) - this.alpha, 0);
          if (Math.abs(newC - (code[i] ?? 0)) > 1e-8) changed = true;
          code[i] = newC;
        }
        if (!changed) break;
      }
      return code;
    });
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return this._sparseCoding(X);
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.fit(X).transform(X); }
  get components(): Float64Array[] { return this.components_; }
}

export class MiniBatchDictionaryLearning {
  private components_!: Float64Array[];
  private fitted_ = false;

  constructor(private nComponents = 10, private alpha = 1.0, private batchSize = 64, private maxIter = 1000) {}

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 1, k = this.nComponents;
    this.components_ = Array.from({ length: k }, (_, i) =>
      new Float64Array(X[i % n]!).map((v, j) => v / (Math.sqrt(X[i % n]!.reduce((s, vv) => s + vv * vv, 0)) + 1e-10))
    );
    const A = Array.from({ length: k }, () => new Float64Array(k));
    const B = Array.from({ length: k }, () => new Float64Array(p));
    for (let iter = 0; iter < this.maxIter; iter++) {
      const batch = Array.from({ length: this.batchSize }, () => X[Math.floor(Math.random() * n)]!);
      const codes = batch.map(x => this._lasso(x));
      // Update A and B
      for (let r = 0; r < batch.length; r++) {
        for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) A[i]![j] = (A[i]![j] ?? 0) + (codes[r]![i] ?? 0) * (codes[r]![j] ?? 0);
        for (let i = 0; i < k; i++) for (let j = 0; j < p; j++) B[i]![j] = (B[i]![j] ?? 0) + (codes[r]![i] ?? 0) * (batch[r]![j] ?? 0);
      }
      // Update dictionary column by column
      for (let i = 0; i < k; i++) {
        const ai = A[i]![i] ?? 1e-10;
        const newComp = new Float64Array(p);
        for (let j = 0; j < p; j++) {
          let val = (B[i]![j] ?? 0);
          for (let l = 0; l < k; l++) if (l !== i) val -= (A[i]![l] ?? 0) * (this.components_[l]![j] ?? 0);
          newComp[j] = val / (ai + 1e-10);
        }
        const norm = Math.sqrt(newComp.reduce((s, v) => s + v * v, 0));
        if (norm > 1) for (let j = 0; j < p; j++) this.components_[i]![j] = (newComp[j] ?? 0) / norm;
        else for (let j = 0; j < p; j++) this.components_[i]![j] = newComp[j] ?? 0;
      }
    }
    this.fitted_ = true;
    return this;
  }

  private _lasso(x: Float64Array): Float64Array {
    const k = this.nComponents, code = new Float64Array(k);
    for (let iter = 0; iter < 50; iter++) {
      for (let i = 0; i < k; i++) {
        const ri = x.reduce((s, v, j) => s + v * (this.components_[i]![j] ?? 0), 0)
          - code.reduce((s, c, l) => s + c * this.components_[i]!.reduce((ss, v, j) => ss + v * (this.components_[l]![j] ?? 0), 0), 0)
          + (code[i] ?? 0);
        code[i] = Math.sign(ri) * Math.max(Math.abs(ri) - this.alpha, 0);
      }
    }
    return code;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return X.map(x => this._lasso(x));
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.fit(X).transform(X); }
  get components(): Float64Array[] { return this.components_; }
}

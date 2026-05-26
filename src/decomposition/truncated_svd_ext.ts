/**
 * TruncatedSVD extensions: streaming SVD, randomized SVD utilities.
 */

export class TruncatedSVDExtended {
  private components_: Float64Array[] = [];
  private singularValues_: Float64Array = new Float64Array(0);
  private explainedVariance_: Float64Array = new Float64Array(0);
  private explainedVarianceRatio_: Float64Array = new Float64Array(0);
  private mean_: Float64Array = new Float64Array(0);
  private fitted = false;

  constructor(
    private readonly nComponents = 2,
    private readonly nIter = 5,
    private readonly nOversamples = 10,
    private readonly centerData = false
  ) {}

  fit(X: Float64Array[]): this {
    const n = X.length;
    const nF = X[0]?.length ?? 1;
    let data = X;
    if (this.centerData) {
      this.mean_ = new Float64Array(nF);
      for (const x of X) for (let f = 0; f < nF; f++) this.mean_[f] = (this.mean_[f] ?? 0) + (x[f] ?? 0) / n;
      data = X.map((x) => new Float64Array(x.map((v, f) => v - (this.mean_[f] ?? 0))));
    }
    const { U, S, Vt } = this._randomizedSVD(data, this.nComponents);
    this.components_ = Vt;
    this.singularValues_ = S;
    const totalVar = S.reduce((s, v) => s + v ** 2, 0) / n;
    this.explainedVariance_ = new Float64Array(S.map((s) => s ** 2 / n));
    this.explainedVarianceRatio_ = new Float64Array(S.map((s) => s ** 2 / n / Math.max(totalVar, 1e-10)));
    void U;
    this.fitted = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new Error("Not fitted");
    return X.map((x) => {
      const v = this.centerData ? x.map((val, f) => val - (this.mean_[f] ?? 0)) : x;
      return new Float64Array(this.components_.map((comp) => {
        let dot = 0;
        for (let f = 0; f < v.length; f++) dot += (v[f] ?? 0) * (comp[f] ?? 0);
        return dot;
      }));
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.fit(X).transform(X); }

  inverseTransform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new Error("Not fitted");
    const nF = this.components_[0]?.length ?? 1;
    return X.map((x) => {
      const result = new Float64Array(nF);
      for (let k = 0; k < x.length; k++) {
        const comp = this.components_[k]!;
        for (let f = 0; f < nF; f++) result[f] = (result[f] ?? 0) + (x[k] ?? 0) * (comp[f] ?? 0);
      }
      if (this.centerData) for (let f = 0; f < nF; f++) result[f] = (result[f] ?? 0) + (this.mean_[f] ?? 0);
      return result;
    });
  }

  private _randomizedSVD(X: Float64Array[], k: number): { U: Float64Array[]; S: Float64Array; Vt: Float64Array[] } {
    const n = X.length;
    const nF = X[0]?.length ?? 1;
    const l = k + this.nOversamples;
    // Random projection
    let Q = Array.from({ length: l }, () => new Float64Array(nF).map(() => this._randn()));
    // Power iteration
    for (let iter = 0; iter < this.nIter; iter++) {
      // Q = X^T X Q
      Q = Q.map((q) => {
        const xq = new Float64Array(n);
        for (let i = 0; i < n; i++) for (let f = 0; f < nF; f++) xq[i] = (xq[i] ?? 0) + (X[i]![f] ?? 0) * (q[f] ?? 0);
        const result = new Float64Array(nF);
        for (let i = 0; i < n; i++) for (let f = 0; f < nF; f++) result[f] = (result[f] ?? 0) + (X[i]![f] ?? 0) * (xq[i] ?? 0);
        return result;
      });
    }
    // QR decomposition (simplified Gram-Schmidt)
    Q = this._gramSchmidt(Q);
    // Project X onto Q
    const B = Q.map((q) => {
      const proj = new Float64Array(n);
      for (let i = 0; i < n; i++) for (let f = 0; f < nF; f++) proj[i] = (proj[i] ?? 0) + (X[i]![f] ?? 0) * (q[f] ?? 0);
      return proj;
    });
    // Compute singular values as norms
    const S = new Float64Array(k);
    const Vt: Float64Array[] = [];
    for (let j = 0; j < k; j++) {
      const b = B[j]!;
      const norm = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
      S[j] = norm;
      Vt.push(new Float64Array(Q[j] ?? new Float64Array(nF)));
    }
    const U: Float64Array[] = Array.from({ length: n }, () => new Float64Array(k));
    return { U, S, Vt };
  }

  private _gramSchmidt(vecs: Float64Array[]): Float64Array[] {
    const result: Float64Array[] = [];
    for (const v of vecs) {
      let u = new Float64Array(v);
      for (const e of result) {
        let dot = 0, norm2 = 0;
        for (let f = 0; f < u.length; f++) { dot += (u[f] ?? 0) * (e[f] ?? 0); norm2 += (e[f] ?? 0) ** 2; }
        const c = norm2 > 1e-10 ? dot / norm2 : 0;
        u = new Float64Array(u.map((val, f) => val - c * (e[f] ?? 0)));
      }
      const norm = Math.sqrt(u.reduce((s, val) => s + val * val, 0));
      if (norm > 1e-10) result.push(new Float64Array(u.map((val) => val / norm)));
    }
    return result;
  }

  private _seed = 42;
  private _randn(): number {
    this._seed = (this._seed * 1664525 + 1013904223) & 0xffffffff;
    const u1 = (this._seed >>> 0) / 0xffffffff;
    this._seed = (this._seed * 1664525 + 1013904223) & 0xffffffff;
    const u2 = (this._seed >>> 0) / 0xffffffff;
    return Math.sqrt(-2 * Math.log(Math.max(u1, 1e-10))) * Math.cos(2 * Math.PI * u2);
  }
}

export class IncrementalSVD {
  private components_: Float64Array[] = [];
  private singularValues_: Float64Array = new Float64Array(0);

  constructor(private readonly nComponents = 2) {}

  partialFit(X: Float64Array[]): this {
    // Simplified incremental update
    const nF = X[0]?.length ?? 1;
    if (this.components_.length === 0) {
      this.components_ = Array.from({ length: this.nComponents }, () => new Float64Array(nF));
      this.singularValues_ = new Float64Array(this.nComponents);
    }
    for (const x of X) {
      for (let k = 0; k < this.nComponents; k++) {
        const comp = this.components_[k]!;
        let dot = 0;
        for (let f = 0; f < nF; f++) dot += (x[f] ?? 0) * (comp[f] ?? 0);
        const lr = 0.01;
        for (let f = 0; f < nF; f++) comp[f] = (comp[f] ?? 0) + lr * dot * (x[f] ?? 0);
        const norm = Math.sqrt(comp.reduce((s, v) => s + v * v, 0));
        if (norm > 1e-10) for (let f = 0; f < nF; f++) comp[f] = (comp[f] ?? 0) / norm;
        this.singularValues_[k] = (this.singularValues_[k] ?? 0) * 0.99 + Math.abs(dot) * 0.01;
      }
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map((x) => new Float64Array(this.components_.map((comp) => {
      let dot = 0;
      for (let f = 0; f < x.length; f++) dot += (x[f] ?? 0) * (comp[f] ?? 0);
      return dot;
    })));
  }
}

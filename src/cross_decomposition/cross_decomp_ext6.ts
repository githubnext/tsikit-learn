/**
 * PLS2 extension and Canonical Correlation Analysis.
 */

export class PLS2Ext {
  private xWeights_!: Float64Array[];
  private yWeights_!: Float64Array[];
  private xLoadings_!: Float64Array[];
  private yLoadings_!: Float64Array[];
  private xScores_!: Float64Array[];
  private yScores_!: Float64Array[];
  private fitted_ = false;

  constructor(private nComponents = 2, private maxIter = 500, private tol = 1e-6) {}

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const n = X.length;
    const pX = X[0]?.length ?? 0;
    const pY = Y[0]?.length ?? 0;
    this.xWeights_ = [];
    this.yWeights_ = [];
    this.xLoadings_ = [];
    this.yLoadings_ = [];
    this.xScores_ = [];
    this.yScores_ = [];

    let Xr = X.map(row => new Float64Array(row));
    let Yr = Y.map(row => new Float64Array(row));

    for (let comp = 0; comp < this.nComponents; comp++) {
      let u = new Float64Array(n).map(() => Math.random());
      let normU = Math.sqrt(u.reduce((s, v) => s + v * v, 0));
      u = new Float64Array(u.map(v => v / (normU + 1e-10)));

      let w = new Float64Array(pX), c = new Float64Array(pY);
      let t = new Float64Array(n);

      for (let iter = 0; iter < this.maxIter; iter++) {
        const oldU = new Float64Array(u);
        // w = X^T u / ||X^T u||
        w = new Float64Array(pX).map((_, j) => Xr.reduce((s, row, i) => s + (row[j] ?? 0) * (u[i] ?? 0), 0));
        const wNorm = Math.sqrt(w.reduce((s, v) => s + v * v, 0));
        w = new Float64Array(w.map(v => v / (wNorm + 1e-10)));
        // t = X w / ||X w||
        t = new Float64Array(n).map((_, i) => Xr[i]!.reduce((s, v, j) => s + v * (w[j] ?? 0), 0));
        const tNorm = Math.sqrt(t.reduce((s, v) => s + v * v, 0));
        t = new Float64Array(t.map(v => v / (tNorm + 1e-10)));
        // c = Y^T t / ||Y^T t||
        c = new Float64Array(pY).map((_, k) => Yr.reduce((s, row, i) => s + (row[k] ?? 0) * (t[i] ?? 0), 0));
        const cNorm = Math.sqrt(c.reduce((s, v) => s + v * v, 0));
        c = new Float64Array(c.map(v => v / (cNorm + 1e-10)));
        // u = Y c / ||Y c||
        u = new Float64Array(n).map((_, i) => Yr[i]!.reduce((s, v, k) => s + v * (c[k] ?? 0), 0));
        const uNorm = Math.sqrt(u.reduce((s, v) => s + v * v, 0));
        u = new Float64Array(u.map(v => v / (uNorm + 1e-10)));
        const delta = Math.sqrt(u.reduce((s, v, i) => s + (v - (oldU[i] ?? 0)) ** 2, 0));
        if (delta < this.tol) break;
      }

      // Loadings
      const p = new Float64Array(pX).map((_, j) =>
        Xr.reduce((s, row, i) => s + (row[j] ?? 0) * (t[i] ?? 0), 0)
        / (t.reduce((s, v) => s + v * v, 0) + 1e-10)
      );
      const q = new Float64Array(pY).map((_, k) =>
        Yr.reduce((s, row, i) => s + (row[k] ?? 0) * (t[i] ?? 0), 0)
        / (t.reduce((s, v) => s + v * v, 0) + 1e-10)
      );

      this.xWeights_.push(w);
      this.yWeights_.push(c);
      this.xLoadings_.push(p);
      this.yLoadings_.push(q);
      this.xScores_.push(t);
      this.yScores_.push(u);

      // Deflate
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < pX; j++) Xr[i]![j] = (Xr[i]![j] ?? 0) - (t[i] ?? 0) * (p[j] ?? 0);
        for (let k = 0; k < pY; k++) Yr[i]![k] = (Yr[i]![k] ?? 0) - (t[i] ?? 0) * (q[k] ?? 0);
      }
    }
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return X.map(row => new Float64Array(this.nComponents).map((_, c) =>
      (this.xWeights_[c] ?? new Float64Array()).reduce((s, w, j) => s + w * (row[j] ?? 0), 0)
    ));
  }

  get xWeights(): Float64Array[] { return this.xWeights_; }
  get yWeights(): Float64Array[] { return this.yWeights_; }
}

export function canonicalCorrelationAnalysis(X: Float64Array[], Y: Float64Array[], nComponents = 2): {
  xProjections: Float64Array[];
  yProjections: Float64Array[];
  correlations: Float64Array;
} {
  const pls = new PLS2Ext(nComponents);
  pls.fit(X, Y);
  const xProjections = pls.transform(X);
  const yProjections = X.map((_, i) => new Float64Array(nComponents).map((_, c) =>
    (pls.yWeights[c] ?? new Float64Array()).reduce((s, w, k) => s + w * (Y[i]![k] ?? 0), 0)
  ));
  const correlations = new Float64Array(nComponents).map((_, c) => {
    const xs = xProjections.map(p => p[c] ?? 0);
    const ys = yProjections.map(p => p[c] ?? 0);
    const meanX = xs.reduce((s, v) => s + v, 0) / xs.length;
    const meanY = ys.reduce((s, v) => s + v, 0) / ys.length;
    const cov = xs.reduce((s, v, i) => s + (v - meanX) * ((ys[i] ?? 0) - meanY), 0) / xs.length;
    const stdX = Math.sqrt(xs.reduce((s, v) => s + (v - meanX) ** 2, 0) / xs.length);
    const stdY = Math.sqrt(ys.reduce((s, v) => s + (v - meanY) ** 2, 0) / ys.length);
    return stdX > 0 && stdY > 0 ? cov / (stdX * stdY) : 0;
  });
  return { xProjections, yProjections, correlations };
}

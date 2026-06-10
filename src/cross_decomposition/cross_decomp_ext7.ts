/**
 * Cross decomposition utilities and OPLS (Orthogonal Projections to Latent Structures).
 */

export function plsRegressionScore(
  yTrue: Float64Array,
  yPred: Float64Array
): { r2: number; rmse: number; q2: number } {
  const n = yTrue.length;
  const meanY = yTrue.reduce((s, v) => s + v, 0) / n;
  const ssTot = yTrue.reduce((s, v) => s + (v - meanY) ** 2, 0);
  const ssRes = yTrue.reduce((s, v, i) => s + (v - (yPred[i] ?? 0)) ** 2, 0);
  const r2 = 1 - ssRes / (ssTot + 1e-10);
  const rmse = Math.sqrt(ssRes / n);
  const press = ssRes; // Simplified PRESS = SSRes for single split
  const q2 = 1 - press / (ssTot + 1e-10);
  return { r2, rmse, q2 };
}

export class OPLS {
  private orthWeights_!: Float64Array[];
  private orthLoadings_!: Float64Array[];
  private predictiveWeight_!: Float64Array;
  private predictiveLoading_!: Float64Array;
  private fitted_ = false;

  constructor(private nOrthogonal = 1) {}

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length, p = X[0]?.length ?? 0;
    this.orthWeights_ = [];
    this.orthLoadings_ = [];

    let Xr = X.map(row => new Float64Array(row));
    const yNorm = Math.sqrt(y.reduce((s, v) => s + v * v, 0));
    const yUnit = new Float64Array(y.map(v => v / (yNorm + 1e-10)));

    // Predictive component
    const tw = new Float64Array(p).map((_, j) => Xr.reduce((s, row, i) => s + (row[j] ?? 0) * (yUnit[i] ?? 0), 0));
    const twNorm = Math.sqrt(tw.reduce((s, v) => s + v * v, 0));
    this.predictiveWeight_ = new Float64Array(tw.map(v => v / (twNorm + 1e-10)));

    for (let orth = 0; orth < this.nOrthogonal; orth++) {
      // Project X on predictive weight
      const tp = new Float64Array(n).map((_, i) =>
        Xr[i]!.reduce((s, v, j) => s + v * (this.predictiveWeight_[j] ?? 0), 0)
      );
      const tpNorm = Math.sqrt(tp.reduce((s, v) => s + v * v, 0));
      const tpUnit = new Float64Array(tp.map(v => v / (tpNorm + 1e-10)));

      // Residual in X space
      const pp = new Float64Array(p).map((_, j) =>
        Xr.reduce((s, row, i) => s + (row[j] ?? 0) * (tpUnit[i] ?? 0), 0) / (n - 1)
      );
      // Orthogonal component: residual of Xr after removing tp
      const Xres = Xr.map((row, i) =>
        new Float64Array(p).map((_, j) => (row[j] ?? 0) - (tpUnit[i] ?? 0) * (pp[j] ?? 0))
      );
      // Orthogonal weight from correlation with y
      const wo = new Float64Array(p).map((_, j) => Xres.reduce((s, row, i) => s + (row[j] ?? 0) * (yUnit[i] ?? 0), 0));
      const woNorm = Math.sqrt(wo.reduce((s, v) => s + v * v, 0));
      const woUnit = new Float64Array(wo.map(v => v / (woNorm + 1e-10)));
      const to = new Float64Array(n).map((_, i) => Xr[i]!.reduce((s, v, j) => s + v * (woUnit[j] ?? 0), 0));
      const toNorm = Math.sqrt(to.reduce((s, v) => s + v * v, 0));
      const toUnit = new Float64Array(to.map(v => v / (toNorm + 1e-10)));
      const lo = new Float64Array(p).map((_, j) =>
        Xr.reduce((s, row, i) => s + (row[j] ?? 0) * (toUnit[i] ?? 0), 0) / (n - 1)
      );
      this.orthWeights_.push(woUnit);
      this.orthLoadings_.push(lo);
      // Deflate X
      Xr = Xr.map((row, i) => new Float64Array(p).map((_, j) => (row[j] ?? 0) - (toUnit[i] ?? 0) * (lo[j] ?? 0)));
    }
    // Final predictive loading
    const tFinal = new Float64Array(n).map((_, i) =>
      Xr[i]!.reduce((s, v, j) => s + v * (this.predictiveWeight_[j] ?? 0), 0)
    );
    const tFinalNorm = Math.sqrt(tFinal.reduce((s, v) => s + v * v, 0));
    const tFinalUnit = new Float64Array(tFinal.map(v => v / (tFinalNorm + 1e-10)));
    this.predictiveLoading_ = new Float64Array(p).map((_, j) =>
      Xr.reduce((s, row, i) => s + (row[j] ?? 0) * (tFinalUnit[i] ?? 0), 0) / (n - 1)
    );
    this.fitted_ = true;
    return this;
  }

  filterOrthogonal(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    let Xf = X.map(row => new Float64Array(row));
    for (let orth = 0; orth < this.nOrthogonal; orth++) {
      const wo = this.orthWeights_[orth]!;
      const lo = this.orthLoadings_[orth]!;
      const to = Xf.map(row => row.reduce((s, v, j) => s + v * (wo[j] ?? 0), 0));
      const toNorm = Math.sqrt(to.reduce((s, v) => s + v * v, 0));
      const toUnit = new Float64Array(to.map(v => v / (toNorm + 1e-10)));
      Xf = Xf.map((row, i) => new Float64Array(row.map((v, j) => v - (toUnit[i] ?? 0) * (lo[j] ?? 0))));
    }
    return Xf;
  }
}

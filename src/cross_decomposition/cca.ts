/**
 * Canonical Correlation Analysis (CCA).
 * Mirrors sklearn.cross_decomposition.CCA.
 */

import { NotFittedError } from "../exceptions.js";

function colMeans(X: Float64Array[]): Float64Array {
  const p = (X[0] ?? new Float64Array(0)).length;
  const m = new Float64Array(p);
  for (const xi of X) {
    for (let j = 0; j < p; j++) m[j] = (m[j] ?? 0) + (xi[j] ?? 0);
  }
  for (let j = 0; j < p; j++) m[j] = (m[j] ?? 0) / X.length;
  return m;
}

function centerMatrix(X: Float64Array[], means: Float64Array): Float64Array[] {
  return X.map((xi) => new Float64Array(xi.map((v, j) => v - (means[j] ?? 0))));
}

/** X^T Y (p x q matrix). */
function crossProd(X: Float64Array[], Y: Float64Array[]): Float64Array[] {
  const p = (X[0] ?? new Float64Array(0)).length;
  const q = (Y[0] ?? new Float64Array(0)).length;
  const C = Array.from({ length: p }, () => new Float64Array(q));
  for (let i = 0; i < X.length; i++) {
    const xi = X[i] ?? new Float64Array(p);
    const yi = Y[i] ?? new Float64Array(q);
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < q; k++) {
        C[j]![k] = (C[j]![k] ?? 0) + (xi[j] ?? 0) * (yi[k] ?? 0);
      }
    }
  }
  return C;
}

/** Gram-Schmidt power iteration to find leading singular vectors. */
function powerSVD(
  M: Float64Array[],
  nComponents: number,
  maxIter = 200,
): { U: Float64Array[]; S: Float64Array; Vt: Float64Array[] } {
  const m = M.length;
  const n = (M[0] ?? new Float64Array(0)).length;
  const U: Float64Array[] = [];
  const S: number[] = [];
  const Vt: Float64Array[] = [];

  let Mdefl = M.map((row) => new Float64Array(row));

  for (let c = 0; c < nComponents; c++) {
    let u = new Float64Array(m);
    u[c % m] = 1;

    for (let iter = 0; iter < maxIter; iter++) {
      // v = M^T u
      const v = new Float64Array(n);
      for (let i = 0; i < m; i++) {
        const row = Mdefl[i] ?? new Float64Array(n);
        for (let j = 0; j < n; j++) v[j] = (v[j] ?? 0) + (u[i] ?? 0) * (row[j] ?? 0);
      }
      // normalize v
      let vnorm = 0;
      for (let j = 0; j < n; j++) vnorm += (v[j] ?? 0) ** 2;
      vnorm = Math.sqrt(vnorm);
      if (vnorm < 1e-10) break;
      for (let j = 0; j < n; j++) v[j] = (v[j] ?? 0) / vnorm;
      // u = M v
      const uNew = new Float64Array(m);
      for (let i = 0; i < m; i++) {
        const row = Mdefl[i] ?? new Float64Array(n);
        for (let j = 0; j < n; j++) uNew[i] = (uNew[i] ?? 0) + (row[j] ?? 0) * (v[j] ?? 0);
      }
      let unorm = 0;
      for (let i = 0; i < m; i++) unorm += (uNew[i] ?? 0) ** 2;
      unorm = Math.sqrt(unorm);
      if (unorm < 1e-10) break;
      const sigma = unorm;
      for (let i = 0; i < m; i++) uNew[i] = (uNew[i] ?? 0) / unorm;
      const diff = Math.sqrt(Array.from({ length: m }, (_, i) => ((uNew[i] ?? 0) - (u[i] ?? 0)) ** 2).reduce((a, b) => a + b, 0));
      u = uNew;
      if (diff < 1e-8) { S.push(sigma); break; }
      if (iter === maxIter - 1) S.push(sigma);
    }

    // Deflate
    const sigma = S[c] ?? 0;
    const v = new Float64Array(n);
    for (let i = 0; i < m; i++) {
      const row = Mdefl[i] ?? new Float64Array(n);
      for (let j = 0; j < n; j++) v[j] = (v[j] ?? 0) + (u[i] ?? 0) * (row[j] ?? 0);
    }
    let vnorm = 0;
    for (let j = 0; j < n; j++) vnorm += (v[j] ?? 0) ** 2;
    vnorm = Math.sqrt(vnorm);
    if (vnorm > 1e-10) for (let j = 0; j < n; j++) v[j] = (v[j] ?? 0) / vnorm;

    U.push(u);
    Vt.push(v);
    Mdefl = Mdefl.map((row, i) => {
      const newRow = new Float64Array(row);
      for (let j = 0; j < n; j++) {
        newRow[j] = (newRow[j] ?? 0) - sigma * (u[i] ?? 0) * (v[j] ?? 0);
      }
      return newRow;
    });
  }

  return { U, S: new Float64Array(S), Vt };
}

/**
 * Canonical Correlation Analysis.
 * Mirrors sklearn.cross_decomposition.CCA.
 */
export class CCA {
  nComponents: number;
  maxIter: number;
  tol: number;
  scale: boolean;

  xWeights_: Float64Array[] | null = null;
  yWeights_: Float64Array[] | null = null;
  xLoadings_: Float64Array[] | null = null;
  yLoadings_: Float64Array[] | null = null;
  xMean_: Float64Array | null = null;
  yMean_: Float64Array | null = null;
  xStd_: Float64Array | null = null;
  yStd_: Float64Array | null = null;

  constructor(
    options: {
      nComponents?: number;
      maxIter?: number;
      tol?: number;
      scale?: boolean;
    } = {},
  ) {
    this.nComponents = options.nComponents ?? 2;
    this.maxIter = options.maxIter ?? 500;
    this.tol = options.tol ?? 1e-6;
    this.scale = options.scale ?? true;
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const q = (Y[0] ?? new Float64Array(0)).length;

    this.xMean_ = colMeans(X);
    this.yMean_ = colMeans(Y);

    let Xc = centerMatrix(X, this.xMean_);
    let Yc = centerMatrix(Y, this.yMean_);

    // Compute std for scaling
    if (this.scale) {
      const xStd = new Float64Array(p);
      const yStd = new Float64Array(q);
      for (const xi of Xc) for (let j = 0; j < p; j++) xStd[j] = (xStd[j] ?? 0) + (xi[j] ?? 0) ** 2;
      for (const yi of Yc) for (let j = 0; j < q; j++) yStd[j] = (yStd[j] ?? 0) + (yi[j] ?? 0) ** 2;
      for (let j = 0; j < p; j++) xStd[j] = Math.sqrt((xStd[j] ?? 0) / n);
      for (let j = 0; j < q; j++) yStd[j] = Math.sqrt((yStd[j] ?? 0) / n);
      this.xStd_ = xStd;
      this.yStd_ = yStd;
      Xc = Xc.map((xi) => new Float64Array(xi.map((v, j) => v / Math.max(xStd[j] ?? 1, 1e-10))));
      Yc = Yc.map((yi) => new Float64Array(yi.map((v, j) => v / Math.max(yStd[j] ?? 1, 1e-10))));
    }

    // CCA via SVD of X^T Y
    const Cxy = crossProd(Xc, Yc);
    const k = Math.min(this.nComponents, p, q);
    const { U, Vt } = powerSVD(Cxy, k, this.maxIter);

    this.xWeights_ = U;
    this.yWeights_ = Vt;

    // Compute loadings
    this.xLoadings_ = Array.from({ length: k }, (_, c) => {
      const w = U[c] ?? new Float64Array(p);
      const t = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < p; j++) t[i] = (t[i] ?? 0) + ((Xc[i] ?? new Float64Array(p))[j] ?? 0) * (w[j] ?? 0);
      }
      const load = new Float64Array(p);
      for (let j = 0; j < p; j++) {
        let cov = 0;
        for (let i = 0; i < n; i++) cov += ((Xc[i] ?? new Float64Array(p))[j] ?? 0) * (t[i] ?? 0);
        let tNorm = 0;
        for (let i = 0; i < n; i++) tNorm += (t[i] ?? 0) ** 2;
        load[j] = tNorm > 0 ? cov / tNorm : 0;
      }
      return load;
    });

    this.yLoadings_ = Array.from({ length: k }, (_, c) => {
      const w = Vt[c] ?? new Float64Array(q);
      const u = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < q; j++) u[i] = (u[i] ?? 0) + ((Yc[i] ?? new Float64Array(q))[j] ?? 0) * (w[j] ?? 0);
      }
      const load = new Float64Array(q);
      for (let j = 0; j < q; j++) {
        let cov = 0;
        for (let i = 0; i < n; i++) cov += ((Yc[i] ?? new Float64Array(q))[j] ?? 0) * (u[i] ?? 0);
        let uNorm = 0;
        for (let i = 0; i < n; i++) uNorm += (u[i] ?? 0) ** 2;
        load[j] = uNorm > 0 ? cov / uNorm : 0;
      }
      return load;
    });

    return this;
  }

  transform(X: Float64Array[], Y?: Float64Array[]): [Float64Array[], Float64Array[] | null] {
    if (this.xWeights_ === null || this.xMean_ === null) throw new NotFittedError("CCA");
    const xMean = this.xMean_;
    const xStd = this.xStd_;
    const k = this.nComponents;

    let Xc = X.map((xi) => new Float64Array(xi.map((v, j) => v - (xMean[j] ?? 0))));
    if (xStd) Xc = Xc.map((xi) => new Float64Array(xi.map((v, j) => v / Math.max(xStd[j] ?? 1, 1e-10))));

    const xScores = X.map((_, i) => {
      const scores = new Float64Array(k);
      for (let c = 0; c < k; c++) {
        const w = this.xWeights_![c] ?? new Float64Array(0);
        for (let j = 0; j < w.length; j++) scores[c] = (scores[c] ?? 0) + ((Xc[i] ?? new Float64Array(0))[j] ?? 0) * (w[j] ?? 0);
      }
      return scores;
    });

    if (Y === undefined) return [xScores, null];

    const yMean = this.yMean_!;
    const yStd = this.yStd_;
    let Yc = Y.map((yi) => new Float64Array(yi.map((v, j) => v - (yMean[j] ?? 0))));
    if (yStd) Yc = Yc.map((yi) => new Float64Array(yi.map((v, j) => v / Math.max(yStd[j] ?? 1, 1e-10))));

    const yScores = Y.map((_, i) => {
      const scores = new Float64Array(k);
      for (let c = 0; c < k; c++) {
        const w = this.yWeights_![c] ?? new Float64Array(0);
        for (let j = 0; j < w.length; j++) scores[c] = (scores[c] ?? 0) + ((Yc[i] ?? new Float64Array(0))[j] ?? 0) * (w[j] ?? 0);
      }
      return scores;
    });

    return [xScores, yScores];
  }

  fitTransform(X: Float64Array[], Y: Float64Array[]): [Float64Array[], Float64Array[]] {
    this.fit(X, Y);
    const [xS, yS] = this.transform(X, Y);
    return [xS, yS!];
  }
}

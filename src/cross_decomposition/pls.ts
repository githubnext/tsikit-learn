/**
 * Cross decomposition: PLSRegression, PLSSVD, PLSCanonical, CCA.
 * Mirrors sklearn.cross_decomposition.
 */

import { NotFittedError } from "../exceptions.js";

/** Compute column means. */
function colMeans(X: Float64Array[]): Float64Array {
  const p = (X[0] ?? new Float64Array(0)).length;
  const m = new Float64Array(p);
  for (const xi of X)
    for (let j = 0; j < p; j++) m[j] = (m[j] ?? 0) + (xi[j] ?? 0);
  for (let j = 0; j < p; j++) m[j] = (m[j] ?? 0) / X.length;
  return m;
}

/** Center X by subtracting column means. */
function center(X: Float64Array[], means: Float64Array): Float64Array[] {
  const p = means.length;
  return X.map((xi) => {
    const out = new Float64Array(p);
    for (let j = 0; j < p; j++) out[j] = (xi[j] ?? 0) - (means[j] ?? 0);
    return out;
  });
}

/** Compute X^T Y (p x q). */
function Xtranspose_Y(X: Float64Array[], Y: Float64Array[]): Float64Array[] {
  const p = (X[0] ?? new Float64Array(0)).length;
  const q = (Y[0] ?? new Float64Array(0)).length;
  const n = X.length;
  const out = Array.from({ length: p }, () => new Float64Array(q));
  for (let i = 0; i < n; i++) {
    const xi = X[i] ?? new Float64Array(p);
    const yi = Y[i] ?? new Float64Array(q);
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < q; k++) {
        out[j]![k] = (out[j]![k] ?? 0) + (xi[j] ?? 0) * (yi[k] ?? 0);
      }
    }
  }
  return out;
}

/** Compute matrix-vector product. */
function matVec(M: Float64Array[], v: Float64Array): Float64Array {
  const out = new Float64Array(M.length);
  for (let i = 0; i < M.length; i++) {
    const row = M[i] ?? new Float64Array(0);
    for (let j = 0; j < v.length; j++)
      out[i] = (out[i] ?? 0) + (row[j] ?? 0) * (v[j] ?? 0);
  }
  return out;
}

/** L2 norm of a vector. */
function norm(v: Float64Array): number {
  let s = 0;
  for (let j = 0; j < v.length; j++) s += (v[j] ?? 0) ** 2;
  return Math.sqrt(s);
}

/** Normalize a vector in-place. */
function normalize(v: Float64Array): void {
  const n = norm(v);
  if (n > 1e-15) for (let j = 0; j < v.length; j++) v[j] = (v[j] ?? 0) / n;
}

/** Dot product. */
function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let j = 0; j < a.length; j++) s += (a[j] ?? 0) * (b[j] ?? 0);
  return s;
}

/** NIPALS: find first left/right singular vectors of M via power iteration. */
function nipals(
  XtY: Float64Array[],
  tol = 1e-10,
  maxIter = 500,
): { u: Float64Array; v: Float64Array } {
  const p = XtY.length;
  const q = (XtY[0] ?? new Float64Array(0)).length;
  let v = new Float64Array(q);
  v[0] = 1;
  let u = new Float64Array(p);
  for (let iter = 0; iter < maxIter; iter++) {
    // u = XtY v / ||XtY v||
    const uNew = matVec(XtY, v);
    normalize(uNew);
    // v = XtY^T u / ||XtY^T u||
    const vNew = new Float64Array(q);
    for (let k = 0; k < q; k++) {
      for (let j = 0; j < p; j++) {
        vNew[k] = (vNew[k] ?? 0) + (XtY[j]![k] ?? 0) * (uNew[j] ?? 0);
      }
    }
    normalize(vNew);
    const diff =
      norm(
        Float64Array.from(
          { length: p },
          (_, i) => (uNew[i] ?? 0) - (u[i] ?? 0),
        ),
      ) +
      norm(
        Float64Array.from(
          { length: q },
          (_, i) => (vNew[i] ?? 0) - (v[i] ?? 0),
        ),
      );
    u = uNew as Float64Array<ArrayBuffer>;
    v = vNew;
    if (diff < tol) break;
  }
  return { u, v };
}

/**
 * PLS regression via NIPALS algorithm.
 * Mirrors sklearn.cross_decomposition.PLSRegression.
 */
export class PLSRegression {
  nComponents: number;
  maxIter: number;
  tol: number;
  scale: boolean;

  xWeights_: Float64Array[] | null = null;
  yWeights_: Float64Array[] | null = null;
  xLoadings_: Float64Array[] | null = null;
  yLoadings_: Float64Array[] | null = null;
  xScores_: Float64Array[] | null = null;
  yScores_: Float64Array[] | null = null;
  coef_: Float64Array[] | null = null;

  xMean_: Float64Array | null = null;
  yMean_: Float64Array | null = null;

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
    const k = Math.min(this.nComponents, p, n);

    this.xMean_ = colMeans(X);
    this.yMean_ = colMeans(Y);
    let Xc = center(X, this.xMean_);
    let Yc = center(Y, this.yMean_);

    this.xWeights_ = [];
    this.yWeights_ = [];
    this.xLoadings_ = [];
    this.yLoadings_ = [];
    this.xScores_ = Array.from({ length: n }, () => new Float64Array(k));
    this.yScores_ = Array.from({ length: n }, () => new Float64Array(k));

    for (let comp = 0; comp < k; comp++) {
      const XtY = Xtranspose_Y(Xc, Yc);
      const { u, v } = nipals(XtY, this.tol, this.maxIter);

      // Scores: t = Xc u, s = Yc v
      const t = new Float64Array(n);
      const s = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const xi = Xc[i] ?? new Float64Array(p);
        const yi = Yc[i] ?? new Float64Array(q);
        t[i] = dot(xi, u);
        s[i] = dot(yi, v);
      }

      // Normalize t in-place for use in loadings and deflation
      const tNorm = norm(t);
      if (tNorm > 1e-15) for (let i = 0; i < n; i++) t[i] = (t[i] ?? 0) / tNorm;

      // X loadings: p_h = Xc^T t_normalized
      const px = new Float64Array(p);
      for (let i = 0; i < n; i++) {
        const xi = Xc[i] ?? new Float64Array(p);
        for (let j = 0; j < p; j++)
          px[j] = (px[j] ?? 0) + (xi[j] ?? 0) * (t[i] ?? 0);
      }

      // Y loadings: q_h = Yc^T t_normalized (inner relation with x scores)
      const qy = new Float64Array(q);
      for (let i = 0; i < n; i++) {
        const yi = Yc[i] ?? new Float64Array(q);
        for (let j = 0; j < q; j++) {
          qy[j] = (qy[j] ?? 0) + (yi[j] ?? 0) * (t[i] ?? 0);
        }
      }
      if (sNorm2 > 1e-15)
        for (let j = 0; j < q; j++) qy[j] = (qy[j] ?? 0) / sNorm2;

      this.xWeights_[comp] = u;
      this.yWeights_[comp] = v;
      this.xLoadings_[comp] = px;
      this.yLoadings_[comp] = qy;
      for (let i = 0; i < n; i++) {
        this.xScores_![i]![comp] = t[i] ?? 0;
        this.yScores_![i]![comp] = s[i] ?? 0;
      }

      // Deflate using t_normalized (consistent with how px and qy were computed)
      Xc = Xc.map((xi, i) => {
        const out = new Float64Array(p);
        for (let j = 0; j < p; j++)
          out[j] = (xi[j] ?? 0) - (tFull[i] ?? 0) * (px[j] ?? 0);
        return out;
      });
      Yc = Yc.map((yi, i) => {
        const out = new Float64Array(q);
        for (let j = 0; j < q; j++)
          out[j] = (yi[j] ?? 0) - (tFull[i] ?? 0) * (qy[j] ?? 0);
        return out;
      });
    }

    // Compute regression coefficients: coef_ = W (P^T W)^{-1} Q^T
    // Simplified: use pseudo-inverse via stored weights and loadings
    this._computeCoef(p, q, k);
    return this;
  }

  private _computeCoef(p: number, q: number, k: number): void {
    // coef_ = xWeights_ @ inv(xLoadings_^T @ xWeights_) @ yLoadings_^T
    // For simplicity, use a direct approach: coef = W (P^T W)^-1 Q^T
    const W = this.xWeights_!;
    const P = this.xLoadings_!;
    const Q = this.yLoadings_!;

    // PtW = P^T W (k x k)
    const PtW = Array.from({ length: k }, () => new Float64Array(k));
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        PtW[i]![j] = dot(
          P[i] ?? new Float64Array(0),
          W[j] ?? new Float64Array(0),
        );
      }
    }

    // Invert PtW (simple LU for small k)
    const inv = this._invertSmall(PtW, k);

    // coef_ (p x q) = W @ inv @ Q^T
    this.coef_ = Array.from({ length: p }, () => new Float64Array(q));
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < q; j++) {
        let s = 0;
        for (let a = 0; a < k; a++) {
          let s2 = 0;
          for (let b = 0; b < k; b++) {
            s2 += (inv[a]![b] ?? 0) * (Q[b]![j] ?? 0);
          }
          s += (W[a]![i] ?? 0) * s2;
        }
        this.coef_![i]![j] = s;
      }
    }
  }

  private _invertSmall(M: Float64Array[], k: number): Float64Array[] {
    // Augmented matrix [M | I]
    const aug = Array.from({ length: k }, (_, i) => {
      const row = new Float64Array(2 * k);
      for (let j = 0; j < k; j++) row[j] = M[i]![j] ?? 0;
      row[k + i] = 1;
      return row;
    });
    for (let col = 0; col < k; col++) {
      // Find pivot
      let maxRow = col;
      for (let row = col + 1; row < k; row++) {
        if (Math.abs(aug[row]![col] ?? 0) > Math.abs(aug[maxRow]![col] ?? 0))
          maxRow = row;
      }
      const tmpPls = aug[col]!;
      aug[col] = aug[maxRow]!;
      aug[maxRow] = tmpPls;
      const pivot = aug[col]![col] ?? 1e-12;
      if (Math.abs(pivot) < 1e-15) continue;
      for (let j = 0; j < 2 * k; j++)
        aug[col]![j] = (aug[col]![j] ?? 0) / pivot;
      for (let row = 0; row < k; row++) {
        if (row === col) continue;
        const factor = aug[row]![col] ?? 0;
        for (let j = 0; j < 2 * k; j++) {
          aug[row]![j] = (aug[row]![j] ?? 0) - factor * (aug[col]![j] ?? 0);
        }
      }
    }
    return aug.map((row) =>
      Float64Array.from({ length: k }, (_, j) => row[k + j] ?? 0),
    );
  }

  predict(X: Float64Array[]): Float64Array[] {
    if (this.coef_ === null || this.xMean_ === null || this.yMean_ === null) {
      throw new NotFittedError();
    }
    const p = this.xMean_.length;
    const q = this.yMean_.length;
    return X.map((xi) => {
      const xc = new Float64Array(p);
      for (let j = 0; j < p; j++) xc[j] = (xi[j] ?? 0) - (this.xMean_![j] ?? 0);
      const out = new Float64Array(q);
      for (let j = 0; j < q; j++) {
        let s = 0;
        for (let k = 0; k < p; k++)
          s += (xc[k] ?? 0) * (this.coef_![k]![j] ?? 0);
        out[j] = s + (this.yMean_![j] ?? 0);
      }
      return out;
    });
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.xWeights_ === null || this.xMean_ === null)
      throw new NotFittedError();
    const k = this.xWeights_.length;
    const p = this.xMean_.length;
    return X.map((xi) => {
      const xc = new Float64Array(p);
      for (let j = 0; j < p; j++) xc[j] = (xi[j] ?? 0) - (this.xMean_![j] ?? 0);
      const out = new Float64Array(k);
      for (let i = 0; i < k; i++) {
        out[i] = dot(xc, this.xWeights_![i] ?? new Float64Array(0));
      }
      return out;
    });
  }

  fitTransform(
    X: Float64Array[],
    Y: Float64Array[],
  ): [Float64Array[], Float64Array[]] {
    this.fit(X, Y);
    return [this.xScores_!, this.yScores_!];
  }
}

/**
 * Partial Least Squares SVD.
 * Mirrors sklearn.cross_decomposition.PLSSVD.
 */
export class PLSSVD {
  nComponents: number;

  xWeights_: Float64Array[] | null = null;
  yWeights_: Float64Array[] | null = null;
  xScores_: Float64Array[] | null = null;
  yScores_: Float64Array[] | null = null;
  xMean_: Float64Array | null = null;
  yMean_: Float64Array | null = null;

  constructor(options: { nComponents?: number } = {}) {
    this.nComponents = options.nComponents ?? 2;
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const q = (Y[0] ?? new Float64Array(0)).length;
    const k = Math.min(this.nComponents, p, q);

    this.xMean_ = colMeans(X);
    this.yMean_ = colMeans(Y);
    const Xc = center(X, this.xMean_);
    const Yc = center(Y, this.yMean_);

    this.xWeights_ = [];
    this.yWeights_ = [];
    this.xScores_ = Array.from({ length: n }, () => new Float64Array(k));
    this.yScores_ = Array.from({ length: n }, () => new Float64Array(k));

    const curXtY = Xtranspose_Y(Xc, Yc);
    for (let comp = 0; comp < k; comp++) {
      const { u, v } = nipals(curXtY);
      this.xWeights_[comp] = u;
      this.yWeights_[comp] = v;
      for (let i = 0; i < n; i++) {
        const xi = Xc[i] ?? new Float64Array(p);
        const yi = Yc[i] ?? new Float64Array(q);
        this.xScores_![i]![comp] = dot(xi, u);
        this.yScores_![i]![comp] = dot(yi, v);
      }
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.xWeights_ === null || this.xMean_ === null)
      throw new NotFittedError();
    const k = this.xWeights_.length;
    const p = this.xMean_.length;
    return X.map((xi) => {
      const xc = new Float64Array(p);
      for (let j = 0; j < p; j++) xc[j] = (xi[j] ?? 0) - (this.xMean_![j] ?? 0);
      const out = new Float64Array(k);
      for (let i = 0; i < k; i++)
        out[i] = dot(xc, this.xWeights_![i] ?? new Float64Array(0));
      return out;
    });
  }

  fitTransform(
    X: Float64Array[],
    Y: Float64Array[],
  ): [Float64Array[], Float64Array[]] {
    this.fit(X, Y);
    return [this.xScores_!, this.yScores_!];
  }
}

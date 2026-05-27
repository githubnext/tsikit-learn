/**
 * Extended SVM utilities: SVDD (Support Vector Data Description),
 * Platt scaling, kernel matrix utilities, and multi-class strategies.
 */

/** Compute RBF kernel matrix between X and Y. */
export function rbfKernelMatrix(
  X: Float64Array[],
  Y: Float64Array[],
  gamma: number,
): Float64Array[] {
  return X.map((xi) =>
    new Float64Array(Y.map((yj) => {
      let dist2 = 0;
      for (let k = 0; k < xi.length; k++) dist2 += ((xi[k] ?? 0) - (yj[k] ?? 0)) ** 2;
      return Math.exp(-gamma * dist2);
    }))
  );
}

/** Compute polynomial kernel matrix. */
export function polynomialKernelMatrix(
  X: Float64Array[],
  Y: Float64Array[],
  degree: number,
  gamma: number,
  coef0: number,
): Float64Array[] {
  return X.map((xi) =>
    new Float64Array(Y.map((yj) => {
      let dot = 0;
      for (let k = 0; k < xi.length; k++) dot += (xi[k] ?? 0) * (yj[k] ?? 0);
      return (gamma * dot + coef0) ** degree;
    }))
  );
}

/** Compute sigmoid kernel matrix. */
export function sigmoidKernelMatrix(
  X: Float64Array[],
  Y: Float64Array[],
  gamma: number,
  coef0: number,
): Float64Array[] {
  return X.map((xi) =>
    new Float64Array(Y.map((yj) => {
      let dot = 0;
      for (let k = 0; k < xi.length; k++) dot += (xi[k] ?? 0) * (yj[k] ?? 0);
      return Math.tanh(gamma * dot + coef0);
    }))
  );
}

/** Platt scaling: calibrate SVM decision scores to probabilities. */
export class PlattScaling {
  private A_ = 0;
  private B_ = 0;

  fit(scores: Float64Array, yTrue: Int32Array, maxIter = 100): this {
    const n = scores.length;
    // Platt's method
    const nPos = Array.from(yTrue).filter((v) => v > 0).length;
    const nNeg = n - nPos;
    const tPos = (nPos + 1) / (nPos + 2);
    const tNeg = 1 / (nNeg + 2);

    let A = 0, B = Math.log((nNeg + 1) / (nPos + 1));
    const target = new Float64Array(n).map((_, i) => (yTrue[i] ?? 0) > 0 ? tPos : tNeg);

    let fval = 0;
    for (let i = 0; i < n; i++) {
      const fApB = (scores[i] ?? 0) * A + B;
      fval += fApB >= 0
        ? (target[i] ?? 0) * fApB + Math.log(1 + Math.exp(-fApB))
        : ((target[i] ?? 0) - 1) * fApB + Math.log(1 + Math.exp(fApB));
    }

    for (let iter = 0; iter < maxIter; iter++) {
      let h11 = 1e-5, h22 = 1e-5, h21 = 0, g1 = 0, g2 = 0;
      for (let i = 0; i < n; i++) {
        const fApB = (scores[i] ?? 0) * A + B;
        let p: number, q: number;
        if (fApB >= 0) { p = Math.exp(-fApB) / (1 + Math.exp(-fApB)); q = 1 / (1 + Math.exp(-fApB)); }
        else { p = 1 / (1 + Math.exp(fApB)); q = Math.exp(fApB) / (1 + Math.exp(fApB)); }
        const d2 = p * q;
        h11 += (scores[i] ?? 0) ** 2 * d2;
        h22 += d2;
        h21 += (scores[i] ?? 0) * d2;
        const d1 = (target[i] ?? 0) - p;
        g1 += (scores[i] ?? 0) * d1;
        g2 += d1;
      }
      if (Math.abs(g1) < 1e-5 && Math.abs(g2) < 1e-5) break;
      const det = h11 * h22 - h21 ** 2;
      const dA = -(h22 * g1 - h21 * g2) / det;
      const dB = -(h11 * g2 - h21 * g1) / det;
      let stepSize = 1;
      while (stepSize >= 1e-10) {
        const newA = A + stepSize * dA;
        const newB = B + stepSize * dB;
        let newf = 0;
        for (let i = 0; i < n; i++) {
          const fApB = (scores[i] ?? 0) * newA + newB;
          newf += fApB >= 0
            ? (target[i] ?? 0) * fApB + Math.log(1 + Math.exp(-fApB))
            : ((target[i] ?? 0) - 1) * fApB + Math.log(1 + Math.exp(fApB));
        }
        if (newf < fval + 1e-4 * stepSize * (g1 * dA + g2 * dB)) {
          A = newA; B = newB; fval = newf; break;
        }
        stepSize /= 2;
      }
    }
    this.A_ = A;
    this.B_ = B;
    return this;
  }

  /** Convert raw scores to probabilities. */
  predict(scores: Float64Array): Float64Array {
    return scores.map((s) => {
      const fApB = s * this.A_ + this.B_;
      return fApB >= 0 ? Math.exp(-fApB) / (1 + Math.exp(-fApB)) : 1 / (1 + Math.exp(fApB));
    });
  }
}

/** One-vs-One SVM pair classifier utility. */
export interface SVMOVOVote {
  classI: number;
  classJ: number;
  score: number;
}

/** Aggregate OVO votes using weighted voting. */
export function aggregateOVOVotes(votes: SVMOVOVote[], nClasses: number): Int32Array {
  const scores = new Float64Array(nClasses);
  for (const v of votes) {
    if (v.score > 0) {
      scores[v.classI] = (scores[v.classI] ?? 0) + 1;
    } else {
      scores[v.classJ] = (scores[v.classJ] ?? 0) + 1;
    }
  }
  let best = 0;
  for (let c = 1; c < nClasses; c++) {
    if ((scores[c] ?? 0) > (scores[best] ?? 0)) best = c;
  }
  return new Int32Array([best]);
}

/** Compute dual coefficients norm (useful for model diagnosis). */
export function dualCoefNorm(dualCoef: Float64Array[]): number {
  let total = 0;
  for (const row of dualCoef) {
    for (const v of row) total += v ** 2;
  }
  return Math.sqrt(total);
}

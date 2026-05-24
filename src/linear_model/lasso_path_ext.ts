/**
 * Extended lasso path utilities.
 * Mirrors scikit-learn's linear_model.lasso_path with enet_path extensions.
 */

export interface LassoPathResult {
  alphas: Float64Array;
  coefs: Float64Array[];
  nIter: Int32Array;
}

/**
 * Compute Lasso path with coordinate descent.
 * Returns coefficients for a range of regularization parameters.
 */
export function lassoPathExt(
  X: Float64Array[],
  y: Float64Array,
  options: {
    eps?: number;
    nAlphas?: number;
    alphas?: Float64Array;
    maxIter?: number;
    tol?: number;
    l1Ratio?: number;
  } = {},
): LassoPathResult {
  const {
    eps = 1e-3,
    nAlphas = 100,
    maxIter = 1000,
    tol = 1e-4,
    l1Ratio = 1.0, // 1 = Lasso, 0 = Ridge
  } = options;

  const n = X.length;
  const p = X[0]?.length ?? 0;

  // Compute alpha_max
  const Xty = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += (X[i]?.[j] ?? 0) * (y[i] ?? 0);
    Xty[j] = s;
  }
  const alphaMax = Math.max(...Array.from(Xty).map(Math.abs)) / n;

  const alphas =
    options.alphas ??
    (() => {
      const result = new Float64Array(nAlphas);
      for (let k = 0; k < nAlphas; k++) {
        result[k] = alphaMax * Math.exp((-Math.log(1 / eps) * k) / (nAlphas - 1));
      }
      return result;
    })();

  const coefs: Float64Array[] = [];
  const nIter: number[] = [];
  let w = new Float64Array(p);

  for (const alpha of alphas) {
    const l1 = l1Ratio * alpha;
    const l2 = (1 - l1Ratio) * alpha;
    let iter = 0;
    for (; iter < maxIter; iter++) {
      let maxChange = 0;
      for (let j = 0; j < p; j++) {
        const wj = w[j] ?? 0;
        let rho = Xty[j] ?? 0;
        for (let k = 0; k < p; k++) {
          if (k !== j) {
            let xjxk = 0;
            for (let i = 0; i < n; i++) {
              xjxk += (X[i]?.[j] ?? 0) * (X[i]?.[k] ?? 0);
            }
            rho -= xjxk * (w[k] ?? 0) / n;
          }
        }
        // Column norm squared
        let norm2 = 0;
        for (let i = 0; i < n; i++) norm2 += (X[i]?.[j] ?? 0) ** 2;
        norm2 /= n;

        const denom = norm2 + l2;
        let newWj: number;
        if (l1 > 0) {
          // Soft threshold
          const thresh = l1 / denom;
          newWj = rho > thresh ? (rho - thresh) / denom
                : rho < -thresh ? (rho + thresh) / denom
                : 0;
        } else {
          newWj = denom > 0 ? rho / denom : 0;
        }
        const change = Math.abs(newWj - wj);
        if (change > maxChange) maxChange = change;
        w[j] = newWj;
      }
      if (maxChange < tol) break;
    }
    coefs.push(w.slice());
    nIter.push(iter);
    w = coefs[coefs.length - 1]!.slice();
  }

  return {
    alphas,
    coefs,
    nIter: Int32Array.from(nIter),
  };
}

/**
 * ElasticNet path (l1_ratio < 1).
 */
export function enetPathExt(
  X: Float64Array[],
  y: Float64Array,
  l1Ratio = 0.5,
  options: Parameters<typeof lassoPathExt>[2] = {},
): LassoPathResult {
  return lassoPathExt(X, y, { ...options, l1Ratio });
}

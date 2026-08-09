/**
 * Optimization utilities.
 * Mirrors sklearn.utils.optimize (line search, L-BFGS helpers).
 */

/** Armijo-Goldstein line search (backtracking). */
export function lineSearchArmijo(
  f: (x: Float64Array) => number,
  xk: Float64Array,
  pk: Float64Array,
  fk: number,
  alpha0 = 1.0,
  c1 = 1e-4,
  rho = 0.5,
  maxIter = 50,
): { alpha: number; fNew: number; nIter: number } {
  let alpha = alpha0;
  const n = xk.length;
  let dotGradPk = 0;
  for (let i = 0; i < n; i++) dotGradPk += (pk[i] ?? 0) * (pk[i] ?? 0); // assumes pk=-grad

  for (let iter = 0; iter < maxIter; iter++) {
    const xNew = new Float64Array(n);
    for (let i = 0; i < n; i++) xNew[i] = (xk[i] ?? 0) + alpha * (pk[i] ?? 0);
    const fNew = f(xNew);
    if (fNew <= fk - c1 * alpha * dotGradPk)
      return { alpha, fNew, nIter: iter + 1 };
    alpha *= rho;
  }
  return {
    alpha,
    fNew: f(
      (() => {
        const r = new Float64Array(n);
        for (let i = 0; i < n; i++) r[i] = (xk[i] ?? 0) + alpha * (pk[i] ?? 0);
        return r;
      })(),
    ),
    nIter: maxIter,
  };
}

/** Two-loop L-BFGS recursion to compute H * (-grad). Returns search direction. */
export function lbfgsTwoLoop(
  grad: Float64Array,
  sHistory: Float64Array[],
  yHistory: Float64Array[],
  rhoHistory: Float64Array,
): Float64Array {
  const n = grad.length;
  const m = sHistory.length;
  const q = new Float64Array(grad); // copy grad
  const alphas = new Float64Array(m);

  for (let i = m - 1; i >= 0; i--) {
    const si = sHistory[i]!;
    const yi = yHistory[i]!;
    const rho_i = rhoHistory[i] ?? 0;
    let syDot = 0;
    for (let j = 0; j < n; j++) syDot += (si[j] ?? 0) * (q[j] ?? 0);
    alphas[i] = rho_i * syDot;
    for (let j = 0; j < n; j++) q[j]! -= (alphas[i] ?? 0) * (yi[j] ?? 0);
  }

  // Scale by H0 = (s^T y) / (y^T y)
  let sTy = 0;
  let yTy = 0;
  if (m > 0) {
    const sLast = sHistory[m - 1]!;
    const yLast = yHistory[m - 1]!;
    for (let j = 0; j < n; j++) {
      sTy += (sLast[j] ?? 0) * (yLast[j] ?? 0);
      yTy += (yLast[j] ?? 0) * (yLast[j] ?? 0);
    }
  }
  const gamma = yTy > 0 ? sTy / yTy : 1.0;
  const r = new Float64Array(n);
  for (let j = 0; j < n; j++) r[j] = gamma * (q[j] ?? 0);

  for (let i = 0; i < m; i++) {
    const si = sHistory[i]!;
    const yi = yHistory[i]!;
    const rho_i = rhoHistory[i] ?? 0;
    let yDotR = 0;
    for (let j = 0; j < n; j++) yDotR += (yi[j] ?? 0) * (r[j] ?? 0);
    const beta = rho_i * yDotR;
    for (let j = 0; j < n; j++)
      r[j]! += (si[j] ?? 0) * ((alphas[i] ?? 0) - beta);
  }

  // Return -r (descent direction)
  const dir = new Float64Array(n);
  for (let j = 0; j < n; j++) dir[j] = -r[j]!;
  return dir;
}

export interface LBFGSResult {
  x: Float64Array;
  fVal: number;
  nIter: number;
  converged: boolean;
}

/**
 * L-BFGS-B minimizer for unconstrained smooth objectives.
 *
 * @param f  objective function; returns [value, gradient]
 * @param x0 starting point
 */
export function minimize(
  f: (x: Float64Array) => [number, Float64Array],
  x0: Float64Array,
  options: { tol?: number; maxIter?: number; m?: number } = {},
): LBFGSResult {
  const { tol = 1e-5, maxIter = 200, m = 10 } = options;
  const n = x0.length;
  let x = new Float64Array(x0);
  const sHistory: Float64Array[] = [];
  const yHistory: Float64Array[] = [];
  const rhoHistory = new Float64Array(m);

  let [fVal, grad] = f(x);
  let converged = false;

  for (let iter = 0; iter < maxIter; iter++) {
    // Convergence check
    let gNorm = 0;
    for (let j = 0; j < n; j++) gNorm = Math.max(gNorm, Math.abs(grad[j] ?? 0));
    if (gNorm < tol) {
      converged = true;
      break;
    }

    const pk =
      sHistory.length === 0
        ? (() => {
            const d = new Float64Array(n);
            for (let j = 0; j < n; j++) d[j] = -(grad[j] ?? 0);
            return d;
          })()
        : lbfgsTwoLoop(grad, sHistory, yHistory, rhoHistory);

    const { alpha } = lineSearchArmijo(
      (xx: Float64Array) => f(xx)[0],
      x,
      pk,
      fVal,
    );

    const xNew = new Float64Array(n);
    for (let j = 0; j < n; j++) xNew[j] = (x[j] ?? 0) + alpha * (pk[j] ?? 0);

    const [fNew, gradNew] = f(xNew);

    const sk = new Float64Array(n);
    const yk = new Float64Array(n);
    let sTy = 0;
    for (let j = 0; j < n; j++) {
      sk[j] = (xNew[j] ?? 0) - (x[j] ?? 0);
      yk[j] = (gradNew[j] ?? 0) - (grad[j] ?? 0);
      sTy += (sk[j] ?? 0) * (yk[j] ?? 0);
    }

    if (sTy > 0) {
      const idx = sHistory.length % m;
      if (sHistory.length < m) {
        sHistory.push(sk);
        yHistory.push(yk);
      } else {
        sHistory[idx] = sk;
        yHistory[idx] = yk;
      }
      rhoHistory[idx] = 1 / sTy;
    }

    x = xNew;
    fVal = fNew;
    grad = gradNew;
  }

  return { x, fVal, nIter: maxIter, converged };
}

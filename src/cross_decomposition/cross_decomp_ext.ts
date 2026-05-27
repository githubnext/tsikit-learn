/**
 * Extended cross-decomposition: CCA extensions, PLSSVD utilities,
 * and canonical correlation analysis helpers.
 */

/** Deflation step for PLS: subtract outer product of scores. */
export function deflate(
  X: Float64Array[],
  xScores: Float64Array,
  xLoadings: Float64Array,
): Float64Array[] {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  return X.map((xi, i) => {
    const t = xScores[i] ?? 0;
    return xi.map((v, j) => v - t * (xLoadings[j] ?? 0));
  });
}

/** NIPALS algorithm step: find first latent variable pair. */
export interface NIPALSResult {
  xWeights: Float64Array;
  yWeights: Float64Array;
  xScores: Float64Array;
  yScores: Float64Array;
  xLoadings: Float64Array;
  yLoadings: Float64Array;
}

export function nipalsStep(
  X: Float64Array[],
  Y: Float64Array[],
  maxIter = 500,
  tol = 1e-6,
): NIPALSResult {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  const q = Y[0]?.length ?? 0;

  // Initialize u as first column of Y
  let u = new Float64Array(n).map((_, i) => Y[i]?.[0] ?? 0);
  let xWeights = new Float64Array(p);
  let yWeights = new Float64Array(q);

  for (let iter = 0; iter < maxIter; iter++) {
    // w = X^T u / ||X^T u||
    const xw = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += (X[i]?.[j] ?? 0) * (u[i] ?? 0);
      xw[j] = sum;
    }
    const xwNorm = Math.sqrt(xw.reduce((s, v) => s + v * v, 0)) + 1e-10;
    for (let j = 0; j < p; j++) xw[j] = (xw[j] ?? 0) / xwNorm;

    // t = X w
    const t = new Float64Array(n).map((_, i) => {
      let sum = 0;
      for (let j = 0; j < p; j++) sum += (X[i]?.[j] ?? 0) * (xw[j] ?? 0);
      return sum;
    });

    // q = Y^T t / ||Y^T t||
    const yq = new Float64Array(q);
    for (let j = 0; j < q; j++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += (Y[i]?.[j] ?? 0) * (t[i] ?? 0);
      yq[j] = sum;
    }
    const yqNorm = Math.sqrt(yq.reduce((s, v) => s + v * v, 0)) + 1e-10;
    for (let j = 0; j < q; j++) yq[j] = (yq[j] ?? 0) / yqNorm;

    // u_new = Y q
    const uNew = new Float64Array(n).map((_, i) => {
      let sum = 0;
      for (let j = 0; j < q; j++) sum += (Y[i]?.[j] ?? 0) * (yq[j] ?? 0);
      return sum;
    });

    const diff = Math.sqrt(uNew.reduce((s, v, i) => s + (v - (u[i] ?? 0)) ** 2, 0));
    u = uNew;
    xWeights = xw;
    yWeights = yq;
    if (diff < tol) break;
  }

  const xScores = new Float64Array(n).map((_, i) => {
    let sum = 0;
    for (let j = 0; j < p; j++) sum += (X[i]?.[j] ?? 0) * (xWeights[j] ?? 0);
    return sum;
  });
  const yScores = u;

  // Loadings: X^T t / ||t||^2
  const tNorm2 = xScores.reduce((s, v) => s + v * v, 0) + 1e-10;
  const xLoadings = new Float64Array(p).map((_, j) => {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += (X[i]?.[j] ?? 0) * (xScores[i] ?? 0);
    return sum / tNorm2;
  });
  const uNorm2 = yScores.reduce((s, v) => s + v * v, 0) + 1e-10;
  const yLoadings = new Float64Array(q).map((_, j) => {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += (Y[i]?.[j] ?? 0) * (yScores[i] ?? 0);
    return sum / uNorm2;
  });

  return { xWeights, yWeights, xScores, yScores, xLoadings, yLoadings };
}

/** Canonical Correlation Analysis helpers. */
export interface CCAResult {
  xWeights: Float64Array[];
  yWeights: Float64Array[];
  xScores: Float64Array[];
  yScores: Float64Array[];
  correlations: Float64Array;
}

/** Compute canonical correlations between X and Y (simplified). */
export function canonicalCorrelations(
  X: Float64Array[],
  Y: Float64Array[],
  nComponents = 1,
): CCAResult {
  const nComp = Math.min(nComponents, X[0]?.length ?? 1, Y[0]?.length ?? 1);
  let XR = X;
  let YR = Y;
  const xWeights: Float64Array[] = [];
  const yWeights: Float64Array[] = [];
  const xScores: Float64Array[] = [];
  const yScores: Float64Array[] = [];
  const correlations = new Float64Array(nComp);

  for (let c = 0; c < nComp; c++) {
    const result = nipalsStep(XR, YR);
    xWeights.push(result.xWeights);
    yWeights.push(result.yWeights);
    xScores.push(result.xScores);
    yScores.push(result.yScores);

    // Correlation between t and u
    const tMean = result.xScores.reduce((s, v) => s + v, 0) / result.xScores.length;
    const uMean = result.yScores.reduce((s, v) => s + v, 0) / result.yScores.length;
    let cov = 0, st = 0, su = 0;
    for (let i = 0; i < result.xScores.length; i++) {
      cov += ((result.xScores[i] ?? 0) - tMean) * ((result.yScores[i] ?? 0) - uMean);
      st += ((result.xScores[i] ?? 0) - tMean) ** 2;
      su += ((result.yScores[i] ?? 0) - uMean) ** 2;
    }
    correlations[c] = cov / (Math.sqrt(st * su) + 1e-10);

    // Deflate
    XR = deflate(XR, result.xScores, result.xLoadings);
    YR = deflate(YR, result.yScores, result.yLoadings);
  }

  return { xWeights, yWeights, xScores, yScores, correlations };
}

/**
 * Deviance metrics: Poisson, Gamma, Tweedie, and related regression metrics.
 */

export function meanPoissonDeviance(yTrue: Float64Array, yPred: Float64Array): number {
  const n = yTrue.length;
  let dev = 0;
  for (let i = 0; i < n; i++) {
    const t = yTrue[i] ?? 0, p = Math.max(yPred[i] ?? 1e-10, 1e-10);
    dev += 2 * (t * Math.log((t + 1e-10) / p) - (t - p));
  }
  return dev / n;
}

export function meanGammaDeviance(yTrue: Float64Array, yPred: Float64Array): number {
  const n = yTrue.length;
  let dev = 0;
  for (let i = 0; i < n; i++) {
    const t = yTrue[i] ?? 0, p = Math.max(yPred[i] ?? 1e-10, 1e-10);
    dev += 2 * (-Math.log(t / p + 1e-10) + (t - p) / p);
  }
  return dev / n;
}

export function meanTweedieDeviance(yTrue: Float64Array, yPred: Float64Array, power = 0): number {
  if (power === 0) {
    const n = yTrue.length;
    let s = 0;
    for (let i = 0; i < n; i++) { const d = (yTrue[i] ?? 0) - (yPred[i] ?? 0); s += d * d; }
    return s / n;
  }
  if (power === 1) return meanPoissonDeviance(yTrue, yPred);
  if (power === 2) return meanGammaDeviance(yTrue, yPred);
  const n = yTrue.length;
  let dev = 0;
  for (let i = 0; i < n; i++) {
    const t = yTrue[i] ?? 0, p = Math.max(yPred[i] ?? 1e-10, 1e-10);
    const a = (2 - power), b = (1 - power);
    dev += 2 * (Math.pow(t, 2 - power) / a - t * Math.pow(p, 1 - power) / b + Math.pow(p, 2 - power) / a);
  }
  return dev / n;
}

export function medianAbsoluteError(yTrue: Float64Array, yPred: Float64Array): number {
  const n = yTrue.length;
  const errs = Array.from({ length: n }, (_, i) => Math.abs((yTrue[i] ?? 0) - (yPred[i] ?? 0)));
  errs.sort((a, b) => a - b);
  return n % 2 === 0
    ? ((errs[n / 2 - 1] ?? 0) + (errs[n / 2] ?? 0)) / 2
    : (errs[Math.floor(n / 2)] ?? 0);
}

export function maxError(yTrue: Float64Array, yPred: Float64Array): number {
  return Math.max(...Array.from({ length: yTrue.length }, (_, i) => Math.abs((yTrue[i] ?? 0) - (yPred[i] ?? 0))));
}

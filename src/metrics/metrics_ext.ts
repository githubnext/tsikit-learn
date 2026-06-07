/**
 * Metrics base utilities — additional scoring functions and helpers.
 */

export function meanPinballLoss(y: Float64Array, yPred: Float64Array, quantile = 0.5): number {
  const n = y.length;
  let loss = 0;
  for (let i = 0; i < n; i++) {
    const res = (y[i] ?? 0) - (yPred[i] ?? 0);
    loss += res >= 0 ? quantile * res : (quantile - 1) * res;
  }
  return loss / n;
}

export function meanTweedieDeviance(y: Float64Array, yPred: Float64Array, power = 0): number {
  const n = y.length;
  let dev = 0;
  for (let i = 0; i < n; i++) {
    const yi = y[i] ?? 0;
    const mi = Math.max(yPred[i] ?? 1e-8, 1e-8);
    if (power === 0) {
      dev += (yi - mi) ** 2;
    } else if (power === 1) {
      dev += 2 * (yi * Math.log(Math.max(yi, 1e-8) / mi) - (yi - mi));
    } else if (power === 2) {
      dev += 2 * (Math.log(mi / Math.max(yi, 1e-8)) + yi / mi - 1);
    } else {
      const t1 = Math.max(yi, 0) ** (2 - power) / ((1 - power) * (2 - power));
      const t2 = yi * mi ** (1 - power) / (1 - power);
      const t3 = mi ** (2 - power) / (2 - power);
      dev += 2 * (t1 - t2 + t3);
    }
  }
  return dev / n;
}

export function meanGammaDeviance(y: Float64Array, yPred: Float64Array): number {
  return meanTweedieDeviance(y, yPred, 2);
}

export function meanPoissonDeviance(y: Float64Array, yPred: Float64Array): number {
  return meanTweedieDeviance(y, yPred, 1);
}

export function d2TweedieScore(y: Float64Array, yPred: Float64Array, power = 0): number {
  const n = y.length;
  const yMean = y.reduce((a, b) => a + b, 0) / n;
  const nullPred = new Float64Array(n).fill(yMean);
  const devFull = meanTweedieDeviance(y, yPred, power);
  const devNull = meanTweedieDeviance(y, nullPred, power);
  return devNull === 0 ? 0 : 1 - devFull / devNull;
}

export function medianAbsoluteError(y: Float64Array, yPred: Float64Array): number {
  const errs = Array.from({ length: y.length }, (_, i) => Math.abs((y[i] ?? 0) - (yPred[i] ?? 0)));
  errs.sort((a, b) => a - b);
  return errs[Math.floor(errs.length / 2)] ?? 0;
}

export function maxError(y: Float64Array, yPred: Float64Array): number {
  return Math.max(...Array.from({ length: y.length }, (_, i) => Math.abs((y[i] ?? 0) - (yPred[i] ?? 0))));
}

export function r2Score(y: Float64Array, yPred: Float64Array, multiOutput: "raw_values" | "uniform_average" = "uniform_average"): number | Float64Array {
  if (y.length !== yPred.length) throw new Error("Length mismatch");
  const yMean = y.reduce((a, b) => a + b, 0) / y.length;
  let ss_res = 0, ss_tot = 0;
  for (let i = 0; i < y.length; i++) {
    ss_res += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
    ss_tot += ((y[i] ?? 0) - yMean) ** 2;
  }
  return ss_tot === 0 ? 0 : 1 - ss_res / ss_tot;
}

export function meanAbsolutePercentageError(y: Float64Array, yPred: Float64Array): number {
  const n = y.length;
  let err = 0;
  for (let i = 0; i < n; i++) {
    const yi = y[i] ?? 0;
    if (Math.abs(yi) > 1e-8) err += Math.abs(yi - (yPred[i] ?? 0)) / Math.abs(yi);
  }
  return err / n;
}

export function symmetricMeanAbsolutePercentageError(y: Float64Array, yPred: Float64Array): number {
  const n = y.length;
  let err = 0;
  for (let i = 0; i < n; i++) {
    const yi = y[i] ?? 0;
    const pi = yPred[i] ?? 0;
    const denom = (Math.abs(yi) + Math.abs(pi)) / 2;
    if (denom > 1e-8) err += Math.abs(yi - pi) / denom;
  }
  return err / n;
}

export function meanDirectionalAccuracy(y: Float64Array, yPred: Float64Array): number {
  if (y.length < 2) return 0;
  let correct = 0;
  for (let i = 1; i < y.length; i++) {
    const actualDir = (y[i] ?? 0) - (y[i - 1] ?? 0);
    const predDir = (yPred[i] ?? 0) - (yPred[i - 1] ?? 0);
    if ((actualDir >= 0 && predDir >= 0) || (actualDir < 0 && predDir < 0)) correct++;
  }
  return correct / (y.length - 1);
}

export function weightedMeanAbsoluteError(y: Float64Array, yPred: Float64Array, weights: Float64Array): number {
  const n = y.length;
  let num = 0, denom = 0;
  for (let i = 0; i < n; i++) {
    const w = weights[i] ?? 1;
    num += w * Math.abs((y[i] ?? 0) - (yPred[i] ?? 0));
    denom += w;
  }
  return denom > 0 ? num / denom : 0;
}

export function rootMeanSquaredLogError(y: Float64Array, yPred: Float64Array): number {
  const n = y.length;
  let err = 0;
  for (let i = 0; i < n; i++) {
    const a = Math.log1p(Math.max(y[i] ?? 0, 0));
    const b = Math.log1p(Math.max(yPred[i] ?? 0, 0));
    err += (a - b) ** 2;
  }
  return Math.sqrt(err / n);
}

export function concordanceCorrelationCoefficient(y: Float64Array, yPred: Float64Array): number {
  const n = y.length;
  const meanY = y.reduce((a, b) => a + b, 0) / n;
  const meanP = yPred.reduce((a, b) => a + b, 0) / n;
  let varY = 0, varP = 0, cov = 0;
  for (let i = 0; i < n; i++) {
    const dy = (y[i] ?? 0) - meanY;
    const dp = (yPred[i] ?? 0) - meanP;
    varY += dy * dy;
    varP += dp * dp;
    cov += dy * dp;
  }
  varY /= n; varP /= n; cov /= n;
  const denom = varY + varP + (meanY - meanP) ** 2;
  return denom > 0 ? 2 * cov / denom : 0;
}

export function normalizedRootMeanSquaredError(y: Float64Array, yPred: Float64Array): number {
  const n = y.length;
  let rmse = 0;
  for (let i = 0; i < n; i++) rmse += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
  rmse = Math.sqrt(rmse / n);
  const range = Math.max(...Array.from(y)) - Math.min(...Array.from(y));
  return range > 0 ? rmse / range : 0;
}

/**
 * D2 score metrics and additional regression metrics.
 * Mirrors sklearn.metrics.d2_tweedie_score, d2_absolute_error_score,
 * d2_pinball_score, mean_tweedie_deviance, mean_poisson_deviance,
 * mean_gamma_deviance.
 */

/**
 * Compute the D² score for Tweedie regression.
 * D² = 1 - deviance(y_true, y_pred) / deviance(y_true, y_null)
 * where y_null is the optimal constant predictor.
 *
 * Mirrors sklearn.metrics.d2_tweedie_score.
 */
export function d2TweedieScore(
  yTrue: Float64Array,
  yPred: Float64Array,
  power = 0
): number {
  const n = yTrue.length;

  function tweedieDeviance(y: Float64Array, mu: Float64Array): number {
    let dev = 0;
    for (let i = 0; i < n; i++) {
      const yi = y[i] ?? 0;
      const mui = mu[i] ?? 0;
      if (power === 0) {
        dev += (yi - mui) ** 2;
      } else if (power === 1) {
        // Poisson: 2*(y*log(y/mu) - (y - mu))
        const term = yi > 0 ? yi * Math.log(Math.max(yi / Math.max(mui, 1e-300), 1e-300)) - (yi - mui) : -yi + mui;
        dev += 2 * term;
      } else if (power === 2) {
        // Gamma: 2*(log(mu/y) + y/mu - 1)
        const muSafe = Math.max(mui, 1e-300);
        const ySafe = Math.max(yi, 1e-300);
        dev += 2 * (Math.log(muSafe / ySafe) + ySafe / muSafe - 1);
      } else {
        // General Tweedie
        const p = power;
        const muSafe = Math.max(mui, 1e-300);
        dev +=
          2 *
          ((Math.pow(Math.max(yi, 0), 2 - p) / ((1 - p) * (2 - p))) -
            (yi * Math.pow(muSafe, 1 - p)) / (1 - p) +
            Math.pow(muSafe, 2 - p) / (2 - p));
      }
    }
    return dev / n;
  }

  // Null model: optimal constant
  let nullMu = 0;
  let totalW = 0;
  for (let i = 0; i < n; i++) {
    const yi = yTrue[i] ?? 0;
    if (power <= 0) {
      nullMu += yi;
      totalW += 1;
    } else {
      // Weighted mean for Tweedie
      nullMu += yi;
      totalW += 1;
    }
  }
  nullMu /= Math.max(totalW, 1);
  const nullMuArr = new Float64Array(n).fill(nullMu);

  const devPred = tweedieDeviance(yTrue, yPred);
  const devNull = tweedieDeviance(yTrue, nullMuArr);
  return devNull === 0 ? 0 : 1 - devPred / devNull;
}

/**
 * Mean Tweedie deviance regression loss.
 * Mirrors sklearn.metrics.mean_tweedie_deviance.
 */
export function meanTweedieDeviance(
  yTrue: Float64Array,
  yPred: Float64Array,
  power = 0
): number {
  const n = yTrue.length;
  let dev = 0;
  for (let i = 0; i < n; i++) {
    const yi = yTrue[i] ?? 0;
    const mui = yPred[i] ?? 0;
    if (power === 0) {
      dev += (yi - mui) ** 2;
    } else if (power === 1) {
      const term = yi > 0 ? yi * Math.log(Math.max(yi / Math.max(mui, 1e-300), 1e-300)) - (yi - mui) : -yi + mui;
      dev += 2 * term;
    } else if (power === 2) {
      const muSafe = Math.max(mui, 1e-300);
      const ySafe = Math.max(yi, 1e-300);
      dev += 2 * (Math.log(muSafe / ySafe) + ySafe / muSafe - 1);
    } else {
      const p = power;
      const muSafe = Math.max(mui, 1e-300);
      dev +=
        2 *
        ((Math.pow(Math.max(yi, 0), 2 - p) / ((1 - p) * (2 - p))) -
          (yi * Math.pow(muSafe, 1 - p)) / (1 - p) +
          Math.pow(muSafe, 2 - p) / (2 - p));
    }
  }
  return dev / n;
}

/**
 * Mean Poisson deviance regression loss (power=1).
 * Mirrors sklearn.metrics.mean_poisson_deviance.
 */
export function meanPoissonDeviance(
  yTrue: Float64Array,
  yPred: Float64Array
): number {
  return meanTweedieDeviance(yTrue, yPred, 1);
}

/**
 * Mean Gamma deviance regression loss (power=2).
 * Mirrors sklearn.metrics.mean_gamma_deviance.
 */
export function meanGammaDeviance(
  yTrue: Float64Array,
  yPred: Float64Array
): number {
  return meanTweedieDeviance(yTrue, yPred, 2);
}

/**
 * D² score for absolute error (MAE-based).
 * D² = 1 - MAE(y_true, y_pred) / MAE(y_true, y_null)
 * where y_null is the median of y_true.
 *
 * Mirrors sklearn.metrics.d2_absolute_error_score.
 */
export function d2AbsoluteErrorScore(
  yTrue: Float64Array,
  yPred: Float64Array
): number {
  const n = yTrue.length;
  const sorted = Array.from(yTrue).sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median =
    sorted.length % 2 === 0
      ? ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2
      : (sorted[mid] ?? 0);

  let maePred = 0;
  let maeNull = 0;
  for (let i = 0; i < n; i++) {
    maePred += Math.abs((yTrue[i] ?? 0) - (yPred[i] ?? 0));
    maeNull += Math.abs((yTrue[i] ?? 0) - median);
  }
  maePred /= n;
  maeNull /= n;
  return maeNull === 0 ? 0 : 1 - maePred / maeNull;
}

/**
 * D² score for pinball loss (quantile regression).
 * Mirrors sklearn.metrics.d2_pinball_score.
 */
export function d2PinballScore(
  yTrue: Float64Array,
  yPred: Float64Array,
  alpha = 0.5
): number {
  const n = yTrue.length;

  function pinball(y: Float64Array, q: Float64Array): number {
    let loss = 0;
    for (let i = 0; i < n; i++) {
      const r = (y[i] ?? 0) - (q[i] ?? 0);
      loss += r >= 0 ? alpha * r : (alpha - 1) * r;
    }
    return loss / n;
  }

  // Null model: constant alpha-quantile
  const sorted = Array.from(yTrue).sort((a, b) => a - b);
  const qIdx = Math.min(Math.floor(alpha * n), n - 1);
  const nullQ = sorted[qIdx] ?? 0;
  const nullArr = new Float64Array(n).fill(nullQ);

  const pinballPred = pinball(yTrue, yPred);
  const pinballNull = pinball(yTrue, nullArr);
  return pinballNull === 0 ? 0 : 1 - pinballPred / pinballNull;
}

/**
 * Mean absolute percentage error (MAPE).
 * Mirrors sklearn.metrics.mean_absolute_percentage_error.
 */
export function meanAbsolutePercentageError(
  yTrue: Float64Array,
  yPred: Float64Array
): number {
  const n = yTrue.length;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const yi = yTrue[i] ?? 0;
    if (yi === 0) continue;
    s += Math.abs(((yi - (yPred[i] ?? 0)) / yi));
  }
  return s / n;
}

/**
 * Max error metric.
 * Mirrors sklearn.metrics.max_error.
 */
export function maxError(yTrue: Float64Array, yPred: Float64Array): number {
  let maxErr = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const e = Math.abs((yTrue[i] ?? 0) - (yPred[i] ?? 0));
    if (e > maxErr) maxErr = e;
  }
  return maxErr;
}

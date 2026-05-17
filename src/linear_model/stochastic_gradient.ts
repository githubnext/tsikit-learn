/**
 * Additional stochastic gradient descent utilities and loss functions.
 * Mirrors sklearn.linear_model._stochastic_gradient utilities.
 */

/** Squared loss: L(y, f) = 0.5 * (y - f)^2 */
export function squaredLoss(y: number, f: number): number {
  const diff = y - f;
  return 0.5 * diff * diff;
}

/** Squared loss gradient w.r.t. f */
export function squaredLossGrad(y: number, f: number): number {
  return f - y;
}

/** Hinge loss: L(y, f) = max(0, 1 - y*f) */
export function hingeLoss(y: number, f: number): number {
  return Math.max(0, 1 - y * f);
}

/** Hinge loss gradient w.r.t. f */
export function hingeLossGrad(y: number, f: number): number {
  return y * f < 1 ? -y : 0;
}

/** Squared hinge loss: L(y, f) = max(0, 1 - y*f)^2 */
export function squaredHingeLoss(y: number, f: number): number {
  const h = Math.max(0, 1 - y * f);
  return h * h;
}

/** Squared hinge loss gradient */
export function squaredHingeLossGrad(y: number, f: number): number {
  const h = Math.max(0, 1 - y * f);
  return h > 0 ? -2 * y * h : 0;
}

/** Log loss (logistic): L(y, f) = log(1 + exp(-y*f)) */
export function logLossFn(y: number, f: number): number {
  const z = y * f;
  if (z > 18) return Math.exp(-z);
  if (z < -18) return -z;
  return Math.log(1 + Math.exp(-z));
}

/** Log loss gradient w.r.t. f */
export function logLossGrad(y: number, f: number): number {
  const z = y * f;
  if (z > 18) return -y * Math.exp(-z);
  return -y / (1 + Math.exp(z));
}

/** Epsilon-insensitive loss (SVR): L(y, f) = max(0, |y - f| - eps) */
export function epsilonInsensitiveLoss(y: number, f: number, eps = 0.1): number {
  return Math.max(0, Math.abs(y - f) - eps);
}

/** Epsilon-insensitive loss gradient */
export function epsilonInsensitiveLossGrad(y: number, f: number, eps = 0.1): number {
  const diff = f - y;
  if (Math.abs(diff) > eps) return diff > 0 ? 1 : -1;
  return 0;
}

/** Huber loss for regression */
export function huberLossFn(y: number, f: number, delta = 1.0): number {
  const diff = Math.abs(y - f);
  return diff <= delta ? 0.5 * diff * diff : delta * (diff - 0.5 * delta);
}

/** Huber loss gradient */
export function huberLossGrad(y: number, f: number, delta = 1.0): number {
  const diff = f - y;
  return Math.abs(diff) <= delta ? diff : delta * Math.sign(diff);
}

export type LossFunction = "squared" | "hinge" | "squared_hinge" | "log" | "epsilon_insensitive" | "huber";

/** Evaluate loss value for a named loss function. */
export function evalLoss(loss: LossFunction, y: number, f: number, extra = 0.1): number {
  switch (loss) {
    case "squared": return squaredLoss(y, f);
    case "hinge": return hingeLoss(y, f);
    case "squared_hinge": return squaredHingeLoss(y, f);
    case "log": return logLossFn(y, f);
    case "epsilon_insensitive": return epsilonInsensitiveLoss(y, f, extra);
    case "huber": return huberLossFn(y, f, extra);
  }
}

/** Evaluate loss gradient for a named loss function. */
export function evalLossGrad(loss: LossFunction, y: number, f: number, extra = 0.1): number {
  switch (loss) {
    case "squared": return squaredLossGrad(y, f);
    case "hinge": return hingeLossGrad(y, f);
    case "squared_hinge": return squaredHingeLossGrad(y, f);
    case "log": return logLossGrad(y, f);
    case "epsilon_insensitive": return epsilonInsensitiveLossGrad(y, f, extra);
    case "huber": return huberLossGrad(y, f, extra);
  }
}

/**
 * Compute L1/L2 penalty gradient contribution.
 * Returns the gradient of alpha * (l1_ratio * ||w||_1 + 0.5 * (1 - l1_ratio) * ||w||_2^2).
 */
export function penaltyGrad(w: Float64Array, alpha: number, l1Ratio: number): Float64Array {
  const n = w.length;
  const grad = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const wj = w[j] ?? 0;
    grad[j] = alpha * (l1Ratio * Math.sign(wj) + (1 - l1Ratio) * wj);
  }
  return grad;
}

/** Apply proximal operator for L1 regularization (soft thresholding). */
export function softThreshold(w: Float64Array, threshold: number): Float64Array {
  const out = new Float64Array(w.length);
  for (let j = 0; j < w.length; j++) {
    const wj = w[j] ?? 0;
    out[j] = Math.sign(wj) * Math.max(0, Math.abs(wj) - threshold);
  }
  return out;
}

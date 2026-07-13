/**
 * Convergence checking and numerical utilities for sklearn optimization.
 */

export class ConvergenceMonitor {
  tol: number;
  patience: number;
  mode: "min" | "max";
  private bestValue: number;
  private noImprovementCount: number = 0;
  private history: number[] = [];

  constructor(tol = 1e-4, patience = 10, mode: "min" | "max" = "min") {
    this.tol = tol;
    this.patience = patience;
    this.mode = mode;
    this.bestValue = mode === "min" ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
  }

  update(value: number): boolean {
    this.history.push(value);
    const improved = this.mode === "min"
      ? value < this.bestValue - this.tol
      : value > this.bestValue + this.tol;

    if (improved) {
      this.bestValue = value;
      this.noImprovementCount = 0;
    } else {
      this.noImprovementCount++;
    }
    return this.noImprovementCount >= this.patience;
  }

  hasConverged(): boolean {
    return this.noImprovementCount >= this.patience;
  }

  getHistory(): number[] {
    return this.history.slice();
  }

  reset(): void {
    this.bestValue = this.mode === "min" ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
    this.noImprovementCount = 0;
    this.history = [];
  }
}

export class GradientNormChecker {
  tol: number;
  private norms: number[] = [];

  constructor(tol = 1e-6) {
    this.tol = tol;
  }

  check(gradient: Float64Array): boolean {
    const norm = Math.sqrt(gradient.reduce((s, v) => s + v * v, 0));
    this.norms.push(norm);
    return norm < this.tol;
  }

  lastNorm(): number {
    return this.norms[this.norms.length - 1] ?? 0;
  }

  history(): number[] {
    return this.norms.slice();
  }
}

export class Backtracking {
  alpha0: number;
  rho: number;
  c: number;
  maxIter: number;

  constructor(alpha0 = 1.0, rho = 0.5, c = 1e-4, maxIter = 100) {
    this.alpha0 = alpha0;
    this.rho = rho;
    this.c = c;
    this.maxIter = maxIter;
  }

  search(
    fn: (x: Float64Array) => number,
    x: Float64Array,
    grad: Float64Array,
    direction: Float64Array,
  ): number {
    const fx = fn(x);
    const slope = grad.reduce((s, g, i) => s + g * (direction[i] ?? 0), 0);
    let alpha = this.alpha0;
    for (let i = 0; i < this.maxIter; i++) {
      const xNew = x.map((xi, j) => xi + alpha * (direction[j] ?? 0));
      if (fn(xNew) <= fx + this.c * alpha * slope) break;
      alpha *= this.rho;
    }
    return alpha;
  }
}

export function finiteDifferenceGradient(fn: (x: Float64Array) => number, x: Float64Array, eps = 1e-5): Float64Array {
  const grad = new Float64Array(x.length);
  const f0 = fn(x);
  for (let i = 0; i < x.length; i++) {
    const xh = new Float64Array(x);
    xh[i]! += eps;
    grad[i] = (fn(xh) - f0) / eps;
  }
  return grad;
}

export function checkGradient(
  fn: (x: Float64Array) => number,
  gradFn: (x: Float64Array) => Float64Array,
  x: Float64Array,
  eps = 1e-5,
  rtol = 1e-3,
): { maxRelError: number; passed: boolean } {
  const numerical = finiteDifferenceGradient(fn, x, eps);
  const analytical = gradFn(x);
  let maxRelError = 0;
  for (let i = 0; i < x.length; i++) {
    const n = numerical[i] ?? 0;
    const a = analytical[i] ?? 0;
    const denom = Math.max(Math.abs(n), Math.abs(a), 1e-8);
    maxRelError = Math.max(maxRelError, Math.abs(n - a) / denom);
  }
  return { maxRelError, passed: maxRelError < rtol };
}

export function wolfeLineSearch(
  fn: (x: Float64Array) => number,
  gradFn: (x: Float64Array) => Float64Array,
  x: Float64Array,
  direction: Float64Array,
  maxIter = 25,
  c1 = 1e-4,
  c2 = 0.9,
): number {
  const fx = fn(x);
  const gx = gradFn(x);
  const dirDotGrad = gx.reduce((s, g, i) => s + g * (direction[i] ?? 0), 0);

  let alpha = 1.0, lo = 0, hi = Number.POSITIVE_INFINITY;
  for (let iter = 0; iter < maxIter; iter++) {
    const xNew = x.map((xi, i) => xi + alpha * (direction[i] ?? 0));
    const fNew = fn(xNew);
    if (fNew > fx + c1 * alpha * dirDotGrad) {
      hi = alpha;
      alpha = (lo + hi) / 2;
    } else {
      const gNew = gradFn(xNew);
      const newSlope = gNew.reduce((s, g, i) => s + g * (direction[i] ?? 0), 0);
      if (newSlope < c2 * dirDotGrad) {
        lo = alpha;
        alpha = hi === Number.POSITIVE_INFINITY ? alpha * 2 : (lo + hi) / 2;
      } else {
        break;
      }
    }
  }
  return alpha;
}

export class NesterovMomentum {
  learningRate: number;
  momentum: number;
  private velocity: Float64Array | null = null;

  constructor(learningRate = 0.01, momentum = 0.9) {
    this.learningRate = learningRate;
    this.momentum = momentum;
  }

  step(params: Float64Array, gradient: Float64Array): Float64Array {
    if (!this.velocity) this.velocity = new Float64Array(params.length);
    const v = this.velocity;
    const newParams = new Float64Array(params.length);
    for (let i = 0; i < params.length; i++) {
      const vOld = v[i] ?? 0;
      v[i] = this.momentum * vOld - this.learningRate * (gradient[i] ?? 0);
      newParams[i] = (params[i] ?? 0) - this.momentum * vOld + (1 + this.momentum) * (v[i] ?? 0);
    }
    return newParams;
  }

  reset(): void {
    this.velocity = null;
  }
}

export class AdaGrad {
  learningRate: number;
  epsilon: number;
  private accumG: Float64Array | null = null;

  constructor(learningRate = 0.01, epsilon = 1e-8) {
    this.learningRate = learningRate;
    this.epsilon = epsilon;
  }

  step(params: Float64Array, gradient: Float64Array): Float64Array {
    if (!this.accumG) this.accumG = new Float64Array(params.length);
    const g = this.accumG;
    const result = new Float64Array(params.length);
    for (let i = 0; i < params.length; i++) {
      g[i] = (g[i] ?? 0) + (gradient[i] ?? 0) ** 2;
      result[i] = (params[i] ?? 0) - this.learningRate * (gradient[i] ?? 0) / Math.sqrt((g[i] ?? 0) + this.epsilon);
    }
    return result;
  }

  reset(): void {
    this.accumG = null;
  }
}

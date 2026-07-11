/**
 * Neural network optimizers: Adam, SGD, Adagrad, RMSProp, Adadelta, Nadam.
 */

export interface OptimizerState {
  step: number;
  m?: Float64Array;
  v?: Float64Array;
  g2?: Float64Array;
  dx?: Float64Array;
}

export interface Optimizer {
  initialize(nParams: number): OptimizerState;
  update(
    params: Float64Array,
    grads: Float64Array,
    state: OptimizerState,
  ): void;
}

export class AdamOptimizer implements Optimizer {
  constructor(
    private readonly lr = 1e-3,
    private readonly beta1 = 0.9,
    private readonly beta2 = 0.999,
    private readonly eps = 1e-8,
  ) {}

  initialize(nParams: number): OptimizerState {
    return {
      step: 0,
      m: new Float64Array(nParams),
      v: new Float64Array(nParams),
    };
  }

  update(
    params: Float64Array,
    grads: Float64Array,
    state: OptimizerState,
  ): void {
    state.step++;
    const m = state.m!;
    const v = state.v!;
    const bc1 = 1 - this.beta1 ** state.step;
    const bc2 = 1 - this.beta2 ** state.step;
    for (let i = 0; i < params.length; i++) {
      const g = grads[i] ?? 0;
      m[i] = this.beta1 * (m[i] ?? 0) + (1 - this.beta1) * g;
      v[i] = this.beta2 * (v[i] ?? 0) + (1 - this.beta2) * g * g;
      const mHat = (m[i] ?? 0) / bc1;
      const vHat = (v[i] ?? 0) / bc2;
      params[i] =
        (params[i] ?? 0) - (this.lr * mHat) / (Math.sqrt(vHat) + this.eps);
    }
  }
}

export class SGDOptimizer implements Optimizer {
  constructor(
    private readonly lr = 0.01,
    private readonly momentum = 0.0,
    private readonly nesterov = false,
    private readonly weightDecay = 0.0,
  ) {}

  initialize(nParams: number): OptimizerState {
    return { step: 0, v: new Float64Array(nParams) };
  }

  update(
    params: Float64Array,
    grads: Float64Array,
    state: OptimizerState,
  ): void {
    state.step++;
    const v = state.v!;
    for (let i = 0; i < params.length; i++) {
      const g = (grads[i] ?? 0) + this.weightDecay * (params[i] ?? 0);
      v[i] = this.momentum * (v[i] ?? 0) - this.lr * g;
      if (this.nesterov) {
        params[i] =
          (params[i] ?? 0) + this.momentum * (v[i] ?? 0) - this.lr * g;
      } else {
        params[i] = (params[i] ?? 0) + (v[i] ?? 0);
      }
    }
  }
}

export class AdagradOptimizer implements Optimizer {
  constructor(
    private readonly lr = 0.01,
    private readonly eps = 1e-8,
    private readonly initAccumulator = 0.1,
  ) {}

  initialize(nParams: number): OptimizerState {
    return {
      step: 0,
      g2: new Float64Array(nParams).fill(this.initAccumulator),
    };
  }

  update(
    params: Float64Array,
    grads: Float64Array,
    state: OptimizerState,
  ): void {
    state.step++;
    const g2 = state.g2!;
    for (let i = 0; i < params.length; i++) {
      const g = grads[i] ?? 0;
      g2[i] = (g2[i] ?? 0) + g * g;
      params[i] =
        (params[i] ?? 0) - (this.lr * g) / (Math.sqrt(g2[i] ?? 0) + this.eps);
    }
  }
}

export class RMSPropOptimizer implements Optimizer {
  constructor(
    private readonly lr = 0.01,
    private readonly alpha = 0.99,
    private readonly eps = 1e-8,
    private readonly momentum = 0.0,
  ) {}

  initialize(nParams: number): OptimizerState {
    return {
      step: 0,
      v: new Float64Array(nParams),
      m: new Float64Array(nParams),
    };
  }

  update(
    params: Float64Array,
    grads: Float64Array,
    state: OptimizerState,
  ): void {
    state.step++;
    const v = state.v!;
    const m = state.m!;
    for (let i = 0; i < params.length; i++) {
      const g = grads[i] ?? 0;
      v[i] = this.alpha * (v[i] ?? 0) + (1 - this.alpha) * g * g;
      const buf = (this.lr * g) / (Math.sqrt(v[i] ?? 0) + this.eps);
      m[i] = this.momentum * (m[i] ?? 0) + buf;
      params[i] = (params[i] ?? 0) - (m[i] ?? 0);
    }
  }
}

export class AdadeltaOptimizer implements Optimizer {
  constructor(
    private readonly rho = 0.95,
    private readonly eps = 1e-6,
  ) {}

  initialize(nParams: number): OptimizerState {
    return {
      step: 0,
      v: new Float64Array(nParams),
      dx: new Float64Array(nParams),
    };
  }

  update(
    params: Float64Array,
    grads: Float64Array,
    state: OptimizerState,
  ): void {
    state.step++;
    const v = state.v!;
    const dx = state.dx!;
    for (let i = 0; i < params.length; i++) {
      const g = grads[i] ?? 0;
      v[i] = this.rho * (v[i] ?? 0) + (1 - this.rho) * g * g;
      const update =
        (Math.sqrt((dx[i] ?? 0) + this.eps) /
          Math.sqrt((v[i] ?? 0) + this.eps)) *
        g;
      dx[i] = this.rho * (dx[i] ?? 0) + (1 - this.rho) * update * update;
      params[i] = (params[i] ?? 0) - update;
    }
  }
}

export class NadamOptimizer implements Optimizer {
  constructor(
    private readonly lr = 2e-3,
    private readonly beta1 = 0.9,
    private readonly beta2 = 0.999,
    private readonly eps = 1e-8,
  ) {}

  initialize(nParams: number): OptimizerState {
    return {
      step: 0,
      m: new Float64Array(nParams),
      v: new Float64Array(nParams),
    };
  }

  update(
    params: Float64Array,
    grads: Float64Array,
    state: OptimizerState,
  ): void {
    state.step++;
    const m = state.m!;
    const v = state.v!;
    const bc1 = 1 - this.beta1 ** state.step;
    const bc2 = 1 - this.beta2 ** state.step;
    for (let i = 0; i < params.length; i++) {
      const g = grads[i] ?? 0;
      m[i] = this.beta1 * (m[i] ?? 0) + (1 - this.beta1) * g;
      v[i] = this.beta2 * (v[i] ?? 0) + (1 - this.beta2) * g * g;
      const mHat =
        (this.beta1 * (m[i] ?? 0)) / bc1 + ((1 - this.beta1) * g) / bc1;
      const vHat = (v[i] ?? 0) / bc2;
      params[i] =
        (params[i] ?? 0) - (this.lr * mHat) / (Math.sqrt(vHat) + this.eps);
    }
  }
}

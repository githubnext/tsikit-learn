/**
 * Extended neural network utilities: additional activation functions,
 * learning rate schedulers, weight initializers, and layer utilities.
 */

/** Activation function types. */
export type ActivationType = "relu" | "sigmoid" | "tanh" | "softmax" | "leaky_relu" | "elu" | "swish" | "gelu" | "mish";

/** Apply activation function element-wise. */
export function activate(x: Float64Array, fn: ActivationType, alpha = 0.01): Float64Array {
  return x.map((v) => {
    switch (fn) {
      case "relu": return Math.max(0, v);
      case "sigmoid": return 1 / (1 + Math.exp(-v));
      case "tanh": return Math.tanh(v);
      case "leaky_relu": return v >= 0 ? v : alpha * v;
      case "elu": return v >= 0 ? v : alpha * (Math.exp(v) - 1);
      case "swish": return v / (1 + Math.exp(-v));
      case "gelu": return 0.5 * v * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v ** 3)));
      case "mish": return v * Math.tanh(Math.log(1 + Math.exp(v)));
      case "softmax": return v;  // softmax handled separately
    }
  });
}

/** Softmax over a 1D vector. */
export function softmax(x: Float64Array): Float64Array {
  const maxV = x.reduce((a, b) => Math.max(a, b), Number.NEGATIVE_INFINITY);
  const exps = x.map((v) => Math.exp(v - maxV));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / (sum + 1e-10));
}

/** Activation derivative. */
export function activateDerivative(x: Float64Array, fn: ActivationType, alpha = 0.01): Float64Array {
  return x.map((v) => {
    switch (fn) {
      case "relu": return v > 0 ? 1 : 0;
      case "sigmoid": { const s = 1 / (1 + Math.exp(-v)); return s * (1 - s); }
      case "tanh": return 1 - Math.tanh(v) ** 2;
      case "leaky_relu": return v >= 0 ? 1 : alpha;
      case "elu": return v >= 0 ? 1 : alpha * Math.exp(v);
      case "swish": { const s = 1 / (1 + Math.exp(-v)); return s + v * s * (1 - s); }
      case "gelu": {
        const cdf = 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v ** 3)));
        return cdf + v * (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * v * v);
      }
      case "mish": {
        const sp = Math.log(1 + Math.exp(v));
        const delta = Math.tanh(sp);
        return delta + v * (1 - delta ** 2) * (1 / (1 + Math.exp(-v)));
      }
      case "softmax": return 1;
    }
  });
}

/** Weight initialization: Xavier/Glorot uniform. */
export function glorotUniform(fanIn: number, fanOut: number, shape: [number, number]): Float64Array[] {
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  return Array.from({ length: shape[0] }, () =>
    new Float64Array(shape[1]).map(() => (Math.random() * 2 - 1) * limit)
  );
}

/** Weight initialization: He (Kaiming) normal for ReLU. */
export function heNormal(fanIn: number, shape: [number, number]): Float64Array[] {
  const std = Math.sqrt(2 / fanIn);
  return Array.from({ length: shape[0] }, () =>
    new Float64Array(shape[1]).map(() => gaussianRandom(0, std))
  );
}

function gaussianRandom(mean: number, std: number): number {
  const u1 = Math.random(), u2 = Math.random();
  return mean + std * Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
}

/** Learning rate schedule: cosine annealing. */
export function cosineAnnealingLR(iteration: number, maxIter: number, etaMin = 0, etaMax = 0.01): number {
  return etaMin + 0.5 * (etaMax - etaMin) * (1 + Math.cos(Math.PI * iteration / maxIter));
}

/** Learning rate schedule: step decay. */
export function stepDecayLR(iteration: number, initialLR: number, dropFactor: number, dropEvery: number): number {
  return initialLR * (dropFactor ** Math.floor(iteration / dropEvery));
}

/** Batch normalization: normalize activations. */
export function batchNorm(
  X: Float64Array[],
  gamma: Float64Array,
  beta: Float64Array,
  eps = 1e-5,
): Float64Array[] {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const mean = new Float64Array(d);
  const variance = new Float64Array(d);

  for (const xi of X) {
    for (let j = 0; j < d; j++) mean[j] = (mean[j] ?? 0) + (xi[j] ?? 0) / n;
  }
  for (const xi of X) {
    for (let j = 0; j < d; j++) variance[j] = (variance[j] ?? 0) + ((xi[j] ?? 0) - (mean[j] ?? 0)) ** 2 / n;
  }

  return X.map((xi) =>
    xi.map((v, j) => {
      const normalized = (v - (mean[j] ?? 0)) / Math.sqrt((variance[j] ?? 0) + eps);
      return (gamma[j] ?? 1) * normalized + (beta[j] ?? 0);
    })
  );
}

/** Layer normalization: normalize across features per sample. */
export function layerNorm(x: Float64Array, gamma: Float64Array, beta: Float64Array, eps = 1e-5): Float64Array {
  const mean = x.reduce((a, b) => a + b, 0) / x.length;
  const variance = x.reduce((a, b) => a + (b - mean) ** 2, 0) / x.length;
  return x.map((v, j) => {
    const normalized = (v - mean) / Math.sqrt(variance + eps);
    return (gamma[j] ?? 1) * normalized + (beta[j] ?? 0);
  });
}

/** Dropout: randomly zero out activations. */
export function dropout(x: Float64Array, rate: number, training: boolean): Float64Array {
  if (!training || rate === 0) return x;
  const scale = 1 / (1 - rate);
  return x.map((v) => Math.random() > rate ? v * scale : 0);
}

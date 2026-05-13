/**
 * MLP Classifier and Regressor (Multi-Layer Perceptron).
 * Mirrors sklearn.neural_network.MLPClassifier / MLPRegressor.
 */

import { NotFittedError } from "../exceptions.js";

function relu(x: number): number {
  return Math.max(0, x);
}

function reluDeriv(x: number): number {
  return x > 0 ? 1 : 0;
}

function tanhDeriv(x: number): number {
  const t = Math.tanh(x);
  return 1 - t * t;
}

function softmax(arr: Float64Array): Float64Array {
  const maxVal = Math.max(...arr);
  const exp = arr.map((x) => Math.exp(x - maxVal));
  const sum = exp.reduce((a, b) => a + b, 0);
  return new Float64Array(exp.map((x) => x / sum));
}

type ActivationFn = (x: number) => number;
type ActivationDerivFn = (x: number) => number;

function getActivation(name: string): [ActivationFn, ActivationDerivFn] {
  if (name === "relu") return [relu, reluDeriv];
  if (name === "tanh") return [Math.tanh, tanhDeriv];
  // logistic
  const sig = (x: number) => 1 / (1 + Math.exp(-x));
  return [sig, (x: number) => { const s = sig(x); return s * (1 - s); }];
}

interface LayerWeights {
  W: Float64Array[];
  b: Float64Array;
}

export class MLPClassifier {
  hiddenLayerSizes: number[];
  activation: string;
  alpha: number;
  learningRate: number;
  maxIter: number;
  tol: number;
  batchSize: number;

  coefs_: LayerWeights[] | null = null;
  classes_: Float64Array | null = null;
  nOutputs_: number = 0;

  constructor(
    options: {
      hiddenLayerSizes?: number[];
      activation?: string;
      alpha?: number;
      learningRate?: number;
      maxIter?: number;
      tol?: number;
      batchSize?: number;
    } = {},
  ) {
    this.hiddenLayerSizes = options.hiddenLayerSizes ?? [100];
    this.activation = options.activation ?? "relu";
    this.alpha = options.alpha ?? 1e-4;
    this.learningRate = options.learningRate ?? 1e-3;
    this.maxIter = options.maxIter ?? 200;
    this.tol = options.tol ?? 1e-4;
    this.batchSize = options.batchSize ?? 32;
  }

  private _initWeights(layerSizes: number[]): LayerWeights[] {
    const weights: LayerWeights[] = [];
    for (let i = 0; i < layerSizes.length - 1; i++) {
      const fan_in = layerSizes[i] ?? 1;
      const fan_out = layerSizes[i + 1] ?? 1;
      const scale = Math.sqrt(2 / fan_in);
      const W: Float64Array[] = [];
      for (let r = 0; r < fan_out; r++) {
        const row = new Float64Array(fan_in);
        for (let c = 0; c < fan_in; c++) {
          row[c] = (Math.random() * 2 - 1) * scale;
        }
        W.push(row);
      }
      weights.push({ W, b: new Float64Array(fan_out) });
    }
    return weights;
  }

  private _forward(
    x: Float64Array,
    weights: LayerWeights[],
    activFn: ActivationFn,
    isOutput = false,
  ): { activations: Float64Array[]; zs: Float64Array[] } {
    const activations: Float64Array[] = [x];
    const zs: Float64Array[] = [];

    for (let l = 0; l < weights.length; l++) {
      const layer = weights[l] as LayerWeights;
      const prev = activations[activations.length - 1] as Float64Array;
      const z = new Float64Array(layer.W.length);
      for (let j = 0; j < layer.W.length; j++) {
        let sum = layer.b[j] ?? 0;
        const wRow = layer.W[j] ?? new Float64Array(0);
        for (let k = 0; k < prev.length; k++) {
          sum += (wRow[k] ?? 0) * (prev[k] ?? 0);
        }
        z[j] = sum;
      }
      zs.push(z);

      const isLast = l === weights.length - 1;
      let a: Float64Array;
      if (isLast && isOutput) {
        a = softmax(z);
      } else if (isLast && !isOutput) {
        a = new Float64Array(z);
      } else {
        a = new Float64Array(z.map(activFn));
      }
      activations.push(a);
    }

    return { activations, zs };
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const nFeatures = (X[0] ?? new Float64Array(0)).length;
    const uniqueClasses = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
    this.classes_ = new Float64Array(uniqueClasses);
    const nClasses = uniqueClasses.length;
    this.nOutputs_ = nClasses;

    const classToIdx = new Map(uniqueClasses.map((c, i) => [c, i]));
    const [activFn, activDeriv] = getActivation(this.activation);

    const layerSizes = [nFeatures, ...this.hiddenLayerSizes, nClasses];
    const weights = this._initWeights(layerSizes);

    for (let iter = 0; iter < this.maxIter; iter++) {
      let totalLoss = 0;

      for (let i = 0; i < n; i++) {
        const xi = X[i] ?? new Float64Array(nFeatures);
        const yi = classToIdx.get(y[i] ?? 0) ?? 0;
        const yOneHot = new Float64Array(nClasses);
        yOneHot[yi] = 1;

        const { activations, zs } = this._forward(xi, weights, activFn, true);
        const output = activations[activations.length - 1] as Float64Array;

        // Cross-entropy loss
        totalLoss += -Math.log((output[yi] ?? 0) + 1e-15);

        // Backprop
        const deltas: Float64Array[] = new Array(weights.length);
        // Output delta
        const outDelta = new Float64Array(nClasses);
        for (let j = 0; j < nClasses; j++) {
          outDelta[j] = (output[j] ?? 0) - (yOneHot[j] ?? 0);
        }
        deltas[weights.length - 1] = outDelta;

        for (let l = weights.length - 2; l >= 0; l--) {
          const nextLayer = weights[l + 1] as LayerWeights;
          const nextDelta = deltas[l + 1] as Float64Array;
          const z = zs[l] as Float64Array;
          const delta = new Float64Array(z.length);
          for (let j = 0; j < z.length; j++) {
            let sum = 0;
            for (let k = 0; k < nextLayer.W.length; k++) {
              sum += ((nextLayer.W[k] ?? new Float64Array(0))[j] ?? 0) * (nextDelta[k] ?? 0);
            }
            delta[j] = sum * activDeriv(z[j] ?? 0);
          }
          deltas[l] = delta;
        }

        // Update weights
        for (let l = 0; l < weights.length; l++) {
          const layer = weights[l] as LayerWeights;
          const prevA = activations[l] as Float64Array;
          const delta = deltas[l] as Float64Array;
          for (let j = 0; j < layer.W.length; j++) {
            const wRow = layer.W[j] as Float64Array;
            for (let k = 0; k < prevA.length; k++) {
              wRow[k] =
                (wRow[k] ?? 0) -
                this.learningRate * ((delta[j] ?? 0) * (prevA[k] ?? 0) + this.alpha * (wRow[k] ?? 0));
            }
            layer.b[j] = (layer.b[j] ?? 0) - this.learningRate * (delta[j] ?? 0);
          }
        }
      }

      if (totalLoss / n < this.tol) break;
    }

    this.coefs_ = weights;
    return this;
  }

  predictProba(X: Float64Array[]): Float64Array[] {
    if (this.coefs_ === null) throw new NotFittedError("MLPClassifier");
    const [activFn] = getActivation(this.activation);
    return X.map((xi) => {
      const { activations } = this._forward(xi, this.coefs_ as LayerWeights[], activFn, true);
      return activations[activations.length - 1] as Float64Array;
    });
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.classes_ === null) throw new NotFittedError("MLPClassifier");
    const proba = this.predictProba(X);
    const classes = this.classes_;
    return new Float64Array(
      proba.map((p) => {
        let maxIdx = 0;
        let maxVal = p[0] ?? 0;
        for (let j = 1; j < p.length; j++) {
          if ((p[j] ?? 0) > maxVal) {
            maxVal = p[j] ?? 0;
            maxIdx = j;
          }
        }
        return classes[maxIdx] ?? 0;
      }),
    );
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      if (pred[i] === y[i]) correct++;
    }
    return correct / y.length;
  }
}

export class MLPRegressor {
  hiddenLayerSizes: number[];
  activation: string;
  alpha: number;
  learningRate: number;
  maxIter: number;
  tol: number;

  coefs_: LayerWeights[] | null = null;

  constructor(
    options: {
      hiddenLayerSizes?: number[];
      activation?: string;
      alpha?: number;
      learningRate?: number;
      maxIter?: number;
      tol?: number;
    } = {},
  ) {
    this.hiddenLayerSizes = options.hiddenLayerSizes ?? [100];
    this.activation = options.activation ?? "relu";
    this.alpha = options.alpha ?? 1e-4;
    this.learningRate = options.learningRate ?? 1e-3;
    this.maxIter = options.maxIter ?? 200;
    this.tol = options.tol ?? 1e-4;
  }

  private _initWeights(layerSizes: number[]): LayerWeights[] {
    const weights: LayerWeights[] = [];
    for (let i = 0; i < layerSizes.length - 1; i++) {
      const fan_in = layerSizes[i] ?? 1;
      const fan_out = layerSizes[i + 1] ?? 1;
      const scale = Math.sqrt(2 / fan_in);
      const W: Float64Array[] = [];
      for (let r = 0; r < fan_out; r++) {
        const row = new Float64Array(fan_in);
        for (let c = 0; c < fan_in; c++) {
          row[c] = (Math.random() * 2 - 1) * scale;
        }
        W.push(row);
      }
      weights.push({ W, b: new Float64Array(fan_out) });
    }
    return weights;
  }

  private _forward(
    x: Float64Array,
    weights: LayerWeights[],
    activFn: ActivationFn,
  ): { activations: Float64Array[]; zs: Float64Array[] } {
    const activations: Float64Array[] = [x];
    const zs: Float64Array[] = [];

    for (let l = 0; l < weights.length; l++) {
      const layer = weights[l] as LayerWeights;
      const prev = activations[activations.length - 1] as Float64Array;
      const z = new Float64Array(layer.W.length);
      for (let j = 0; j < layer.W.length; j++) {
        let sum = layer.b[j] ?? 0;
        const wRow = layer.W[j] ?? new Float64Array(0);
        for (let k = 0; k < prev.length; k++) {
          sum += (wRow[k] ?? 0) * (prev[k] ?? 0);
        }
        z[j] = sum;
      }
      zs.push(z);
      const isLast = l === weights.length - 1;
      activations.push(isLast ? new Float64Array(z) : new Float64Array(z.map(activFn)));
    }
    return { activations, zs };
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const nFeatures = (X[0] ?? new Float64Array(0)).length;
    const [activFn, activDeriv] = getActivation(this.activation);

    const layerSizes = [nFeatures, ...this.hiddenLayerSizes, 1];
    const weights = this._initWeights(layerSizes);

    for (let iter = 0; iter < this.maxIter; iter++) {
      let totalLoss = 0;
      for (let i = 0; i < n; i++) {
        const xi = X[i] ?? new Float64Array(nFeatures);
        const { activations, zs } = this._forward(xi, weights, activFn);
        const output = (activations[activations.length - 1] as Float64Array)[0] ?? 0;
        const err = output - (y[i] ?? 0);
        totalLoss += err ** 2;

        const deltas: Float64Array[] = new Array(weights.length);
        deltas[weights.length - 1] = new Float64Array([err]);

        for (let l = weights.length - 2; l >= 0; l--) {
          const nextLayer = weights[l + 1] as LayerWeights;
          const nextDelta = deltas[l + 1] as Float64Array;
          const z = zs[l] as Float64Array;
          const delta = new Float64Array(z.length);
          for (let j = 0; j < z.length; j++) {
            let sum = 0;
            for (let k = 0; k < nextLayer.W.length; k++) {
              sum += ((nextLayer.W[k] ?? new Float64Array(0))[j] ?? 0) * (nextDelta[k] ?? 0);
            }
            delta[j] = sum * activDeriv(z[j] ?? 0);
          }
          deltas[l] = delta;
        }

        for (let l = 0; l < weights.length; l++) {
          const layer = weights[l] as LayerWeights;
          const prevA = activations[l] as Float64Array;
          const delta = deltas[l] as Float64Array;
          for (let j = 0; j < layer.W.length; j++) {
            const wRow = layer.W[j] as Float64Array;
            for (let k = 0; k < prevA.length; k++) {
              wRow[k] =
                (wRow[k] ?? 0) -
                this.learningRate * ((delta[j] ?? 0) * (prevA[k] ?? 0) + this.alpha * (wRow[k] ?? 0));
            }
            layer.b[j] = (layer.b[j] ?? 0) - this.learningRate * (delta[j] ?? 0);
          }
        }
      }
      if (totalLoss / n < this.tol) break;
    }

    this.coefs_ = weights;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.coefs_ === null) throw new NotFittedError("MLPRegressor");
    const [activFn] = getActivation(this.activation);
    return new Float64Array(
      X.map((xi) => {
        const { activations } = this._forward(xi, this.coefs_ as LayerWeights[], activFn);
        return (activations[activations.length - 1] as Float64Array)[0] ?? 0;
      }),
    );
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    const yMean = Array.from(y).reduce((a, b) => a + b, 0) / y.length;
    let ssTot = 0;
    let ssRes = 0;
    for (let i = 0; i < y.length; i++) {
      ssTot += ((y[i] ?? 0) - yMean) ** 2;
      ssRes += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
    }
    return ssTot > 0 ? 1 - ssRes / ssTot : 0;
  }
}

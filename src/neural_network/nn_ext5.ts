/**
 * Neural network extensions: Transformer, LSTM, GRU, ResidualBlock
 * Port of sklearn-compatible neural network components
 */

import { NotFittedError } from "../exceptions.js";

function softmax(x: Float64Array): Float64Array {
  const max = x.reduce((a, b) => Math.max(a, b), -Number.POSITIVE_INFINITY);
  const exps = x.map(v => Math.exp((v ?? 0) - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return Float64Array.from(exps.map(v => v / (sum + 1e-15)));
}

function layerNorm(x: Float64Array, gamma: Float64Array, beta: Float64Array): Float64Array {
  const n = x.length;
  let mean = 0;
  let variance = 0;
  for (let i = 0; i < n; i++) mean += (x[i] ?? 0) / n;
  for (let i = 0; i < n; i++) variance += ((x[i] ?? 0) - mean) ** 2 / n;
  const std = Math.sqrt(variance + 1e-5);
  return Float64Array.from(x.map((v, i) => ((v ?? 0) - mean) / std * (gamma[i] ?? 1) + (beta[i] ?? 0)));
}

export class LayerNorm {
  dim: number;
  eps: number;
  gamma_: Float64Array;
  beta_: Float64Array;

  constructor(opts: { dim?: number; eps?: number } = {}) {
    this.dim = opts.dim ?? 64;
    this.eps = opts.eps ?? 1e-5;
    this.gamma_ = new Float64Array(this.dim).fill(1);
    this.beta_ = new Float64Array(this.dim).fill(0);
  }

  forward(x: Float64Array): Float64Array {
    return layerNorm(x, this.gamma_, this.beta_);
  }
}

export class MultiHeadAttention {
  nHeads: number;
  dModel: number;
  dK: number;
  randomState: number;

  Wq_: Float64Array[][] | null = null;
  Wk_: Float64Array[][] | null = null;
  Wv_: Float64Array[][] | null = null;
  Wo_: Float64Array[] | null = null;

  constructor(opts: { nHeads?: number; dModel?: number; randomState?: number } = {}) {
    this.nHeads = opts.nHeads ?? 4;
    this.dModel = opts.dModel ?? 64;
    this.dK = Math.floor((opts.dModel ?? 64) / (opts.nHeads ?? 4));
    this.randomState = opts.randomState ?? 42;
  }

  initialize(): this {
    let seed = this.randomState;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    const initMat = (rows: number, cols: number) => Array.from({ length: rows }, () => {
      const row = new Float64Array(cols);
      const scale = Math.sqrt(2 / (rows + cols));
      for (let j = 0; j < cols; j++) row[j] = (rng() * 2 - 1) * scale;
      return row;
    });
    this.Wq_ = Array.from({ length: this.nHeads }, () => initMat(this.dModel, this.dK)).flat().reduce<Float64Array[][]>((acc, _, i, arr) => {
      if (i % this.dModel === 0) acc.push(arr.slice(i, i + this.dModel));
      return acc;
    }, []);
    this.Wk_ = Array.from({ length: this.nHeads }, () => initMat(this.dModel, this.dK)).flat().reduce<Float64Array[][]>((acc, _, i, arr) => {
      if (i % this.dModel === 0) acc.push(arr.slice(i, i + this.dModel));
      return acc;
    }, []);
    this.Wv_ = Array.from({ length: this.nHeads }, () => initMat(this.dModel, this.dK)).flat().reduce<Float64Array[][]>((acc, _, i, arr) => {
      if (i % this.dModel === 0) acc.push(arr.slice(i, i + this.dModel));
      return acc;
    }, []);
    this.Wo_ = initMat(this.nHeads * this.dK, this.dModel);
    return this;
  }

  private matVec(W: Float64Array[], x: Float64Array): Float64Array {
    const out = new Float64Array(W.length);
    for (let i = 0; i < W.length; i++) {
      let sum = 0;
      for (let j = 0; j < x.length; j++) sum += (W[i]![j] ?? 0) * (x[j] ?? 0);
      out[i] = sum;
    }
    return out;
  }

  forward(queries: Float64Array[], keys: Float64Array[], values: Float64Array[]): Float64Array[] {
    if (!this.Wq_ || !this.Wk_ || !this.Wv_ || !this.Wo_) {
      this.initialize();
    }
    const seqLen = queries.length;
    const outputs: Float64Array[] = [];
    for (let qi = 0; qi < seqLen; qi++) {
      const headOutputs: Float64Array[] = [];
      for (let h = 0; h < this.nHeads; h++) {
        const q = this.matVec(this.Wq_![h]!, queries[qi]!);
        const scale = Math.sqrt(this.dK);
        const scores = keys.map(k => {
          const kProj = this.matVec(this.Wk_![h]!, k);
          return q.reduce((s, v, j) => s + (v ?? 0) * (kProj[j] ?? 0), 0) / scale;
        });
        const attnWeights = softmax(Float64Array.from(scores));
        const headOut = new Float64Array(this.dK);
        for (let vi = 0; vi < values.length; vi++) {
          const vProj = this.matVec(this.Wv_![h]!, values[vi]!);
          for (let d = 0; d < this.dK; d++) headOut[d] = (headOut[d] ?? 0) + (attnWeights[vi] ?? 0) * (vProj[d] ?? 0);
        }
        headOutputs.push(headOut);
      }
      const concat = new Float64Array(this.nHeads * this.dK);
      for (let h = 0; h < this.nHeads; h++) for (let d = 0; d < this.dK; d++) concat[h * this.dK + d] = headOutputs[h]![d] ?? 0;
      outputs.push(this.matVec(this.Wo_!, concat));
    }
    return outputs;
  }
}

export class LSTMCell {
  inputSize: number;
  hiddenSize: number;

  Wf_: Float64Array[] | null = null;
  Wi_: Float64Array[] | null = null;
  Wc_: Float64Array[] | null = null;
  Wo_: Float64Array[] | null = null;

  constructor(opts: { inputSize?: number; hiddenSize?: number } = {}) {
    this.inputSize = opts.inputSize ?? 32;
    this.hiddenSize = opts.hiddenSize ?? 64;
  }

  initialize(): this {
    let seed = 42;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    const inputDim = this.inputSize + this.hiddenSize;
    const scale = Math.sqrt(2 / inputDim);
    const initW = () => Array.from({ length: this.hiddenSize }, () => {
      const row = new Float64Array(inputDim + 1);
      for (let j = 0; j < inputDim; j++) row[j] = (rng() * 2 - 1) * scale;
      return row;
    });
    this.Wf_ = initW();
    this.Wi_ = initW();
    this.Wc_ = initW();
    this.Wo_ = initW();
    return this;
  }

  private gateLinear(W: Float64Array[], x: Float64Array, h: Float64Array): Float64Array {
    const combined = new Float64Array(x.length + h.length + 1);
    for (let j = 0; j < x.length; j++) combined[j] = x[j] ?? 0;
    for (let j = 0; j < h.length; j++) combined[x.length + j] = h[j] ?? 0;
    combined[x.length + h.length] = 1.0;
    const out = new Float64Array(W.length);
    for (let i = 0; i < W.length; i++) {
      let s = 0;
      for (let j = 0; j < combined.length; j++) s += (W[i]![j] ?? 0) * (combined[j] ?? 0);
      out[i] = s;
    }
    return out;
  }

  forward(x: Float64Array, h: Float64Array, c: Float64Array): { h: Float64Array; c: Float64Array } {
    if (!this.Wf_) this.initialize();
    const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));
    const f = this.gateLinear(this.Wf_!, x, h).map(v => sigmoid(v));
    const i = this.gateLinear(this.Wi_!, x, h).map(v => sigmoid(v));
    const cHat = this.gateLinear(this.Wc_!, x, h).map(v => Math.tanh(v));
    const o = this.gateLinear(this.Wo_!, x, h).map(v => sigmoid(v));
    const newC = Float64Array.from(c.map((v, j) => (f[j] ?? 0) * (v ?? 0) + (i[j] ?? 0) * (cHat[j] ?? 0)));
    const newH = Float64Array.from(newC.map((v, j) => (o[j] ?? 0) * Math.tanh(v)));
    return { h: newH, c: newC };
  }
}

export class GRUCell {
  inputSize: number;
  hiddenSize: number;

  Wz_: Float64Array[] | null = null;
  Wr_: Float64Array[] | null = null;
  Wh_: Float64Array[] | null = null;

  constructor(opts: { inputSize?: number; hiddenSize?: number } = {}) {
    this.inputSize = opts.inputSize ?? 32;
    this.hiddenSize = opts.hiddenSize ?? 64;
  }

  initialize(): this {
    let seed = 0;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    const inputDim = this.inputSize + this.hiddenSize;
    const scale = Math.sqrt(2 / inputDim);
    const initW = () => Array.from({ length: this.hiddenSize }, () => {
      const row = new Float64Array(inputDim);
      for (let j = 0; j < inputDim; j++) row[j] = (rng() * 2 - 1) * scale;
      return row;
    });
    this.Wz_ = initW();
    this.Wr_ = initW();
    this.Wh_ = initW();
    return this;
  }

  forward(x: Float64Array, h: Float64Array): Float64Array {
    if (!this.Wz_) this.initialize();
    const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));
    const combined = new Float64Array(x.length + h.length);
    for (let j = 0; j < x.length; j++) combined[j] = x[j] ?? 0;
    for (let j = 0; j < h.length; j++) combined[x.length + j] = h[j] ?? 0;
    const matVec = (W: Float64Array[], inp: Float64Array) => Float64Array.from(W.map(row => inp.reduce((s, v, j) => s + (row[j] ?? 0) * (v ?? 0), 0)));
    const z = matVec(this.Wz_!, combined).map(v => sigmoid(v));
    const r = matVec(this.Wr_!, combined).map(v => sigmoid(v));
    const rh = Float64Array.from(h.map((v, j) => (r[j] ?? 0) * (v ?? 0)));
    const combined2 = new Float64Array(x.length + rh.length);
    for (let j = 0; j < x.length; j++) combined2[j] = x[j] ?? 0;
    for (let j = 0; j < rh.length; j++) combined2[x.length + j] = rh[j] ?? 0;
    const hHat = matVec(this.Wh_!, combined2).map(v => Math.tanh(v));
    return Float64Array.from(h.map((v, j) => (1 - (z[j] ?? 0)) * (v ?? 0) + (z[j] ?? 0) * (hHat[j] ?? 0)));
  }
}

export class TransformerEncoder {
  dModel: number;
  nHeads: number;
  dFF: number;
  nLayers: number;

  private attention_: MultiHeadAttention[] | null = null;
  private norm1_: LayerNorm[] | null = null;
  private norm2_: LayerNorm[] | null = null;
  private ff1_: Float64Array[][] | null = null;
  private ff2_: Float64Array[][] | null = null;

  constructor(opts: { dModel?: number; nHeads?: number; dFF?: number; nLayers?: number } = {}) {
    this.dModel = opts.dModel ?? 64;
    this.nHeads = opts.nHeads ?? 4;
    this.dFF = opts.dFF ?? 256;
    this.nLayers = opts.nLayers ?? 2;
  }

  initialize(): this {
    this.attention_ = Array.from({ length: this.nLayers }, (_, l) => new MultiHeadAttention({ nHeads: this.nHeads, dModel: this.dModel, randomState: l }).initialize());
    this.norm1_ = Array.from({ length: this.nLayers }, () => new LayerNorm({ dim: this.dModel }));
    this.norm2_ = Array.from({ length: this.nLayers }, () => new LayerNorm({ dim: this.dModel }));
    let seed = 0;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    const scale1 = Math.sqrt(2 / this.dModel);
    const scale2 = Math.sqrt(2 / this.dFF);
    this.ff1_ = Array.from({ length: this.nLayers }, () => Array.from({ length: this.dFF }, () => { const r = new Float64Array(this.dModel); for (let j = 0; j < this.dModel; j++) r[j] = (rng() * 2 - 1) * scale1; return r; }));
    this.ff2_ = Array.from({ length: this.nLayers }, () => Array.from({ length: this.dModel }, () => { const r = new Float64Array(this.dFF); for (let j = 0; j < this.dFF; j++) r[j] = (rng() * 2 - 1) * scale2; return r; }));
    return this;
  }

  forward(X: Float64Array[]): Float64Array[] {
    if (!this.attention_) this.initialize();
    let out = X.map(x => x.slice());
    for (let l = 0; l < this.nLayers; l++) {
      const attended = this.attention_![l]!.forward(out, out, out);
      out = out.map((x, i) => this.norm1_![l]!.forward(Float64Array.from(x.map((v, j) => (v ?? 0) + (attended[i]![j] ?? 0))))) as Float64Array<ArrayBuffer>[];
      out = out.map(x => {
        const ff1Out = this.ff1_![l]!.map(row => Math.max(0, x.reduce((s, v, j) => s + (row[j] ?? 0) * (v ?? 0), 0)));
        const ff2Out = new Float64Array(this.dModel);
        for (let j = 0; j < this.dModel; j++) {
          for (let k = 0; k < this.dFF; k++) ff2Out[j] = (ff2Out[j] ?? 0) + (this.ff2_![l]![j]![k] ?? 0) * (ff1Out[k] ?? 0);
        }
        return this.norm2_![l]!.forward(Float64Array.from(x.map((v, j) => (v ?? 0) + (ff2Out[j] ?? 0))));
      });
    }
    return out;
  }
}

export class ResidualMLP {
  layers: number[];
  randomState: number;

  private weights_: Float64Array[][] | null = null;
  private biases_: Float64Array[] | null = null;
  classes_: Int32Array | null = null;

  constructor(opts: { layers?: number[]; randomState?: number } = {}) {
    this.layers = opts.layers ?? [64, 64, 64];
    this.randomState = opts.randomState ?? 42;
  }

  initialize(inputDim: number, outputDim: number): this {
    let seed = this.randomState;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    const dims = [inputDim, ...this.layers, outputDim];
    this.weights_ = [];
    this.biases_ = [];
    for (let l = 0; l < dims.length - 1; l++) {
      const inDim = dims[l]!;
      const outDim = dims[l + 1]!;
      const scale = Math.sqrt(2 / inDim);
      this.weights_.push(Array.from({ length: outDim }, () => { const r = new Float64Array(inDim); for (let j = 0; j < inDim; j++) r[j] = (rng() * 2 - 1) * scale; return r; }));
      this.biases_.push(new Float64Array(outDim));
    }
    return this;
  }

  forward(x: Float64Array): Float64Array {
    if (!this.weights_) throw new NotFittedError("ResidualMLP not initialized.");
    let current = x.slice();
    for (let l = 0; l < this.weights_.length; l++) {
      const W = this.weights_[l]!;
      const b = this.biases_![l]!;
      const next = new Float64Array(W.length);
      for (let i = 0; i < W.length; i++) {
        let s = b[i] ?? 0;
        for (let j = 0; j < current.length; j++) s += (W[i]![j] ?? 0) * (current[j] ?? 0);
        next[i] = s;
      }
      if (l < this.weights_.length - 1) {
        for (let i = 0; i < next.length; i++) next[i] = Math.max(0, next[i] ?? 0);
        if (current.length === next.length) {
          for (let i = 0; i < next.length; i++) next[i] = (next[i] ?? 0) + (current[i] ?? 0);
        }
      }
      current = next;
    }
    return current;
  }
}

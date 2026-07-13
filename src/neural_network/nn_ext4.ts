/**
 * Echo State Network (Reservoir Computing) — neural network port.
 */

export class EchoStateNetwork {
  reservoirSize: number;
  spectralRadius: number;
  inputScaling: number;
  leakingRate: number;
  sparsity: number;
  alpha: number;
  randomState: number;
  private W_in: Float64Array[] | null = null;
  private W_res: Float64Array[] | null = null;
  private W_out: Float64Array[] | null = null;
  coef_: Float64Array[] | null = null;

  constructor(
    reservoirSize = 100,
    spectralRadius = 0.95,
    inputScaling = 1.0,
    leakingRate = 0.8,
    sparsity = 0.1,
    alpha = 1e-6,
    randomState = 42,
  ) {
    this.reservoirSize = reservoirSize;
    this.spectralRadius = spectralRadius;
    this.inputScaling = inputScaling;
    this.leakingRate = leakingRate;
    this.sparsity = sparsity;
    this.alpha = alpha;
    this.randomState = randomState;
  }

  private _initWeights(nInput: number): void {
    const { reservoirSize } = this;
    // Seeded pseudo-random using linear congruential generator
    let seed = this.randomState;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return (seed >>> 0) / 4294967296;
    };

    this.W_in = Array.from({ length: reservoirSize }, () => {
      const row = new Float64Array(nInput);
      for (let j = 0; j < nInput; j++) row[j] = (rand() - 0.5) * 2 * this.inputScaling;
      return row;
    });

    // Sparse random reservoir
    const W_raw: Float64Array[] = Array.from({ length: reservoirSize }, () => {
      const row = new Float64Array(reservoirSize);
      for (let j = 0; j < reservoirSize; j++) {
        if (rand() < this.sparsity) row[j] = (rand() - 0.5) * 2;
      }
      return row;
    });

    // Scale to desired spectral radius
    const maxEig = _powerIteration(W_raw, reservoirSize, 20, rand);
    this.W_res = W_raw.map((row) => row.map((v) => v * this.spectralRadius / (maxEig || 1)));
  }

  private _runReservoir(X_seq: Float64Array[]): Float64Array[] {
    const n = X_seq.length;
    const { reservoirSize, leakingRate } = this;
    const W_in = this.W_in as Float64Array[];
    const W_res = this.W_res as Float64Array[];
    let state = new Float64Array(reservoirSize);
    const states: Float64Array[] = [];

    for (let t = 0; t < n; t++) {
      const x = X_seq[t] as Float64Array;
      const pre = new Float64Array(reservoirSize);
      for (let i = 0; i < reservoirSize; i++) {
        let s = 0;
        for (let j = 0; j < x.length; j++) s += (W_in[i]?.[j] ?? 0) * (x[j] ?? 0);
        for (let j = 0; j < reservoirSize; j++) s += (W_res[i]?.[j] ?? 0) * (state[j] ?? 0);
        pre[i] = Math.tanh(s);
      }
      const newState = new Float64Array(reservoirSize);
      for (let i = 0; i < reservoirSize; i++) {
        newState[i] = (1 - leakingRate) * (state[i] ?? 0) + leakingRate * (pre[i] ?? 0);
      }
      state = newState;
      states.push(new Float64Array(state));
    }
    return states;
  }

  fit(X: Float64Array[][], y: Float64Array): this {
    const nSamples = X.length;
    const nInput = X[0]?.[0]?.length ?? 0;
    this._initWeights(nInput);

    // Collect reservoir states for all sequences (use last state)
    const extended: Float64Array[] = [];
    for (let s = 0; s < nSamples; s++) {
      const states = this._runReservoir(X[s] as Float64Array[]);
      const lastState = states[states.length - 1] ?? new Float64Array(this.reservoirSize);
      // Augment with input
      const inputLast = (X[s] as Float64Array[])[((X[s] as Float64Array[]).length - 1)] ?? new Float64Array(nInput);
      const ext = new Float64Array(this.reservoirSize + nInput + 1);
      ext.set(lastState);
      for (let j = 0; j < nInput; j++) ext[this.reservoirSize + j] = inputLast[j] ?? 0;
      ext[this.reservoirSize + nInput] = 1;
      extended.push(ext);
    }

    // Ridge regression
    const cols = extended[0]?.length ?? 0;
    const XtX: Float64Array[] = Array.from({ length: cols }, () => new Float64Array(cols));
    const Xty = new Float64Array(cols);
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < cols; j++) {
        Xty[j]! += (extended[i]?.[j] ?? 0) * (y[i] ?? 0);
        for (let k = 0; k < cols; k++) {
          (XtX[j]! as Float64Array)[k]! += (extended[i]?.[j] ?? 0) * (extended[i]?.[k] ?? 0);
        }
      }
    }
    for (let j = 0; j < cols - 1; j++) (XtX[j]! as Float64Array)[j]! += this.alpha;

    const w = solveESN(XtX, Xty, cols);
    this.W_out = [w];
    this.coef_ = [w.slice(0, this.reservoirSize)];
    return this;
  }

  predict(X: Float64Array[][]): Float64Array {
    if (!this.W_out) throw new Error("Not fitted");
    const w = this.W_out[0] as Float64Array;
    const nInput = X[0]?.[0]?.length ?? 0;
    const nSamples = X.length;
    const result = new Float64Array(nSamples);
    for (let s = 0; s < nSamples; s++) {
      const states = this._runReservoir(X[s] as Float64Array[]);
      const lastState = states[states.length - 1] ?? new Float64Array(this.reservoirSize);
      const inputLast = (X[s] as Float64Array[])[((X[s] as Float64Array[]).length - 1)] ?? new Float64Array(nInput);
      let v = w[this.reservoirSize + nInput] ?? 0;
      for (let j = 0; j < this.reservoirSize; j++) v += (lastState[j] ?? 0) * (w[j] ?? 0);
      for (let j = 0; j < nInput; j++) v += (inputLast[j] ?? 0) * (w[this.reservoirSize + j] ?? 0);
      result[s] = v;
    }
    return result;
  }
}

function _powerIteration(A: Float64Array[], n: number, nIter: number, rand: () => number): number {
  let v = Float64Array.from({ length: n }, () => rand() - 0.5);
  let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  for (let i = 0; i < n; i++) v[i]! /= norm;
  let lambda = 1;
  for (let iter = 0; iter < nIter; iter++) {
    const Av = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) Av[i]! += (A[i]?.[j] ?? 0) * (v[j] ?? 0);
    }
    norm = Math.sqrt(Av.reduce((s, x) => s + x * x, 0)) || 1;
    lambda = norm;
    for (let i = 0; i < n; i++) v[i] = (Av[i] ?? 0) / norm;
  }
  return lambda;
}

function solveESN(A: Float64Array[], b: Float64Array, n: number): Float64Array {
  const M = A.map((r) => new Float64Array(r));
  const rhs = new Float64Array(b);
  for (let c = 0; c < n; c++) {
    let maxR = c;
    for (let r = c + 1; r < n; r++) {
      if (Math.abs((M[r] as Float64Array)[c] ?? 0) > Math.abs((M[maxR] as Float64Array)[c] ?? 0)) maxR = r;
    }
    const t = M[c]; M[c] = M[maxR] as Float64Array; M[maxR] = t as Float64Array as Float64Array<ArrayBuffer> as Float64Array<ArrayBuffer>;
    const tr = rhs[c] ?? 0; rhs[c] = rhs[maxR] ?? 0; rhs[maxR] = tr;
    const piv = (M[c] as Float64Array)[c] ?? 1e-12;
    for (let r = c + 1; r < n; r++) {
      const f = ((M[r] as Float64Array)[c] ?? 0) / piv;
      for (let k = c; k < n; k++) (M[r] as Float64Array)[k] = ((M[r] as Float64Array)[k] ?? 0) - f * ((M[c] as Float64Array)[k] ?? 0);
      rhs[r] = (rhs[r] ?? 0) - f * (rhs[c] ?? 0);
    }
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = rhs[i] ?? 0;
    for (let j = i + 1; j < n; j++) s -= ((M[i] as Float64Array)[j] ?? 0) * (x[j] ?? 0);
    x[i] = s / ((M[i] as Float64Array)[i] ?? 1e-12);
  }
  return x;
}

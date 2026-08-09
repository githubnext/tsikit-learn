/**
 * Kernel approximation methods: RBFSampler, Nystroem, AdditiveChi2Sampler, SkewedChi2Sampler.
 * Mirrors sklearn.kernel_approximation.
 */

import { NotFittedError } from "../exceptions.js";

/**
 * Approximates feature map of an RBF kernel by Monte Carlo approximation.
 * Mirrors sklearn.kernel_approximation.RBFSampler.
 */
export class RBFSampler {
  gamma: number;
  nComponents: number;
  randomState: number;

  randomWeights_: Float64Array[] | null = null;
  randomOffset_: Float64Array | null = null;

  constructor(
    options: {
      gamma?: number;
      nComponents?: number;
      randomState?: number;
    } = {},
  ) {
    this.gamma = options.gamma ?? 1.0;
    this.nComponents = options.nComponents ?? 100;
    this.randomState = options.randomState ?? 42;
  }

  private _rng(): () => number {
    let s = this.randomState;
    return () => {
      s = (s * 1664525 + 1013904223) & 0xffffffff;
      return (s >>> 0) / 0x100000000;
    };
  }

  private _randn(rng: () => number): number {
    const u = rng();
    const v = rng();
    return Math.sqrt(-2 * Math.log(u + 1e-15)) * Math.cos(2 * Math.PI * v);
  }

  fit(X: Float64Array[]): this {
    const p = (X[0] ?? new Float64Array(0)).length;
    const rng = this._rng();
    const scale = Math.sqrt(2 * this.gamma);
    this.randomWeights_ = Array.from({ length: this.nComponents }, () => {
      const w = new Float64Array(p);
      for (let j = 0; j < p; j++) w[j] = this._randn(rng) * scale;
      return w;
    });
    this.randomOffset_ = new Float64Array(this.nComponents);
    for (let i = 0; i < this.nComponents; i++) {
      this.randomOffset_[i] = rng() * 2 * Math.PI;
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.randomWeights_ === null || this.randomOffset_ === null) {
      throw new NotFittedError();
    }
    const scale = Math.sqrt(2 / this.nComponents);
    return X.map((xi) => {
      const out = new Float64Array(this.nComponents);
      for (let i = 0; i < this.nComponents; i++) {
        const w = this.randomWeights_![i] ?? new Float64Array(0);
        let dot = 0;
        for (let j = 0; j < xi.length; j++) dot += (xi[j] ?? 0) * (w[j] ?? 0);
        out[i] = scale * Math.cos(dot + (this.randomOffset_![i] ?? 0));
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

/**
 * Approximate a kernel map using a subset of the training data (Nystroem method).
 * Mirrors sklearn.kernel_approximation.Nystroem.
 */
export class Nystroem {
  kernel: "rbf" | "polynomial" | "linear";
  gamma: number;
  coef0: number;
  degree: number;
  nComponents: number;
  randomState: number;

  components_: Float64Array[] | null = null;
  normalizationMatrix_: Float64Array[] | null = null;

  constructor(
    options: {
      kernel?: "rbf" | "polynomial" | "linear";
      gamma?: number;
      coef0?: number;
      degree?: number;
      nComponents?: number;
      randomState?: number;
    } = {},
  ) {
    this.kernel = options.kernel ?? "rbf";
    this.gamma = options.gamma ?? 1.0;
    this.coef0 = options.coef0 ?? 1.0;
    this.degree = options.degree ?? 3;
    this.nComponents = options.nComponents ?? 100;
    this.randomState = options.randomState ?? 42;
  }

  private _kernelFunc(a: Float64Array, b: Float64Array): number {
    const p = a.length;
    if (this.kernel === "rbf") {
      let dist = 0;
      for (let j = 0; j < p; j++) dist += ((a[j] ?? 0) - (b[j] ?? 0)) ** 2;
      return Math.exp(-this.gamma * dist);
    }
    if (this.kernel === "polynomial") {
      let dot = 0;
      for (let j = 0; j < p; j++) dot += (a[j] ?? 0) * (b[j] ?? 0);
      return (this.gamma * dot + this.coef0) ** this.degree;
    }
    let dot = 0;
    for (let j = 0; j < p; j++) dot += (a[j] ?? 0) * (b[j] ?? 0);
    return dot;
  }

  private _choleskyInverse(K: Float64Array[]): Float64Array[] {
    const n = K.length;
    const L = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = K[i]![j] ?? 0;
        for (let k = 0; k < j; k++) s -= (L[i]![k] ?? 0) * (L[j]![k] ?? 0);
        if (i === j) {
          L[i]![j] = Math.sqrt(Math.max(s, 1e-12));
        } else {
          L[i]![j] = s / ((L[j]![j] ?? 1e-12) || 1e-12);
        }
      }
    }
    // Invert L
    const Linv = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      Linv[i]![i] = 1 / ((L[i]![i] ?? 1e-12) || 1e-12);
      for (let j = i - 1; j >= 0; j--) {
        let s = 0;
        for (let k = j + 1; k <= i; k++)
          s += (L[i]![k] ?? 0) * (Linv[k]![j] ?? 0);
        Linv[i]![j] = -s / ((L[i]![i] ?? 1e-12) || 1e-12);
      }
    }
    // K^{-1} = (L^T L)^{-1} = Linv^T Linv
    const out = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let k = 0; k < n; k++)
          s += (Linv[k]![i] ?? 0) * (Linv[k]![j] ?? 0);
        out[i]![j] = s;
      }
    }
    return out;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const m = Math.min(this.nComponents, n);
    // Random subsample
    let seed = this.randomState;
    const indices: number[] = [];
    const used = new Set<number>();
    for (let i = 0; i < m; i++) {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      let idx = (seed >>> 0) % n;
      let tries = 0;
      while (used.has(idx) && tries < n) {
        idx = (idx + 1) % n;
        tries++;
      }
      used.add(idx);
      indices.push(idx);
    }
    this.components_ = indices.map((i) => X[i] ?? new Float64Array(0));
    // Compute kernel matrix K_mm
    const Kmm = Array.from({ length: m }, () => new Float64Array(m));
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        Kmm[i]![j] = this._kernelFunc(
          this.components_![i] ?? new Float64Array(0),
          this.components_![j] ?? new Float64Array(0),
        );
      }
    }
    this.normalizationMatrix_ = this._choleskyInverse(Kmm);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.components_ === null || this.normalizationMatrix_ === null) {
      throw new NotFittedError();
    }
    const m = this.components_.length;
    return X.map((xi) => {
      const kv = new Float64Array(m);
      for (let j = 0; j < m; j++) {
        kv[j] = this._kernelFunc(
          xi,
          this.components_![j] ?? new Float64Array(0),
        );
      }
      // out = kv @ normalizationMatrix_
      const out = new Float64Array(m);
      for (let j = 0; j < m; j++) {
        let s = 0;
        for (let k = 0; k < m; k++)
          s += (kv[k] ?? 0) * (this.normalizationMatrix_![k]![j] ?? 0);
        out[j] = s;
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

/**
 * Approximate feature map for additive chi2 kernel.
 * Mirrors sklearn.kernel_approximation.AdditiveChi2Sampler.
 */
export class AdditiveChi2Sampler {
  sampleSteps: number;
  sampleInterval: number | null;

  sampleInterval_: number | null = null;

  constructor(
    options: { sampleSteps?: number; sampleInterval?: number | null } = {},
  ) {
    this.sampleSteps = options.sampleSteps ?? 2;
    this.sampleInterval = options.sampleInterval ?? null;
  }

  fit(X: Float64Array[]): this {
    this.sampleInterval_ = this.sampleInterval ?? 0.4;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.sampleInterval_ === null) throw new NotFittedError();
    const p = (X[0] ?? new Float64Array(0)).length;
    const steps = this.sampleSteps;
    const interval = this.sampleInterval_;
    const outDim = p * (2 * steps + 1);
    return X.map((xi) => {
      const out = new Float64Array(outDim);
      for (let j = 0; j < p; j++) {
        const x = xi[j] ?? 0;
        const sqrtX = Math.sqrt(x + 1e-12);
        out[j] = sqrtX;
        for (let s = 1; s <= steps; s++) {
          const c = Math.sqrt(2 * Math.exp(-Math.PI * s * interval));
          const cos = c * sqrtX * Math.cos(s * Math.log(x + 1e-12));
          const sin = c * sqrtX * Math.sin(s * Math.log(x + 1e-12));
          out[j + p * (2 * s - 1)] = cos;
          out[j + p * (2 * s)] = sin;
        }
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

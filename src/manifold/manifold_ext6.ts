/**
 * Manifold learning extensions: DiffusionMaps, ForceDirectedLayout, SphericalMDS
 * Port of sklearn.manifold extensions
 */

import { NotFittedError } from "../exceptions.js";

function pairwiseDists(X: Float64Array[]): Float64Array[] {
  const n = X.length;
  return Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      let d = 0;
      for (let k = 0; k < (X[i]?.length ?? 0); k++) d += ((X[i]![k] ?? 0) - (X[j]![k] ?? 0)) ** 2;
      row[j] = Math.sqrt(d);
    }
    return row;
  });
}

export class DiffusionMaps {
  nComponents: number;
  alpha: number;
  epsilon: number | "auto";
  nDiffusionSteps: number;
  randomState: number;

  embedding_: Float64Array[] | null = null;
  lambdas_: Float64Array | null = null;

  constructor(opts: {
    nComponents?: number;
    alpha?: number;
    epsilon?: number | "auto";
    nDiffusionSteps?: number;
    randomState?: number;
  } = {}) {
    this.nComponents = opts.nComponents ?? 2;
    this.alpha = opts.alpha ?? 0.5;
    this.epsilon = opts.epsilon ?? "auto";
    this.nDiffusionSteps = opts.nDiffusionSteps ?? 1;
    this.randomState = opts.randomState ?? 42;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const dists = pairwiseDists(X);
    const eps = this.epsilon === "auto"
      ? dists.reduce((s, row) => s + row.reduce((a, b) => a + b, 0), 0) / (n * n)
      : this.epsilon;
    const K = dists.map(row => Float64Array.from(row.map(d => Math.exp(-((d ?? 0) ** 2) / (2 * eps + 1e-15)))));
    const q = K.map(row => {
      const sum = row.reduce((a, b) => a + b, 0);
      return Math.pow(sum, this.alpha);
    });
    const Khat = K.map((row, i) => Float64Array.from(row.map((v, j) => (v ?? 0) / ((q[i] ?? 1) * (q[j] ?? 1) + 1e-15))));
    const d = Khat.map(row => row.reduce((a, b) => a + b, 0));
    const P = Khat.map((row, i) => Float64Array.from(row.map((v, j) => (v ?? 0) / ((d[i] ?? 1) * 1 + 1e-15))));
    let seed = this.randomState;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    const k = Math.min(this.nComponents + 1, n);
    const vectors: Float64Array[] = Array.from({ length: k }, () => {
      const v = new Float64Array(n);
      for (let i = 0; i < n; i++) v[i] = rng() * 2 - 1;
      return v;
    });
    const lambdas = new Float64Array(k);
    for (let step = 0; step < this.nDiffusionSteps; step++) {
      for (let kk = 0; kk < k; kk++) {
        let Pv = new Float64Array(n);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Pv[i] = (Pv[i] ?? 0) + (P[i]![j] ?? 0) * (vectors[kk]![j] ?? 0);
        for (let prev = 0; prev < kk; prev++) {
          let dot = 0;
          for (let i = 0; i < n; i++) dot += (Pv[i] ?? 0) * (vectors[prev]![i] ?? 0);
          for (let i = 0; i < n; i++) Pv[i] = (Pv[i] ?? 0) - dot * (vectors[prev]![i] ?? 0);
        }
        let norm = 0;
        for (let i = 0; i < n; i++) norm += (Pv[i] ?? 0) ** 2;
        norm = Math.sqrt(norm) + 1e-15;
        lambdas[kk] = norm;
        for (let i = 0; i < n; i++) vectors[kk]![i] = (Pv[i] ?? 0) / norm;
      }
      void step;
    }
    this.lambdas_ = lambdas.slice(1);
    this.embedding_ = Array.from({ length: n }, (_, i) =>
      Float64Array.from({ length: this.nComponents }, (__, kk) => (vectors[kk + 1]![i] ?? 0) * Math.pow(lambdas[kk + 1] ?? 1, this.nDiffusionSteps))
    );
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.embedding_) throw new NotFittedError("DiffusionMaps not fitted.");
    return this.embedding_;
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class ForceDirectedLayout {
  nComponents: number;
  nIter: number;
  repulsionStrength: number;
  attractionStrength: number;
  randomState: number;

  embedding_: Float64Array[] | null = null;

  constructor(opts: {
    nComponents?: number;
    nIter?: number;
    repulsionStrength?: number;
    attractionStrength?: number;
    randomState?: number;
  } = {}) {
    this.nComponents = opts.nComponents ?? 2;
    this.nIter = opts.nIter ?? 100;
    this.repulsionStrength = opts.repulsionStrength ?? 1.0;
    this.attractionStrength = opts.attractionStrength ?? 0.01;
    this.randomState = opts.randomState ?? 42;
  }

  fit(adjacency: Float64Array[]): this {
    const n = adjacency.length;
    const k = this.nComponents;
    let seed = this.randomState;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    let pos = Array.from({ length: n }, () => Float64Array.from({ length: k }, () => rng() * 2 - 1));
    for (let iter = 0; iter < this.nIter; iter++) {
      const t = 1 - iter / this.nIter;
      const forces = Array.from({ length: n }, () => new Float64Array(k));
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (i === j) continue;
          const diff = Float64Array.from({ length: k }, (_, d) => (pos[i]![d] ?? 0) - (pos[j]![d] ?? 0));
          const dist2 = diff.reduce((s, v) => s + (v ?? 0) ** 2, 0) + 1e-15;
          const dist = Math.sqrt(dist2);
          const rep = this.repulsionStrength / dist2;
          for (let d = 0; d < k; d++) forces[i]![d] = (forces[i]![d] ?? 0) + rep * (diff[d] ?? 0) / dist;
          if ((adjacency[i]![j] ?? 0) > 0) {
            const attr = this.attractionStrength * dist;
            for (let d = 0; d < k; d++) forces[i]![d] = (forces[i]![d] ?? 0) - attr * (diff[d] ?? 0) / dist;
          }
        }
      }
      pos = pos.map((pi, i) => Float64Array.from({ length: k }, (_, d) => (pi[d] ?? 0) + t * (forces[i]![d] ?? 0)));
      void iter;
    }
    this.embedding_ = pos;
    return this;
  }

  transform(): Float64Array[] {
    if (!this.embedding_) throw new NotFittedError("ForceDirectedLayout not fitted.");
    return this.embedding_;
  }

  fitTransform(adjacency: Float64Array[]): Float64Array[] {
    return this.fit(adjacency).transform();
  }
}

export class SphericalMDS {
  nComponents: number;
  nIter: number;
  randomState: number;

  embedding_: Float64Array[] | null = null;
  stress_: number | null = null;

  constructor(opts: { nComponents?: number; nIter?: number; randomState?: number } = {}) {
    this.nComponents = opts.nComponents ?? 2;
    this.nIter = opts.nIter ?? 300;
    this.randomState = opts.randomState ?? 0;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const k = this.nComponents;
    const dists = pairwiseDists(X);
    let seed = this.randomState;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    let Z = Array.from({ length: n }, () => {
      const v = Float64Array.from({ length: k }, () => rng() * 2 - 1);
      let norm = 0;
      for (let d = 0; d < k; d++) norm += (v[d] ?? 0) ** 2;
      norm = Math.sqrt(norm) + 1e-15;
      for (let d = 0; d < k; d++) v[d] = (v[d] ?? 0) / norm;
      return v;
    });
    let stress = 0;
    for (let iter = 0; iter < this.nIter; iter++) {
      const B = Array.from({ length: n }, () => new Float64Array(n));
      stress = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        if (i === j) continue;
        let dotProd = 0;
        for (let d = 0; d < k; d++) dotProd += (Z[i]![d] ?? 0) * (Z[j]![d] ?? 0);
        const arcDist = Math.acos(Math.max(-1, Math.min(1, dotProd)));
        const targetDist = dists[i]![j] ?? 0;
        stress += (arcDist - targetDist) ** 2;
        const w = arcDist > 1e-10 ? targetDist / (arcDist + 1e-15) : 0;
        B[i]![j] = -w;
        B[i]![i] = (B[i]![i] ?? 0) + w;
      }
      const newZ = Array.from({ length: n }, (_, i) => {
        const zi = new Float64Array(k);
        for (let j = 0; j < n; j++) {
          const bij = B[i]![j] ?? 0;
          for (let d = 0; d < k; d++) zi[d] = (zi[d] ?? 0) + bij * (Z[j]![d] ?? 0);
        }
        for (let d = 0; d < k; d++) zi[d] = (zi[d] ?? 0) / n;
        let norm = 0;
        for (let d = 0; d < k; d++) norm += (zi[d] ?? 0) ** 2;
        norm = Math.sqrt(norm) + 1e-15;
        for (let d = 0; d < k; d++) zi[d] = (zi[d] ?? 0) / norm;
        return zi;
      });
      Z = newZ;
      void iter;
    }
    this.embedding_ = Z;
    this.stress_ = stress;
    return this;
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).embedding_!;
  }
}

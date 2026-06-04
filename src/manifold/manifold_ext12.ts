/**
 * Manifold learning extensions: Force-directed layout, Laplacian Eigenmaps, Trimap
 */

export class LaplacianEigenmapsExt {
  private embedding_: Float64Array[] = [];
  private fitted_ = false;

  constructor(
    private nComponents: number = 2,
    private nNeighbors: number = 5,
    private gamma: number | 'auto' = 'auto'
  ) {}

  fitTransform(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    // Build adjacency with KNN
    const W = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      const dists = X.map((xj, j) => ({ j, d: X[i]!.reduce((s, v, k) => s + (v - (xj[k] ?? 0)) ** 2, 0) }));
      dists.sort((a, b) => a.d - b.d);
      const sigma = dists[this.nNeighbors]?.d ?? 1;
      const g = this.gamma === 'auto' ? 1 / (2 * sigma + 1e-10) : this.gamma;
      for (const { j, d } of dists.slice(1, this.nNeighbors + 1)) {
        W[i]![j] = Math.exp(-g * d);
        W[j]![i] = Math.exp(-g * d);
      }
    }
    // Degree matrix and Laplacian
    const D = new Float64Array(n).map((_, i) => W[i]!.reduce((s, v) => s + v, 0));
    const L = Array.from({ length: n }, (_, i) => new Float64Array(n).map((_, j) => (i === j ? D[i]! : 0) - (W[i]?.[j] ?? 0)));

    // Generalized eigendecomposition L v = lambda D v => D^{-1/2} L D^{-1/2} v = lambda v
    const DinvSqrt = D.map(d => 1 / Math.sqrt(d + 1e-10));
    const Lsym = Array.from({ length: n }, (_, i) => new Float64Array(n).map((_, j) => (DinvSqrt[i] ?? 0) * (L[i]?.[j] ?? 0) * (DinvSqrt[j] ?? 0)));

    const { vecs } = this._eigenDecomp(Lsym, n, this.nComponents + 1);
    // Skip first eigenvector (constant), use next nComponents
    const embedding = Array.from({ length: n }, (_, i) => new Float64Array(this.nComponents).map((_, c) => (vecs[c + 1]?.[i] ?? 0) * (DinvSqrt[i] ?? 0)));
    this.embedding_ = embedding;
    this.fitted_ = true;
    return embedding;
  }

  private _eigenDecomp(A: Float64Array[], n: number, k: number): { vecs: Float64Array[]; vals: number[] } {
    let rng = 42;
    const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return (rng / 0xffffffff) * 2 - 1; };
    const vecs: Float64Array[] = [], vals: number[] = [];
    let Ak = A.map(r => r.slice());
    for (let s = 0; s < k; s++) {
      let v = new Float64Array(n).map(() => rand());
      let lambda = 0;
      for (let iter = 0; iter < 100; iter++) {
        let Av = new Float64Array(n);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Av[i] = (Av[i] ?? 0) + (Ak[i]?.[j] ?? 0) * (v[j] ?? 0);
        const newLambda = Math.sqrt(Av.reduce((ss, x) => ss + x * x, 0)) || 1;
        v = Av.map(x => x / newLambda);
        if (Math.abs(newLambda - lambda) < 1e-6) break;
        lambda = newLambda;
      }
      vecs.push(v); vals.push(lambda);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Ak[i]![j] = (Ak[i]?.[j] ?? 0) - lambda * (v[i] ?? 0) * (v[j] ?? 0);
    }
    return { vecs, vals };
  }

  get embedding(): Float64Array[] { return this.embedding_; }
}

export class ForceDirectedLayoutExt {
  private positions_: Float64Array[] = [];
  private fitted_ = false;

  constructor(
    private nDim: number = 2,
    private nIter: number = 500,
    private repulsionStrength: number = 1.0,
    private attractionStrength: number = 0.01,
    private coolingFactor: number = 0.99,
    private randomState: number = 42
  ) {}

  fitTransform(adjacency: Float64Array[]): Float64Array[] {
    const n = adjacency.length;
    let rng = this.randomState;
    const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 0xffffffff; };

    // Random init
    let pos = Array.from({ length: n }, () => new Float64Array(this.nDim).map(() => rand() * 2 - 1));
    let temp = 0.1 * Math.sqrt(n);

    for (let iter = 0; iter < this.nIter; iter++) {
      const forces = Array.from({ length: n }, () => new Float64Array(this.nDim));

      // Repulsion
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const diff = new Float64Array(this.nDim).map((_, k) => (pos[i]?.[k] ?? 0) - (pos[j]?.[k] ?? 0));
          const dist2 = diff.reduce((s, v) => s + v * v, 0) + 1e-10;
          const dist = Math.sqrt(dist2);
          const force = this.repulsionStrength / dist2;
          for (let k = 0; k < this.nDim; k++) {
            forces[i]![k] = (forces[i]?.[k] ?? 0) + force * (diff[k] ?? 0) / dist;
            forces[j]![k] = (forces[j]?.[k] ?? 0) - force * (diff[k] ?? 0) / dist;
          }
        }
      }

      // Attraction (for edges)
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const w = adjacency[i]?.[j] ?? 0;
          if (w === 0) continue;
          const diff = new Float64Array(this.nDim).map((_, k) => (pos[j]?.[k] ?? 0) - (pos[i]?.[k] ?? 0));
          const dist = Math.sqrt(diff.reduce((s, v) => s + v * v, 0) + 1e-10);
          const force = this.attractionStrength * w * dist;
          for (let k = 0; k < this.nDim; k++) {
            forces[i]![k] = (forces[i]?.[k] ?? 0) + force * (diff[k] ?? 0) / dist;
            forces[j]![k] = (forces[j]?.[k] ?? 0) - force * (diff[k] ?? 0) / dist;
          }
        }
      }

      // Apply forces with cooling
      for (let i = 0; i < n; i++) {
        const fNorm = Math.sqrt((forces[i]!).reduce((s, v) => s + v * v, 0)) || 1;
        const scale = Math.min(temp, fNorm) / fNorm;
        for (let k = 0; k < this.nDim; k++) pos[i]![k] = (pos[i]?.[k] ?? 0) + (forces[i]?.[k] ?? 0) * scale;
      }
      temp *= this.coolingFactor;
    }

    this.positions_ = pos;
    this.fitted_ = true;
    return pos;
  }

  get positions(): Float64Array[] { return this.positions_; }
}

export class TriMapExt {
  private embedding_: Float64Array[] = [];
  private fitted_ = false;

  constructor(
    private nComponents: number = 2,
    private nInliers: number = 10,
    private nOutliers: number = 5,
    private nRandom: number = 3,
    private nIter: number = 400,
    private learningRate: number = 0.1,
    private randomState: number = 42
  ) {}

  fitTransform(X: Float64Array[]): Float64Array[] {
    const n = X.length, p = X[0]?.length ?? 0;
    let rng = this.randomState;
    const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 0xffffffff; };

    // Initialize embedding
    let Y = Array.from({ length: n }, () => new Float64Array(this.nComponents).map(() => (rand() * 2 - 1) * 0.1));

    // Build triplets: (anchor, near, far)
    const dists = Array.from({ length: n }, (_, i) =>
      X.map((xj, j) => ({ j, d: X[i]!.reduce((s, v, k) => s + (v - (xj[k] ?? 0)) ** 2, 0) })).sort((a, b) => a.d - b.d)
    );

    const triplets: Array<[number, number, number]> = [];
    for (let i = 0; i < n; i++) {
      const nearN = Math.min(this.nInliers, dists[i]!.length - 1);
      for (let ni = 1; ni <= nearN; ni++) {
        const near = dists[i]![ni]!.j;
        for (let oi = 0; oi < this.nOutliers; oi++) {
          const far = Math.floor(rand() * n);
          if (far !== i && far !== near) triplets.push([i, near, far]);
        }
      }
    }

    // Gradient descent
    for (let iter = 0; iter < this.nIter; iter++) {
      const grads = Array.from({ length: n }, () => new Float64Array(this.nComponents));
      for (const [a, b, c] of triplets) {
        const dab = Y[a]!.reduce((s, v, k) => s + (v - (Y[b]?.[k] ?? 0)) ** 2, 0);
        const dac = Y[a]!.reduce((s, v, k) => s + (v - (Y[c]?.[k] ?? 0)) ** 2, 0);
        // Loss: log(1 + dab) - log(1 + dac) if dab > dac, else 0
        if (dab >= dac) {
          const w = 1 / ((1 + dab) * (1 + dac));
          for (let k = 0; k < this.nComponents; k++) {
            const ga = 2 * w * ((Y[a]?.[k] ?? 0) - (Y[b]?.[k] ?? 0)) / (1 + dab) - 2 * w * ((Y[a]?.[k] ?? 0) - (Y[c]?.[k] ?? 0)) / (1 + dac);
            grads[a]![k] = (grads[a]?.[k] ?? 0) + ga;
          }
        }
      }
      const lr = this.learningRate / (1 + 0.001 * iter);
      for (let i = 0; i < n; i++) for (let k = 0; k < this.nComponents; k++) Y[i]![k] = (Y[i]?.[k] ?? 0) - lr * (grads[i]?.[k] ?? 0);
    }

    this.embedding_ = Y;
    this.fitted_ = true;
    return Y;
  }

  get embedding(): Float64Array[] { return this.embedding_; }
}

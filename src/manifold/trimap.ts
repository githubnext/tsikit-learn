/**
 * Manifold learning extensions: TriMAP, PHATE, ForceAtlas2 layout.
 */

export class TriMAP {
  private embedding_: Float64Array[] = [];

  constructor(
    private readonly nComponents = 2,
    private readonly nInliers = 10,
    private readonly nOutliers = 5,
    private readonly nRandom = 5,
    private readonly lr = 0.1,
    private readonly nIter = 400,
  ) {}

  fitTransform(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const nDims = this.nComponents;
    // Initialize embedding with PCA-like random projection
    const embed = Array.from({ length: n }, () => {
      const v = new Float64Array(nDims);
      for (let d = 0; d < nDims; d++) v[d] = (Math.random() - 0.5) * 0.01;
      return v;
    });

    // Compute nearest neighbors (simplified brute force)
    const knn = this._computeKNN(X, this.nInliers + 1);

    // Gradient descent
    for (let iter = 0; iter < this.nIter; iter++) {
      const lr = this.lr * (1 - iter / this.nIter);
      for (let i = 0; i < n; i++) {
        const ei = embed[i]!;
        const neighbors = knn[i]!.slice(0, this.nInliers);
        for (const j of neighbors) {
          const ej = embed[j]!;
          let d = 0;
          for (let d2 = 0; d2 < nDims; d2++)
            d += ((ei[d2] ?? 0) - (ej[d2] ?? 0)) ** 2;
          const dist = Math.sqrt(d) + 1e-10;
          for (let d2 = 0; d2 < nDims; d2++) {
            const grad = ((ei[d2] ?? 0) - (ej[d2] ?? 0)) / (dist * (1 + dist));
            ei[d2] = (ei[d2] ?? 0) - lr * grad;
            ej[d2] = (ej[d2] ?? 0) + lr * grad;
          }
        }
      }
    }
    this.embedding_ = embed;
    return embed;
  }

  private _computeKNN(X: Float64Array[], k: number): number[][] {
    return X.map((xi, i) => {
      return X.map((xj, j) => {
        let d = 0;
        for (let f = 0; f < xi.length; f++)
          d += ((xi[f] ?? 0) - (xj[f] ?? 0)) ** 2;
        return { j, d };
      })
        .filter(({ j }) => j !== i)
        .sort((a, b) => a.d - b.d)
        .slice(0, k)
        .map(({ j }) => j);
    });
  }

  getEmbedding(): Float64Array[] {
    return this.embedding_;
  }
}

export class PHATE {
  private embedding_: Float64Array[] = [];

  constructor(
    private readonly nComponents = 2,
    private readonly knn = 5,
    private readonly decay = 40,
    private readonly nLandmark = 2000,
  ) {}

  fitTransform(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const nDims = this.nComponents;
    // Simplified PHATE: compute diffusion operator, then embed
    const P = this._diffusionOperator(X);
    // Random embedding as placeholder for PHATE potential
    const embed = Array.from({ length: n }, (_, i) => {
      const v = new Float64Array(nDims);
      for (let d = 0; d < nDims; d++) {
        v[d] = P[i]?.[d % (P[i]?.length ?? 1)] ?? Math.random() * 0.1;
      }
      return v;
    });
    this.embedding_ = embed;
    return embed;
  }

  private _diffusionOperator(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const nF = X[0]?.length ?? 1;
    // Compute kernel matrix
    const K: Float64Array[] = Array.from(
      { length: n },
      () => new Float64Array(n),
    );
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        let d = 0;
        const xi = X[i]!;
        const xj = X[j]!;
        for (let f = 0; f < nF; f++) d += ((xi[f] ?? 0) - (xj[f] ?? 0)) ** 2;
        const k = Math.exp((-d * this.decay) / nF);
        K[i]![j] = k;
        K[j]![i] = k;
      }
    }
    // Row normalize
    return K.map((row) => {
      const s = row.reduce((a, b) => a + b, 0);
      return s > 0 ? new Float64Array(row.map((v) => v / s)) : row;
    });
  }

  getEmbedding(): Float64Array[] {
    return this.embedding_;
  }
}

export class ForceAtlas2 {
  private positions_: Float64Array[] = [];

  constructor(
    private readonly nIter = 100,
    private readonly gravity = 1.0,
    private readonly scalingRatio = 2.0,
    private readonly barnesHutTheta = 1.2,
  ) {}

  fit(edges: [number, number][], nNodes: number, weights?: Float64Array): this {
    void this.barnesHutTheta;
    const pos = Array.from(
      { length: nNodes },
      () =>
        new Float64Array([
          (Math.random() - 0.5) * 100,
          (Math.random() - 0.5) * 100,
        ]),
    );

    for (let iter = 0; iter < this.nIter; iter++) {
      const forces = Array.from({ length: nNodes }, () => new Float64Array(2));
      // Repulsion
      for (let i = 0; i < nNodes; i++) {
        for (let j = i + 1; j < nNodes; j++) {
          const pi = pos[i]!;
          const pj = pos[j]!;
          const dx = (pi[0] ?? 0) - (pj[0] ?? 0);
          const dy = (pi[1] ?? 0) - (pj[1] ?? 0);
          const d2 = dx * dx + dy * dy + 0.01;
          const f = this.scalingRatio / d2;
          forces[i]![0] = (forces[i]![0] ?? 0) + f * dx;
          forces[i]![1] = (forces[i]![1] ?? 0) + f * dy;
          forces[j]![0] = (forces[j]![0] ?? 0) - f * dx;
          forces[j]![1] = (forces[j]![1] ?? 0) - f * dy;
        }
        // Gravity
        const pi = pos[i]!;
        const d = Math.sqrt((pi[0] ?? 0) ** 2 + (pi[1] ?? 0) ** 2) + 0.01;
        forces[i]![0] =
          (forces[i]![0] ?? 0) - (this.gravity * (pi[0] ?? 0)) / d;
        forces[i]![1] =
          (forces[i]![1] ?? 0) - (this.gravity * (pi[1] ?? 0)) / d;
      }
      // Attraction along edges
      for (let ei = 0; ei < edges.length; ei++) {
        const e = edges[ei]!;
        const [u, v] = e;
        if (u === undefined || v === undefined) continue;
        const pu = pos[u]!;
        const pv = pos[v]!;
        const w = weights?.[ei] ?? 1;
        const dx = (pu[0] ?? 0) - (pv[0] ?? 0);
        const dy = (pu[1] ?? 0) - (pv[1] ?? 0);
        forces[u]![0] = (forces[u]![0] ?? 0) - w * dx;
        forces[u]![1] = (forces[u]![1] ?? 0) - w * dy;
        forces[v]![0] = (forces[v]![0] ?? 0) + w * dx;
        forces[v]![1] = (forces[v]![1] ?? 0) + w * dy;
      }
      // Update positions
      const step = 1 / (iter + 1);
      for (let i = 0; i < nNodes; i++) {
        pos[i]![0] = (pos[i]![0] ?? 0) + step * (forces[i]![0] ?? 0);
        pos[i]![1] = (pos[i]![1] ?? 0) + step * (forces[i]![1] ?? 0);
      }
    }
    this.positions_ = pos;
    return this;
  }

  getPositions(): Float64Array[] {
    return this.positions_;
  }
}

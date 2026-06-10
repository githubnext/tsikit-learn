/**
 * Uniform Manifold Approximation and Projection (UMAP) simplified implementation.
 */

export class UMAPSimplified {
  private embedding_!: Float64Array[];
  private fitted_ = false;

  constructor(
    private nComponents = 2,
    private nNeighbors = 15,
    private minDist = 0.1,
    private nEpochs = 200,
    private learningRate = 1.0
  ) {}

  fit(X: Float64Array[]): this {
    const n = X.length;
    // Build kNN graph
    const knnGraph = this._buildKNNGraph(X);
    // Initialize embedding
    this.embedding_ = Array.from({ length: n }, () =>
      new Float64Array(this.nComponents).map(() => (Math.random() - 0.5) * 10)
    );
    // Optimize
    const a = this._findABParams();
    for (let epoch = 0; epoch < this.nEpochs; epoch++) {
      const lr = this.learningRate * (1 - epoch / this.nEpochs);
      for (let i = 0; i < n; i++) {
        for (const { j, weight } of knnGraph[i]!) {
          // Attractive force
          const d = this._dist(this.embedding_[i]!, this.embedding_[j]!);
          const grad_coef = -(2 * a.a * d ** (2 * (a.b - 1)) * weight) / (a.a * d ** (2 * a.b) + 1);
          for (let k = 0; k < this.nComponents; k++) {
            this.embedding_[i]![k] = (this.embedding_[i]![k] ?? 0) + lr * grad_coef * ((this.embedding_[i]![k] ?? 0) - (this.embedding_[j]![k] ?? 0));
          }
          // Repulsive force (negative sampling)
          const ni = (i + 1 + Math.floor(Math.random() * (n - 2))) % n;
          const dr = Math.max(this._dist(this.embedding_[i]!, this.embedding_[ni]!), 1e-6);
          const repGrad = 2 * a.b / ((0.001 + dr ** 2) * (a.a * dr ** (2 * a.b) + 1));
          for (let k = 0; k < this.nComponents; k++) {
            this.embedding_[i]![k] = (this.embedding_[i]![k] ?? 0) + lr * repGrad * ((this.embedding_[i]![k] ?? 0) - (this.embedding_[ni]![k] ?? 0));
          }
        }
      }
    }
    this.fitted_ = true;
    return this;
  }

  private _dist(a: Float64Array, b: Float64Array): number {
    return Math.sqrt(a.reduce((s, v, i) => s + (v - (b[i] ?? 0)) ** 2, 0));
  }

  private _buildKNNGraph(X: Float64Array[]): Array<Array<{ j: number; weight: number }>> {
    const n = X.length;
    return Array.from({ length: n }, (_, i) => {
      const dists = X.map((xj, j) => ({ j, d: this._dist(X[i]!, xj) }));
      dists.sort((a, b) => a.d - b.d);
      const knn = dists.slice(1, this.nNeighbors + 1);
      // Compute bandwidth sigma for each point
      const rho = knn[0]?.d ?? 0;
      const target = Math.log2(this.nNeighbors);
      let lo = 0, hi = 1000, sigma = 1;
      for (let iter = 0; iter < 64; iter++) {
        const mid = (lo + hi) / 2;
        const s = knn.reduce((acc, { d }) => acc + Math.exp(-(Math.max(d - rho, 0)) / mid), 0);
        if (s < target) lo = mid; else hi = mid;
        sigma = mid;
      }
      return knn.map(({ j, d }) => ({ j, weight: Math.exp(-(Math.max(d - rho, 0)) / sigma) }));
    });
  }

  private _findABParams(): { a: number; b: number } {
    // Approximate a, b from minDist
    const spread = 1.0;
    const b = this.minDist < 0.5 ? 1 : 1.0793 - 0.0944 * Math.log(this.minDist);
    const a = Math.pow(spread, b) / Math.pow(this.minDist, b);
    return { a, b };
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return this.embedding_;
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.fit(X).transform(X); }
  get embedding(): Float64Array[] { return this.embedding_; }
}

export class TriMapSimplified {
  private embedding_!: Float64Array[];
  private fitted_ = false;

  constructor(private nComponents = 2, private nInliers = 10, private nOutliers = 5, private nRandom = 5, private lr = 0.1, private nIters = 400) {}

  fit(X: Float64Array[]): this {
    const n = X.length;
    this.embedding_ = Array.from({ length: n }, () => new Float64Array(this.nComponents).map(() => (Math.random() - 0.5)));
    // Build triplets
    const triplets: Array<[number, number, number]> = [];
    for (let i = 0; i < n; i++) {
      const dists = X.map((xj, j) => ({ j, d: X[i]!.reduce((s, v, k) => s + (v - (xj[k] ?? 0)) ** 2, 0) })).sort((a, b) => a.d - b.d);
      for (let k = 1; k <= this.nInliers; k++) {
        for (let o = 0; o < this.nOutliers; o++) {
          const outlier = Math.floor(Math.random() * n);
          if (outlier !== i) triplets.push([i, dists[k]!.j, outlier]);
        }
      }
    }
    // Gradient descent
    for (let iter = 0; iter < this.nIters; iter++) {
      const lr = this.lr * (1 - iter / this.nIters);
      for (const [i, j, k] of triplets) {
        const ei = this.embedding_[i]!, ej = this.embedding_[j]!, ek = this.embedding_[k]!;
        const dij = Math.max(ei.reduce((s, v, d) => s + (v - (ej[d] ?? 0)) ** 2, 0), 1e-10);
        const dik = Math.max(ei.reduce((s, v, d) => s + (v - (ek[d] ?? 0)) ** 2, 0), 1e-10);
        if (dij < dik) continue; // Triplet satisfied
        const g = 1 / (1 + Math.exp(dik - dij));
        for (let d = 0; d < this.nComponents; d++) {
          const gij = g * 2 * ((ei[d] ?? 0) - (ej[d] ?? 0)) / dij;
          const gik = -g * 2 * ((ei[d] ?? 0) - (ek[d] ?? 0)) / dik;
          this.embedding_[i]![d] = (this.embedding_[i]![d] ?? 0) - lr * (gij + gik);
          this.embedding_[j]![d] = (this.embedding_[j]![d] ?? 0) + lr * gij;
          this.embedding_[k]![d] = (this.embedding_[k]![d] ?? 0) - lr * gik;
        }
      }
    }
    this.fitted_ = true;
    void this.nRandom;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return this.embedding_;
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.fit(X).transform(X); }
}

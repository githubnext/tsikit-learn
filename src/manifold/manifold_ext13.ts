/**
 * Diffusion Maps and spectral embedding extensions.
 */

export class DiffusionMaps {
  private embedding_!: Float64Array[];
  private diffusionEigenvalues_!: Float64Array;
  private fitted_ = false;

  constructor(
    private nComponents = 2,
    private epsilon: number | null = null,
    private nSteps = 1,
    private alpha = 0.5
  ) {}

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 1;
    // Estimate epsilon as median pairwise distance
    const sigma = this.epsilon ?? (() => {
      const dists: number[] = [];
      for (let i = 0; i < Math.min(n, 50); i++) {
        for (let j = i + 1; j < Math.min(n, 50); j++) {
          dists.push(Math.sqrt(X[i]!.reduce((s, v, k) => s + (v - (X[j]![k] ?? 0)) ** 2, 0)));
        }
      }
      dists.sort((a, b) => a - b);
      return (dists[Math.floor(dists.length / 2)] ?? 1) ** 2;
    })();

    // Build affinity matrix
    const K = Array.from({ length: n }, (_, i) =>
      new Float64Array(n).map((_, j) => {
        const d2 = X[i]!.reduce((s, v, k) => s + (v - (X[j]![k] ?? 0)) ** 2, 0);
        return Math.exp(-d2 / sigma);
      })
    );

    // Anisotropic normalization (alpha-normalization)
    const q = new Float64Array(n).map((_, i) => K[i]!.reduce((s, v) => s + v, 0));
    const Knorm = K.map((row, i) => new Float64Array(row.map((v, j) => v / ((q[i] ?? 1) ** this.alpha * (q[j] ?? 1) ** this.alpha))));

    // Row normalize to get Markov matrix
    const rowSums = Knorm.map(row => row.reduce((s, v) => s + v, 0));
    const M = Knorm.map((row, i) => new Float64Array(row.map(v => v / (rowSums[i] ?? 1))));

    // Power iteration for eigenvectors
    const { eigenvalues, eigenvectors } = this._powerIteration(M, this.nComponents + 1, n);

    // Skip trivial eigenvalue (first one is 1)
    this.diffusionEigenvalues_ = eigenvalues.slice(1, this.nComponents + 1);
    const vecs = eigenvectors.slice(1, this.nComponents + 1);

    // Scale by eigenvalues^nSteps
    this.embedding_ = Array.from({ length: n }, (_, i) =>
      new Float64Array(this.nComponents).map((_, k) => (vecs[k]![i] ?? 0) * (this.diffusionEigenvalues_[k] ?? 1) ** this.nSteps)
    );
    this.fitted_ = true;
    void p;
    return this;
  }

  private _powerIteration(M: Float64Array[], k: number, n: number): { eigenvalues: Float64Array; eigenvectors: Float64Array[] } {
    const eigenvalues = new Float64Array(k);
    const eigenvectors: Float64Array[] = [];
    let deflated = M.map(row => new Float64Array(row));

    for (let e = 0; e < k; e++) {
      let v = new Float64Array(n).map(() => Math.random());
      let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
      for (let j = 0; j < n; j++) v[j] = (v[j] ?? 0) / norm;

      for (let iter = 0; iter < 100; iter++) {
        const vNew = new Float64Array(n).map((_, i) => deflated[i]!.reduce((s, mij, j) => s + mij * (v[j] ?? 0), 0));
        norm = Math.sqrt(vNew.reduce((s, x) => s + x * x, 0));
        for (let j = 0; j < n; j++) vNew[j] = (vNew[j] ?? 0) / (norm + 1e-10);
        const diff = Math.sqrt(vNew.reduce((s, vj, j) => s + (vj - (v[j] ?? 0)) ** 2, 0));
        v = vNew;
        if (diff < 1e-8) break;
      }

      const lambda = v.reduce((s, vj, i) => s + vj * deflated[i]!.reduce((ss, mij, j) => ss + mij * (v[j] ?? 0), 0), 0);
      eigenvalues[e] = lambda;
      eigenvectors.push(new Float64Array(v));

      // Deflate
      deflated = deflated.map((row, i) => new Float64Array(row.map((mij, j) => mij - lambda * (v[i] ?? 0) * (v[j] ?? 0))));
    }
    return { eigenvalues, eigenvectors };
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return this.embedding_;
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.fit(X).transform(X); }
  get embedding(): Float64Array[] { return this.embedding_; }
}

export class LaplacianEigenmaps {
  private embedding_!: Float64Array[];
  private fitted_ = false;

  constructor(private nComponents = 2, private nNeighbors = 10) {}

  fit(X: Float64Array[]): this {
    const n = X.length;
    const W = Array.from({ length: n }, () => new Float64Array(n));

    // Build kNN graph
    for (let i = 0; i < n; i++) {
      const dists = X.map((xj, j) => ({ j, d: Math.sqrt(X[i]!.reduce((s, v, k) => s + (v - (xj[k] ?? 0)) ** 2, 0)) }));
      dists.sort((a, b) => a.d - b.d);
      for (let k = 1; k <= Math.min(this.nNeighbors, n - 1); k++) {
        W[i]![dists[k]!.j] = 1;
        W[dists[k]!.j]![i] = 1;
      }
    }

    // Degree matrix and normalized Laplacian
    const D = new Float64Array(n).map((_, i) => W[i]!.reduce((s, v) => s + v, 0));
    const L = Array.from({ length: n }, (_, i) =>
      new Float64Array(n).map((_, j) => {
        if (i === j) return 1;
        return D[i]! > 0 && D[j]! > 0 ? -(W[i]![j] ?? 0) / Math.sqrt((D[i] ?? 1) * (D[j] ?? 1)) : 0;
      })
    );

    // Power iteration for smallest eigenvectors (excluding trivial)
    const k = this.nComponents + 1;
    const eigenvalues = new Float64Array(k);
    const eigenvectors: Float64Array[] = [];
    let deflated = L.map(row => new Float64Array(row));

    for (let e = 0; e < k; e++) {
      let v = new Float64Array(n).map((_, i) => i === e ? 1 : Math.random() * 0.01);
      let nrm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
      for (let j = 0; j < n; j++) v[j] = (v[j] ?? 0) / (nrm + 1e-10);
      for (let iter = 0; iter < 200; iter++) {
        // Inverse iteration approximation: shift-and-invert with identity shift
        const Lv = new Float64Array(n).map((_, i) => deflated[i]!.reduce((s, lij, j) => s + lij * (v[j] ?? 0), 0));
        const lambda = v.reduce((s, vj, i) => s + vj * (Lv[i] ?? 0), 0);
        // Orthogonalize against previous
        let vNext = new Float64Array(v.map((vi, i) => vi - lambda * (Lv[i] ?? 0) * 0.01));
        for (const prev of eigenvectors) {
          const dot = vNext.reduce((s, vj, i) => s + vj * (prev[i] ?? 0), 0);
          for (let i = 0; i < n; i++) vNext[i] = (vNext[i] ?? 0) - dot * (prev[i] ?? 0);
        }
        nrm = Math.sqrt(vNext.reduce((s, x) => s + x * x, 0));
        vNext = new Float64Array(vNext.map(x => x / (nrm + 1e-10)));
        if (Math.sqrt(vNext.reduce((s, vj, j) => s + (vj - (v[j] ?? 0)) ** 2, 0)) < 1e-6) { v = vNext; break; }
        v = vNext;
      }
      eigenvalues[e] = v.reduce((s, vj, i) => s + vj * deflated[i]!.reduce((ss, lij, j) => ss + lij * (v[j] ?? 0), 0), 0);
      eigenvectors.push(new Float64Array(v));
    }

    this.embedding_ = Array.from({ length: n }, (_, i) =>
      new Float64Array(this.nComponents).map((_, k_) => eigenvectors[k_ + 1]![i] ?? 0)
    );
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return this.embedding_;
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.fit(X).transform(X); }
}

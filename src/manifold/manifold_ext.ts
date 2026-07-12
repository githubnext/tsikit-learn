/**
 * UMAP-like dimensionality reduction — manifold learning extension.
 */

function euclidean(a: Float64Array, b: Float64Array): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  return Math.sqrt(d);
}

export class UMAPLite {
  nComponents: number;
  nNeighbors: number;
  minDist: number;
  spread: number;
  nEpochs: number;
  learningRate: number;
  randomState: number;
  embedding_: Float64Array[] | null = null;

  constructor(
    nComponents = 2,
    nNeighbors = 15,
    minDist = 0.1,
    spread = 1.0,
    nEpochs = 200,
    learningRate = 1.0,
    randomState = 42,
  ) {
    this.nComponents = nComponents;
    this.nNeighbors = nNeighbors;
    this.minDist = minDist;
    this.spread = spread;
    this.nEpochs = nEpochs;
    this.learningRate = learningRate;
    this.randomState = randomState;
  }

  private _knn(X: Float64Array[], k: number): { indices: Int32Array[]; dists: Float64Array[] } {
    const n = X.length;
    const indices: Int32Array[] = [];
    const dists: Float64Array[] = [];
    for (let i = 0; i < n; i++) {
      const ds = Array.from({ length: n }, (_, j) => ({ j, d: euclidean(X[i] as Float64Array, X[j] as Float64Array) }));
      ds.sort((a, b) => a.d - b.d);
      const knn = ds.slice(1, k + 1);
      indices.push(Int32Array.from(knn, (e) => e.j));
      dists.push(Float64Array.from(knn, (e) => e.d));
    }
    return { indices, dists };
  }

  private _fuzzyMembership(dists: Float64Array[], rho: Float64Array, sigma: Float64Array): Float64Array[] {
    const n = dists.length;
    const k = dists[0]?.length ?? 0;
    return dists.map((di, i) => di.map((d) => Math.exp(-(Math.max(0, d - (rho[i] ?? 0)) / (sigma[i] ?? 1)))));
  }

  private _abParams(minDist: number, spread: number): { a: number; b: number } {
    // Fit a, b for the UMAP curve 1 / (1 + a * d^(2b))
    let a = 1.0, b = 1.0;
    for (let iter = 0; iter < 50; iter++) {
      let gradA = 0, gradB = 0;
      for (let d = 0; d < 10; d++) {
        const dVal = d * spread / 10;
        const y = dVal <= minDist ? 1 : Math.exp(-(dVal - minDist) / spread);
        const pred = 1 / (1 + a * dVal ** (2 * b));
        const err = pred - y;
        gradA += err * -(dVal ** (2 * b)) / (1 + a * dVal ** (2 * b)) ** 2;
        gradB += err * a * -(dVal ** (2 * b)) * Math.log(Math.max(dVal, 1e-8)) * 2 / (1 + a * dVal ** (2 * b)) ** 2;
      }
      a -= 0.1 * gradA;
      b -= 0.1 * gradB;
    }
    return { a: Math.max(0.1, a), b: Math.max(0.1, b) };
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const k = Math.min(this.nNeighbors, n - 1);
    const { indices, dists } = this._knn(X, k);

    // Compute rho (distance to nearest neighbor) and sigma
    const rho = Float64Array.from({ length: n }, (_, i) => (dists[i] as Float64Array)[0] ?? 0);
    const sigma = new Float64Array(n).fill(1.0);
    for (let i = 0; i < n; i++) {
      let lo = 0, hi = 1000;
      for (let iter = 0; iter < 30; iter++) {
        const mid = (lo + hi) / 2;
        const s = (dists[i] as Float64Array).reduce((sum, d) => sum + Math.exp(-(Math.max(0, d - (rho[i] ?? 0)) / mid)), 0);
        if (s > Math.log2(k)) hi = mid; else lo = mid;
        sigma[i] = mid;
      }
    }

    // Initialize embedding randomly
    let seed = this.randomState;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return (seed >>> 0) / 4294967296;
    };
    this.embedding_ = Array.from({ length: n }, () => Float64Array.from({ length: this.nComponents }, () => rand() * 0.1 - 0.05));

    const { a, b } = this._abParams(this.minDist, this.spread);

    // Optimize embedding via SGD
    for (let epoch = 0; epoch < this.nEpochs; epoch++) {
      const lr = this.learningRate * (1 - epoch / this.nEpochs);
      for (let i = 0; i < n; i++) {
        // Attractive force for each neighbor
        for (let ki = 0; ki < k; ki++) {
          const j = (indices[i] as Int32Array)[ki] ?? 0;
          const wi = Math.exp(-((dists[i] as Float64Array)[ki] ?? 0));
          const ei = this.embedding_[i] as Float64Array;
          const ej = this.embedding_[j] as Float64Array;
          const d2 = ei.reduce((s, v, d) => s + (v - (ej[d] ?? 0)) ** 2, 0);
          const denom = 1 + a * d2 ** b;
          const grad = wi * 2 * a * b * d2 ** (b - 1) / denom;
          for (let d = 0; d < this.nComponents; d++) {
            (this.embedding_[i] as Float64Array)[d] = (ei[d] ?? 0) - lr * grad * ((ei[d] ?? 0) - (ej[d] ?? 0));
          }
        }

        // Repulsive force for random negative samples
        const nNeg = 5;
        for (let ni = 0; ni < nNeg; ni++) {
          const j = Math.floor(rand() * n);
          if (j === i) continue;
          const ei = this.embedding_[i] as Float64Array;
          const ej = this.embedding_[j] as Float64Array;
          const d2 = Math.max(ei.reduce((s, v, d) => s + (v - (ej[d] ?? 0)) ** 2, 0), 1e-4);
          const grad = 2 * b / (d2 * (1 + a * d2 ** b));
          for (let d = 0; d < this.nComponents; d++) {
            (this.embedding_[i] as Float64Array)[d] = (ei[d] ?? 0) + lr * grad * ((ei[d] ?? 0) - (ej[d] ?? 0));
          }
        }
      }
      void rho; void sigma;
    }
    return this;
  }

  transform(_X: Float64Array[]): Float64Array[] {
    if (!this.embedding_) throw new Error("Not fitted");
    return this.embedding_;
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class TruncatedSNE {
  nComponents: number;
  perplexity: number;
  nIter: number;
  learningRate: number;
  momentum: number;
  embedding_: Float64Array[] | null = null;

  constructor(nComponents = 2, perplexity = 30, nIter = 250, learningRate = 200, momentum = 0.8) {
    this.nComponents = nComponents;
    this.perplexity = perplexity;
    this.nIter = nIter;
    this.learningRate = learningRate;
    this.momentum = momentum;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const d = this.nComponents;

    // Initialize embedding
    this.embedding_ = Array.from({ length: n }, () => Float64Array.from({ length: d }, () => (Math.random() - 0.5) * 0.0001));

    // Compute P (pairwise affinities) — simplified with fixed sigma
    const P: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      let sumExp = 0;
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const dist2 = (X[i] as Float64Array).reduce((s, v, k) => s + (v - ((X[j] as Float64Array)[k] ?? 0)) ** 2, 0);
        const exp = Math.exp(-dist2 / 2);
        (P[i] as Float64Array)[j] = exp;
        sumExp += exp;
      }
      for (let j = 0; j < n; j++) (P[i] as Float64Array)[j]! /= Math.max(sumExp, 1e-12);
    }
    // Symmetrize
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < i; j++) {
        const pij = (((P[i] as Float64Array)[j] ?? 0) + ((P[j] as Float64Array)[i] ?? 0)) / (2 * n);
        (P[i] as Float64Array)[j] = Math.max(pij, 1e-12);
        (P[j] as Float64Array)[i] = Math.max(pij, 1e-12);
      }
    }

    let gains = Array.from({ length: n }, () => new Float64Array(d).fill(1));
    let incs = Array.from({ length: n }, () => new Float64Array(d));

    for (let iter = 0; iter < this.nIter; iter++) {
      // Compute Q
      const Q: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));
      let sumQ = 0;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const dist2 = (this.embedding_[i] as Float64Array).reduce((s, v, k) => s + (v - ((this.embedding_![j] as Float64Array)[k] ?? 0)) ** 2, 0);
          const q = 1 / (1 + dist2);
          (Q[i] as Float64Array)[j] = q;
          (Q[j] as Float64Array)[i] = q;
          sumQ += 2 * q;
        }
      }
      if (sumQ < 1e-12) sumQ = 1e-12;

      // Compute gradient
      const grad = Array.from({ length: n }, () => new Float64Array(d));
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (i === j) continue;
          const pij = (P[i] as Float64Array)[j] ?? 0;
          const qij = (Q[i] as Float64Array)[j] ?? 0;
          const factor = 4 * (pij - qij / sumQ) / (1 + (this.embedding_![i] as Float64Array).reduce((s, v, k) => s + (v - ((this.embedding_![j] as Float64Array)[k]! ?? 0)) ** 2, 0));
          for (let k = 0; k < d; k++) {
            (grad[i] as Float64Array)[k]! += factor * (((this.embedding_[i] as Float64Array)[k] ?? 0) - ((this.embedding_![j] as Float64Array)[k] ?? 0));
          }
        }
      }

      // Update embedding
      for (let i = 0; i < n; i++) {
        for (let k = 0; k < d; k++) {
          const g = (grad[i] as Float64Array)[k] ?? 0;
          const inc = (incs[i] as Float64Array)[k] ?? 0;
          const gain = Math.max(0.1, ((g > 0) !== (inc > 0)) ? ((gains[i] as Float64Array)[k] ?? 0) + 0.2 : ((gains[i] as Float64Array)[k] ?? 0) * 0.8);
          (gains[i] as Float64Array)[k] = gain;
          const newInc = this.momentum * inc - this.learningRate * gain * g;
          (incs[i] as Float64Array)[k] = newInc;
          (this.embedding_![i] as Float64Array)[k] = ((this.embedding_![i] as Float64Array)[k] ?? 0) + newInc;
        }
      }
    }
    return this;
  }

  transform(_X: Float64Array[]): Float64Array[] {
    return this.embedding_ ?? [];
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

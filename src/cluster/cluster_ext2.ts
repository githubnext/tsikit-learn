/**
 * KohonenSOM and CompetitiveLearning — unsupervised neural clustering.
 */

export class KohonenSOM {
  mapWidth: number;
  mapHeight: number;
  inputDim: number;
  nIter: number;
  learningRate0: number;
  sigma0: number;
  private weights: Float64Array[][] | null = null;
  labels_: Int32Array | null = null;
  inertia_: number = 0;

  constructor(
    mapWidth = 10,
    mapHeight = 10,
    inputDim = 2,
    nIter = 1000,
    learningRate0 = 0.5,
    sigma0 = 3.0,
  ) {
    this.mapWidth = mapWidth;
    this.mapHeight = mapHeight;
    this.inputDim = inputDim;
    this.nIter = nIter;
    this.learningRate0 = learningRate0;
    this.sigma0 = sigma0;
  }

  private _bmuIndex(x: Float64Array): [number, number] {
    let bestDist = Number.POSITIVE_INFINITY;
    let bestR = 0, bestC = 0;
    for (let r = 0; r < this.mapHeight; r++) {
      for (let c = 0; c < this.mapWidth; c++) {
        const w = this.weights![r]![c] as Float64Array;
        const dist = x.reduce((s, v, j) => s + (v - (w[j] ?? 0)) ** 2, 0);
        if (dist < bestDist) { bestDist = dist; bestR = r; bestC = c; }
      }
    }
    return [bestR, bestC];
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? this.inputDim;
    this.inputDim = p;

    // Initialize weights randomly from data range
    const mins = new Float64Array(p), maxs = new Float64Array(p).fill(1);
    for (let j = 0; j < p; j++) {
      mins[j] = Math.min(...X.map((r) => r[j] ?? 0));
      maxs[j] = Math.max(...X.map((r) => r[j] ?? 0));
    }
    this.weights = Array.from({ length: this.mapHeight }, () =>
      Array.from({ length: this.mapWidth }, () =>
        Float64Array.from({ length: p }, (_, j) =>
          (mins[j] ?? 0) + Math.random() * ((maxs[j] ?? 1) - (mins[j] ?? 0))
        )
      )
    );

    for (let iter = 0; iter < this.nIter; iter++) {
      const t = iter / this.nIter;
      const lr = this.learningRate0 * Math.exp(-t * 4);
      const sigma = this.sigma0 * Math.exp(-t * 4);
      const x = X[Math.floor(Math.random() * n)] as Float64Array;
      const [br, bc] = this._bmuIndex(x);
      for (let r = 0; r < this.mapHeight; r++) {
        for (let c = 0; c < this.mapWidth; c++) {
          const dist2 = (r - br) ** 2 + (c - bc) ** 2;
          const h = Math.exp(-dist2 / (2 * sigma ** 2));
          const w = this.weights[r]![c] as Float64Array;
          for (let j = 0; j < p; j++) w[j] = (w[j] ?? 0) + lr * h * ((x[j] ?? 0) - (w[j] ?? 0));
        }
      }
    }

    // Assign labels
    this.labels_ = new Int32Array(n);
    let inertia = 0;
    for (let i = 0; i < n; i++) {
      const [r, c] = this._bmuIndex(X[i] as Float64Array);
      this.labels_[i] = r * this.mapWidth + c;
      const w = this.weights[r]![c] as Float64Array;
      inertia += (X[i] as Float64Array).reduce((s, v, j) => s + (v - (w[j] ?? 0)) ** 2, 0);
    }
    this.inertia_ = inertia;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.weights) throw new Error("Not fitted");
    return Int32Array.from(X.map((x) => {
      const [r, c] = this._bmuIndex(x);
      return r * this.mapWidth + c;
    }));
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.weights) throw new Error("Not fitted");
    return X.map((x) => {
      const dists = new Float64Array(this.mapHeight * this.mapWidth);
      for (let r = 0; r < this.mapHeight; r++) {
        for (let c = 0; c < this.mapWidth; c++) {
          const w = this.weights![r]![c] as Float64Array;
          dists[r * this.mapWidth + c] = Math.sqrt(x.reduce((s, v, j) => s + (v - (w[j] ?? 0)) ** 2, 0));
        }
      }
      return dists;
    });
  }

  getWeightMatrix(): Float64Array[][] {
    return this.weights ?? [];
  }
}

export class NeuralGas {
  nNeurons: number;
  maxIter: number;
  learningRate0: number;
  lambda0: number;
  private weights: Float64Array[] | null = null;
  labels_: Int32Array | null = null;
  inertia_: number = 0;

  constructor(nNeurons = 20, maxIter = 500, learningRate0 = 0.5, lambda0 = 5.0) {
    this.nNeurons = nNeurons;
    this.maxIter = maxIter;
    this.learningRate0 = learningRate0;
    this.lambda0 = lambda0;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    // Initialize from random data points
    const indices = Array.from({ length: this.nNeurons }, () => Math.floor(Math.random() * n));
    this.weights = indices.map((i) => new Float64Array(X[i] as Float64Array));

    for (let iter = 0; iter < this.maxIter; iter++) {
      const t = iter / this.maxIter;
      const lr = this.learningRate0 * Math.exp(-t * 6);
      const lam = this.lambda0 * Math.exp(-t * 6);
      const x = X[Math.floor(Math.random() * n)] as Float64Array;

      // Sort neurons by distance to x
      const dists = this.weights.map((w, k) => ({
        k,
        d: w.reduce((s, v, j) => s + (v - (x[j] ?? 0)) ** 2, 0),
      }));
      dists.sort((a, b) => a.d - b.d);

      for (let rank = 0; rank < dists.length; rank++) {
        const h = Math.exp(-rank / lam);
        const w = this.weights[dists[rank]!.k] as Float64Array;
        for (let j = 0; j < p; j++) w[j] = (w[j] ?? 0) + lr * h * ((x[j] ?? 0) - (w[j] ?? 0));
      }
    }

    this.labels_ = new Int32Array(n);
    let inertia = 0;
    for (let i = 0; i < n; i++) {
      let bestK = 0, bestD = Number.POSITIVE_INFINITY;
      for (let k = 0; k < this.weights.length; k++) {
        const d = (X[i] as Float64Array).reduce((s, v, j) => s + (v - ((this.weights![k] as Float64Array)[j] ?? 0)) ** 2, 0);
        if (d < bestD) { bestD = d; bestK = k; }
      }
      this.labels_[i] = bestK;
      inertia += bestD;
    }
    this.inertia_ = inertia;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.weights) throw new Error("Not fitted");
    return Int32Array.from(X.map((x) => {
      let bestK = 0, bestD = Number.POSITIVE_INFINITY;
      for (let k = 0; k < (this.weights?.length ?? 0); k++) {
        const d = x.reduce((s, v, j) => s + (v - ((this.weights![k] as Float64Array)[j] ?? 0)) ** 2, 0);
        if (d < bestD) { bestD = d; bestK = k; }
      }
      return bestK;
    }));
  }
}

export class CompetitiveLearning {
  nClusters: number;
  learningRate: number;
  nIter: number;
  private centers: Float64Array[] | null = null;
  labels_: Int32Array | null = null;
  inertia_: number = 0;

  constructor(nClusters = 8, learningRate = 0.1, nIter = 100) {
    this.nClusters = nClusters;
    this.learningRate = learningRate;
    this.nIter = nIter;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const indices = Array.from({ length: this.nClusters }, () => Math.floor(Math.random() * n));
    this.centers = indices.map((i) => new Float64Array(X[i] as Float64Array));

    for (let iter = 0; iter < this.nIter; iter++) {
      for (const x of X) {
        let bestK = 0, bestD = Number.POSITIVE_INFINITY;
        for (let k = 0; k < this.nClusters; k++) {
          const d = x.reduce((s, v, j) => s + (v - ((this.centers![k] as Float64Array)[j] ?? 0)) ** 2, 0);
          if (d < bestD) { bestD = d; bestK = k; }
        }
        const w = this.centers[bestK] as Float64Array;
        for (let j = 0; j < p; j++) w[j] = (w[j] ?? 0) + this.learningRate * ((x[j] ?? 0) - (w[j] ?? 0));
      }
    }

    this.labels_ = new Int32Array(n);
    let inertia = 0;
    for (let i = 0; i < n; i++) {
      let bestK = 0, bestD = Number.POSITIVE_INFINITY;
      for (let k = 0; k < this.nClusters; k++) {
        const d = (X[i] as Float64Array).reduce((s, v, j) => s + (v - ((this.centers![k] as Float64Array)[j] ?? 0)) ** 2, 0);
        if (d < bestD) { bestD = d; bestK = k; }
      }
      this.labels_[i] = bestK;
      inertia += bestD;
    }
    this.inertia_ = inertia;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.centers) throw new Error("Not fitted");
    return Int32Array.from(X.map((x) => {
      let bestK = 0, bestD = Number.POSITIVE_INFINITY;
      for (let k = 0; k < this.nClusters; k++) {
        const d = x.reduce((s, v, j) => s + (v - ((this.centers![k] as Float64Array)[j] ?? 0)) ** 2, 0);
        if (d < bestD) { bestD = d; bestK = k; }
      }
      return bestK;
    }));
  }

  clusterCenters(): Float64Array[] {
    return this.centers ?? [];
  }
}

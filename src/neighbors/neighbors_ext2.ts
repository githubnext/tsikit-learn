/**
 * Extended neighbors: NCA (Neighborhood Components Analysis), RadiusNeighborsClassifierExt, NearestNeighborsGraphExt
 */

export class NCA {
  private nComponents: number;
  private maxIter: number;
  private learningRate: number;
  components_: Float64Array[] | null = null;
  nIter_: number = 0;

  constructor(nComponents?: number, maxIter = 50, learningRate = 0.01) {
    this.nComponents = nComponents ?? 0;
    this.maxIter = maxIter;
    this.learningRate = learningRate;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    const k = this.nComponents > 0 ? this.nComponents : d;

    // Initialize with identity-like matrix
    let A: Float64Array[] = Array.from({ length: k }, (_, i) => {
      const row = new Float64Array(d);
      if (i < d) row[i] = 1;
      return row;
    });

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Compute transformed X
      const Xt = X.map((xi) => {
        const row = new Float64Array(k);
        for (let l = 0; l < k; l++) for (let j = 0; j < d; j++) row[l] += (A[l]![j] ?? 0) * (xi[j] ?? 0);
        return row;
      });

      // Compute softmax probabilities p_ij
      const grad: Float64Array[] = Array.from({ length: k }, () => new Float64Array(d));
      let objective = 0;

      for (let i = 0; i < n; i++) {
        const dists = new Float64Array(n);
        for (let j = 0; j < n; j++) {
          if (j === i) { dists[j] = 0; continue; }
          let d2 = 0;
          for (let l = 0; l < k; l++) d2 += ((Xt[i]![l] ?? 0) - (Xt[j]![l] ?? 0)) ** 2;
          dists[j] = d2;
        }
        const minDist = Math.min(...Array.from(dists).filter((_, j) => j !== i));
        const expDists = new Float64Array(n);
        let sumExp = 0;
        for (let j = 0; j < n; j++) {
          if (j === i) continue;
          expDists[j] = Math.exp(-(dists[j] ?? 0) + minDist);
          sumExp += expDists[j] ?? 0;
        }
        const pij = new Float64Array(n);
        for (let j = 0; j < n; j++) pij[j] = sumExp > 0 ? (expDists[j] ?? 0) / sumExp : 0;
        const pi = Array.from(pij).reduce((acc, p, j) => acc + (y[j] === y[i] ? p : 0), 0);
        objective += pi;

        // Gradient computation (simplified)
        for (let j = 0; j < n; j++) {
          if (j === i) continue;
          const pij_val = pij[j] ?? 0;
          const sameClass = y[j] === y[i] ? 1 : 0;
          const factor = pij_val * ((sameClass ? 1 : 0) - pi);
          const diff = new Float64Array(k);
          for (let l = 0; l < k; l++) diff[l] = (Xt[i]![l] ?? 0) - (Xt[j]![l] ?? 0);
          for (let l = 0; l < k; l++) {
            for (let m = 0; m < d; m++) {
              grad[l]![m] = (grad[l]![m] ?? 0) + 2 * factor * (diff[l] ?? 0) * ((X[i]![m] ?? 0) - (X[j]![m] ?? 0));
            }
          }
        }
      }

      // Update A
      for (let l = 0; l < k; l++) for (let m = 0; m < d; m++) {
        A[l]![m] = (A[l]![m] ?? 0) + this.learningRate * (grad[l]![m] ?? 0) / n;
      }

      this.nIter_ = iter + 1;
    }

    this.components_ = A;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_) throw new Error("Not fitted");
    const A = this.components_;
    const k = A.length;
    const d = A[0]?.length ?? 0;
    return X.map((xi) => {
      const row = new Float64Array(k);
      for (let l = 0; l < k; l++) for (let j = 0; j < d; j++) row[l] += (A[l]![j] ?? 0) * (xi[j] ?? 0);
      return row;
    });
  }

  fitTransform(X: Float64Array[], y: Int32Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}

export class RadiusNeighborsClassifierExt {
  private radius: number;
  private weights: "uniform" | "distance";
  private outlierLabel: number;
  private X_: Float64Array[] | null = null;
  private y_: Int32Array | null = null;
  classes_: Int32Array | null = null;

  constructor(radius = 1.0, weights: "uniform" | "distance" = "uniform", outlierLabel = -1) {
    this.radius = radius;
    this.weights = weights;
    this.outlierLabel = outlierLabel;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    this.X_ = X;
    this.y_ = y;
    this.classes_ = new Int32Array([...new Set(Array.from(y))].sort((a, b) => a - b));
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.X_ || !this.y_) throw new Error("Not fitted");
    return new Int32Array(X.map((xi) => {
      const neighbors: { j: number; dist: number }[] = [];
      for (let j = 0; j < this.X_!.length; j++) {
        let d2 = 0;
        for (let k = 0; k < xi.length; k++) d2 += ((xi[k] ?? 0) - (this.X_![j]![k] ?? 0)) ** 2;
        if (Math.sqrt(d2) <= this.radius) neighbors.push({ j, dist: Math.sqrt(d2) });
      }
      if (neighbors.length === 0) return this.outlierLabel;
      const votes = new Map<number, number>();
      for (const { j, dist } of neighbors) {
        const label = this.y_![j] ?? 0;
        const w = this.weights === "distance" ? 1 / (dist + 1e-10) : 1;
        votes.set(label, (votes.get(label) ?? 0) + w);
      }
      let bestLabel = 0, bestVote = -1;
      for (const [label, vote] of votes) if (vote > bestVote) { bestVote = vote; bestLabel = label; }
      return bestLabel;
    }));
  }

  predictProba(X: Float64Array[]): Float64Array[] {
    if (!this.X_ || !this.y_ || !this.classes_) throw new Error("Not fitted");
    const nClasses = this.classes_.length;
    const classIndex = new Map(Array.from(this.classes_).map((c, i) => [c, i]));
    return X.map((xi) => {
      const proba = new Float64Array(nClasses);
      let total = 0;
      for (let j = 0; j < this.X_!.length; j++) {
        let d2 = 0;
        for (let k = 0; k < xi.length; k++) d2 += ((xi[k] ?? 0) - (this.X_![j]![k] ?? 0)) ** 2;
        if (Math.sqrt(d2) <= this.radius) {
          const label = this.y_![j] ?? 0;
          const ci = classIndex.get(label) ?? 0;
          const w = this.weights === "distance" ? 1 / (Math.sqrt(d2) + 1e-10) : 1;
          proba[ci] += w;
          total += w;
        }
      }
      if (total > 0) for (let c = 0; c < nClasses; c++) proba[c] = (proba[c] ?? 0) / total;
      return proba;
    });
  }
}

export function nearestNeighborsGraph(
  X: Float64Array[],
  nNeighbors = 5,
  mode: "connectivity" | "distance" = "connectivity"
): Float64Array[] {
  const n = X.length;
  const graph: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    const dists = Array.from({ length: n }, (_, j) => {
      if (j === i) return { j, d: Number.POSITIVE_INFINITY };
      let d2 = 0;
      for (let k = 0; k < (X[i]?.length ?? 0); k++) d2 += ((X[i]![k] ?? 0) - (X[j]![k] ?? 0)) ** 2;
      return { j, d: Math.sqrt(d2) };
    }).sort((a, b) => a.d - b.d);
    for (let k = 0; k < Math.min(nNeighbors, n - 1); k++) {
      const neighbor = dists[k]!;
      graph[i]![neighbor.j] = mode === "distance" ? neighbor.d : 1;
    }
  }
  return graph;
}

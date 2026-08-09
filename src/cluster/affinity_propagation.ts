/**
 * AffinityPropagation clustering.
 */

import { NotFittedError } from "../exceptions.js";

export interface AffinityPropagationOptions {
  dampingFactor?: number;
  maxIter?: number;
  convergenceIter?: number;
  preference?: number;
}

export class AffinityPropagation {
  private dampingFactor: number;
  private maxIter: number;
  private convergenceIter: number;
  private preference: number | undefined;

  labels_: Int32Array | null = null;
  clusterCentersIndices_: Int32Array | null = null;
  nIter_ = 0;

  constructor(options: AffinityPropagationOptions = {}) {
    this.dampingFactor = options.dampingFactor ?? 0.5;
    this.maxIter = options.maxIter ?? 200;
    this.convergenceIter = options.convergenceIter ?? 15;
    this.preference = options.preference;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    if (n === 0) {
      this.labels_ = new Int32Array(0);
      this.clusterCentersIndices_ = new Int32Array(0);
      return this;
    }

    // Build similarity matrix S = -||xi - xj||^2
    const S: Float64Array[] = Array.from(
      { length: n },
      () => new Float64Array(n),
    );
    for (let i = 0; i < n; i++) {
      const xi = X[i] ?? new Float64Array(0);
      for (let j = i; j < n; j++) {
        const xj = X[j] ?? new Float64Array(0);
        let d = 0;
        for (let k = 0; k < xi.length; k++)
          d += ((xi[k] ?? 0) - (xj[k] ?? 0)) ** 2;
        (S[i] as Float64Array)[j] = -d;
        (S[j] as Float64Array)[i] = -d;
      }
    }

    // Set preference (diagonal)
    let pref = this.preference;
    if (pref === undefined) {
      // Median of similarities
      const vals: number[] = [];
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++)
          vals.push((S[i] as Float64Array)[j] ?? 0);
      vals.sort((a, b) => a - b);
      pref = vals[Math.floor(vals.length / 2)] ?? -1;
    }
    for (let i = 0; i < n; i++) (S[i] as Float64Array)[i] = pref;

    // Responsibility R and Availability A matrices
    const R: Float64Array[] = Array.from(
      { length: n },
      () => new Float64Array(n),
    );
    const A: Float64Array[] = Array.from(
      { length: n },
      () => new Float64Array(n),
    );
    const d = this.dampingFactor;
    let stableCount = 0;
    let prevExemplars: Set<number> = new Set();

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Update responsibilities: R(i,k) = S(i,k) - max_{k'!=k}[A(i,k')+S(i,k')]
      for (let i = 0; i < n; i++) {
        const Si = S[i] ?? new Float64Array(n);
        const Ai = A[i] ?? new Float64Array(n);
        // Find two highest A+S values
        let max1 = Number.NEGATIVE_INFINITY;
        let max2 = Number.NEGATIVE_INFINITY;
        let argmax1 = -1;
        for (let k = 0; k < n; k++) {
          const v = (Ai[k] ?? 0) + (Si[k] ?? 0);
          if (v > max1) {
            max2 = max1;
            max1 = v;
            argmax1 = k;
          } else if (v > max2) max2 = v;
        }
        const Ri = R[i] ?? new Float64Array(n);
        for (let k = 0; k < n; k++) {
          const maxOther = k === argmax1 ? max2 : max1;
          const newR = (Si[k] ?? 0) - maxOther;
          Ri[k] = d * (Ri[k] ?? 0) + (1 - d) * newR;
        }
      }

      // Update availabilities
      for (let k = 0; k < n; k++) {
        // sum of positive R(i',k) for i'!=k
        let sumPos = 0;
        for (let i = 0; i < n; i++) {
          if (i === k) continue;
          const v = (R[i] as Float64Array)[k] ?? 0;
          if (v > 0) sumPos += v;
        }
        const rkk = (R[k] as Float64Array)[k] ?? 0;
        for (let i = 0; i < n; i++) {
          const Ai = A[i] ?? new Float64Array(n);
          let newA: number;
          if (i === k) {
            newA = sumPos;
          } else {
            const rik = (R[i] as Float64Array)[k] ?? 0;
            const sumWithout = sumPos - (rik > 0 ? rik : 0);
            newA = Math.min(0, rkk + sumWithout);
          }
          Ai[k] = d * (Ai[k] ?? 0) + (1 - d) * newA;
        }
      }

      // Check convergence
      const exemplars = new Set<number>();
      for (let i = 0; i < n; i++) {
        const Ai = A[i] ?? new Float64Array(n);
        const Ri = R[i] ?? new Float64Array(n);
        let best = Number.NEGATIVE_INFINITY;
        let bestK = 0;
        for (let k = 0; k < n; k++) {
          const v = (Ai[k] ?? 0) + (Ri[k] ?? 0);
          if (v > best) {
            best = v;
            bestK = k;
          }
        }
        exemplars.add(bestK);
      }

      const same =
        exemplars.size === prevExemplars.size &&
        [...exemplars].every((e) => prevExemplars.has(e));
      if (same) {
        stableCount++;
        if (stableCount >= this.convergenceIter) {
          this.nIter_ = iter + 1;
          break;
        }
      } else {
        stableCount = 0;
      }
      prevExemplars = exemplars;
      this.nIter_ = iter + 1;
    }

    // Assign labels
    const labels = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const Ai = A[i] ?? new Float64Array(n);
      const Ri = R[i] ?? new Float64Array(n);
      let best = Number.NEGATIVE_INFINITY;
      let bestK = 0;
      for (let k = 0; k < n; k++) {
        const v = (Ai[k] ?? 0) + (Ri[k] ?? 0);
        if (v > best) {
          best = v;
          bestK = k;
        }
      }
      labels[i] = bestK;
    }

    const centerSet = new Set<number>(Array.from(labels));
    const centers = Int32Array.from([...centerSet].sort((a, b) => a - b));
    // Relabel to 0..k-1
    const map = new Map<number, number>();
    centers.forEach((c, idx) => map.set(c, idx));
    for (let i = 0; i < n; i++) labels[i] = map.get(labels[i] ?? 0) ?? 0;

    this.labels_ = labels;
    this.clusterCentersIndices_ = centers;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.labels_ || !this.clusterCentersIndices_)
      throw new NotFittedError("AffinityPropagation");
    // Not supported post-fit without stored data; return empty
    return new Int32Array(X.length).fill(-1);
  }
}

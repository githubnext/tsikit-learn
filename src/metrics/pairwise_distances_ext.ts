/**
 * Extended pairwise distance utilities.
 * Mirrors scikit-learn's metrics.pairwise additional distances.
 */

export type DistanceMetricExt =
  | "euclidean"
  | "manhattan"
  | "chebyshev"
  | "minkowski"
  | "wminkowski"
  | "seuclidean"
  | "mahalanobis"
  | "hamming"
  | "canberra"
  | "braycurtis"
  | "matching"
  | "jaccard"
  | "dice"
  | "kulsinski"
  | "rogerstanimoto"
  | "russellrao"
  | "sokalmichener"
  | "sokalsneath";

export function pairwiseDistancesExt(
  X: Float64Array[],
  Y: Float64Array[] | null = null,
  metric: DistanceMetricExt = "euclidean",
  p = 2,
): Float64Array[] {
  const Yp = Y ?? X;
  return X.map((xi) => Float64Array.from(Yp, (yj) => distanceExt(xi, yj, metric, p)));
}

export function distanceExt(
  u: Float64Array,
  v: Float64Array,
  metric: DistanceMetricExt,
  p = 2,
): number {
  const n = u.length;
  switch (metric) {
    case "euclidean": {
      let s = 0;
      for (let i = 0; i < n; i++) s += ((u[i] ?? 0) - (v[i] ?? 0)) ** 2;
      return Math.sqrt(s);
    }
    case "manhattan": {
      let s = 0;
      for (let i = 0; i < n; i++) s += Math.abs((u[i] ?? 0) - (v[i] ?? 0));
      return s;
    }
    case "chebyshev": {
      let mx = 0;
      for (let i = 0; i < n; i++) mx = Math.max(mx, Math.abs((u[i] ?? 0) - (v[i] ?? 0)));
      return mx;
    }
    case "minkowski": {
      let s = 0;
      for (let i = 0; i < n; i++) s += Math.abs((u[i] ?? 0) - (v[i] ?? 0)) ** p;
      return s ** (1 / p);
    }
    case "hamming": {
      let diff = 0;
      for (let i = 0; i < n; i++) if (u[i] !== v[i]) diff++;
      return diff / n;
    }
    case "canberra": {
      let s = 0;
      for (let i = 0; i < n; i++) {
        const num = Math.abs((u[i] ?? 0) - (v[i] ?? 0));
        const den = Math.abs(u[i] ?? 0) + Math.abs(v[i] ?? 0);
        s += den > 0 ? num / den : 0;
      }
      return s;
    }
    case "braycurtis": {
      let num = 0, den = 0;
      for (let i = 0; i < n; i++) {
        num += Math.abs((u[i] ?? 0) - (v[i] ?? 0));
        den += Math.abs(u[i] ?? 0) + Math.abs(v[i] ?? 0);
      }
      return den > 0 ? num / den : 0;
    }
    case "jaccard": {
      let both = 0, either = 0;
      for (let i = 0; i < n; i++) {
        const ui = u[i] !== 0;
        const vi = v[i] !== 0;
        if (ui && vi) both++;
        if (ui || vi) either++;
      }
      return either === 0 ? 0 : 1 - both / either;
    }
    case "dice": {
      let both = 0, sumU = 0, sumV = 0;
      for (let i = 0; i < n; i++) {
        const ui = u[i] !== 0;
        const vi = v[i] !== 0;
        if (ui && vi) both++;
        if (ui) sumU++;
        if (vi) sumV++;
      }
      return sumU + sumV === 0 ? 0 : 1 - (2 * both) / (sumU + sumV);
    }
    default: {
      let s = 0;
      for (let i = 0; i < n; i++) s += ((u[i] ?? 0) - (v[i] ?? 0)) ** 2;
      return Math.sqrt(s);
    }
  }
}

/**
 * Compute the additive chi2 kernel between sets of vectors.
 */
export function additiveChi2Kernel(
  X: Float64Array[],
  Y: Float64Array[] | null = null,
): Float64Array[] {
  const Yp = Y ?? X;
  return X.map((xi) =>
    Float64Array.from(Yp, (yj) => {
      let s = 0;
      for (let i = 0; i < xi.length; i++) {
        const a = xi[i] ?? 0;
        const b = yj[i] ?? 0;
        if (a + b > 0) s += 2 * a * b / (a + b);
      }
      return s;
    }),
  );
}

/**
 * Compute the chi2 kernel K(x,y) = exp(-gamma * sum((x_i - y_i)^2 / (x_i + y_i))).
 */
export function chi2Kernel(
  X: Float64Array[],
  Y: Float64Array[] | null = null,
  gamma = 1,
): Float64Array[] {
  const K = additiveChi2Kernel(X, Y);
  // chi2 kernel: exp(-gamma * additive_chi2_distance) = exp(-gamma * (K_max - k))
  // Actually: chi2_kernel = exp(-gamma * chi2_distance) where chi2_dist = sum((xi-yi)^2/(xi+yi))
  // K_additive = sum(2*xi*yi/(xi+yi)) so chi2_dist = n - K_additive when norms=1
  // Use the formula directly
  const Yp = Y ?? X;
  return X.map((xi, i) =>
    Float64Array.from(Yp, (yj, j) => {
      let s = 0;
      for (let k = 0; k < xi.length; k++) {
        const a = xi[k] ?? 0;
        const b = yj[k] ?? 0;
        if (a + b > 0) s += (a - b) ** 2 / (a + b);
      }
      return Math.exp(-gamma * s);
    }),
  );
}

/**
 * Additional pairwise kernel functions.
 * Mirrors sklearn.metrics.pairwise (laplacian, sigmoid, chi2, etc.)
 */

/**
 * Laplacian kernel: K(x, y) = exp(-gamma * ||x - y||_1)
 */
export function laplacianKernel(
  X: Float64Array[],
  Y?: Float64Array[],
  gamma = 1.0,
): Float64Array[] {
  const Ymat = Y ?? X;
  const n = X.length;
  const m = Ymat.length;
  const K: Float64Array[] = Array.from({ length: n }, () => new Float64Array(m));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let dist = 0;
      const xi = X[i]!;
      const yj = Ymat[j]!;
      for (let k = 0; k < xi.length; k++) {
        dist += Math.abs((xi[k] ?? 0) - (yj[k] ?? 0));
      }
      K[i]![j] = Math.exp(-gamma * dist);
    }
  }
  return K;
}

/**
 * Sigmoid kernel: K(x, y) = tanh(gamma * <x, y> + coef0)
 */
export function sigmoidKernel(
  X: Float64Array[],
  Y?: Float64Array[],
  gamma = 1.0,
  coef0 = 0.0,
): Float64Array[] {
  const Ymat = Y ?? X;
  const n = X.length;
  const m = Ymat.length;
  const K: Float64Array[] = Array.from({ length: n }, () => new Float64Array(m));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let dot = 0;
      const xi = X[i]!;
      const yj = Ymat[j]!;
      for (let k = 0; k < xi.length; k++) {
        dot += (xi[k] ?? 0) * (yj[k] ?? 0);
      }
      K[i]![j] = Math.tanh(gamma * dot + coef0);
    }
  }
  return K;
}

/**
 * Additive chi-squared kernel: K(x, y) = sum_k 2*x_k*y_k / (x_k + y_k)
 * A positive semidefinite kernel for histograms.
 */
export function additiveChi2Kernel(
  X: Float64Array[],
  Y?: Float64Array[],
): Float64Array[] {
  const Ymat = Y ?? X;
  const n = X.length;
  const m = Ymat.length;
  const K: Float64Array[] = Array.from({ length: n }, () => new Float64Array(m));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let k_val = 0;
      const xi = X[i]!;
      const yj = Ymat[j]!;
      for (let k = 0; k < xi.length; k++) {
        const xk = xi[k] ?? 0;
        const yk = yj[k] ?? 0;
        const denom = xk + yk;
        if (denom > 0) k_val += 2 * xk * yk / denom;
      }
      K[i]![j] = k_val;
    }
  }
  return K;
}

/**
 * Exponentiated chi-squared kernel: K(x, y) = exp(-gamma * sum_k (x_k-y_k)^2 / (x_k+y_k))
 */
export function chi2Kernel(
  X: Float64Array[],
  Y?: Float64Array[],
  gamma = 1.0,
): Float64Array[] {
  const Ymat = Y ?? X;
  const n = X.length;
  const m = Ymat.length;
  const K: Float64Array[] = Array.from({ length: n }, () => new Float64Array(m));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let chi2 = 0;
      const xi = X[i]!;
      const yj = Ymat[j]!;
      for (let k = 0; k < xi.length; k++) {
        const xk = xi[k] ?? 0;
        const yk = yj[k] ?? 0;
        const denom = xk + yk;
        if (denom > 0) chi2 += (xk - yk) ** 2 / denom;
      }
      K[i]![j] = Math.exp(-gamma * chi2);
    }
  }
  return K;
}

/**
 * Compute the kernel matrix from a set of named kernels.
 */
export type KernelName =
  | "rbf"
  | "linear"
  | "polynomial"
  | "poly"
  | "laplacian"
  | "sigmoid"
  | "chi2"
  | "additive_chi2"
  | "cosine";

export interface PairwiseKernelOptions {
  gamma?: number;
  coef0?: number;
  degree?: number;
}

/**
 * Compute the kernel matrix between X and Y using the specified kernel.
 */
export function pairwiseKernels(
  X: Float64Array[],
  Y: Float64Array[] | undefined,
  metric: KernelName,
  options: PairwiseKernelOptions = {},
): Float64Array[] {
  const { gamma = 1.0, coef0 = 1.0, degree = 3 } = options;
  switch (metric) {
    case "laplacian":
      return laplacianKernel(X, Y, gamma);
    case "sigmoid":
      return sigmoidKernel(X, Y, gamma, coef0);
    case "chi2":
      return chi2Kernel(X, Y, gamma);
    case "additive_chi2":
      return additiveChi2Kernel(X, Y);
    case "linear":
      return linearKernelLocal(X, Y);
    case "polynomial":
    case "poly":
      return polynomialKernelLocal(X, Y, degree, gamma, coef0);
    case "rbf":
      return rbfKernelLocal(X, Y, gamma);
    case "cosine":
      return cosineKernelLocal(X, Y);
    default:
      throw new Error(`Unknown kernel: ${metric as string}`);
  }
}

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let k = 0; k < a.length; k++) s += (a[k] ?? 0) * (b[k] ?? 0);
  return s;
}

function linearKernelLocal(X: Float64Array[], Y?: Float64Array[]): Float64Array[] {
  const Ymat = Y ?? X;
  return X.map(xi => {
    const row = new Float64Array(Ymat.length);
    for (let j = 0; j < Ymat.length; j++) row[j] = dot(xi, Ymat[j]!);
    return row;
  });
}

function polynomialKernelLocal(
  X: Float64Array[],
  Y?: Float64Array[],
  degree = 3,
  gamma = 1.0,
  coef0 = 1.0,
): Float64Array[] {
  const Ymat = Y ?? X;
  return X.map(xi => {
    const row = new Float64Array(Ymat.length);
    for (let j = 0; j < Ymat.length; j++) {
      row[j] = (gamma * dot(xi, Ymat[j]!) + coef0) ** degree;
    }
    return row;
  });
}

function rbfKernelLocal(
  X: Float64Array[],
  Y?: Float64Array[],
  gamma = 1.0,
): Float64Array[] {
  const Ymat = Y ?? X;
  return X.map(xi => {
    const row = new Float64Array(Ymat.length);
    for (let j = 0; j < Ymat.length; j++) {
      let d2 = 0;
      const yj = Ymat[j]!;
      for (let k = 0; k < xi.length; k++) d2 += ((xi[k] ?? 0) - (yj[k] ?? 0)) ** 2;
      row[j] = Math.exp(-gamma * d2);
    }
    return row;
  });
}

function cosineKernelLocal(
  X: Float64Array[],
  Y?: Float64Array[],
): Float64Array[] {
  const Ymat = Y ?? X;
  const normX = X.map(xi => Math.sqrt(dot(xi, xi)) || 1e-12);
  const normY = Ymat.map(yi => Math.sqrt(dot(yi, yi)) || 1e-12);
  return X.map((xi, i) => {
    const row = new Float64Array(Ymat.length);
    for (let j = 0; j < Ymat.length; j++) {
      row[j] = dot(xi, Ymat[j]!) / (normX[i]! * normY[j]!);
    }
    return row;
  });
}

/**
 * Compute the Euclidean distance matrix between rows of X and Y.
 */
export function pairwiseEuclideanDistances(
  X: Float64Array[],
  Y?: Float64Array[],
): Float64Array[] {
  const Ymat = Y ?? X;
  return X.map(xi => {
    const row = new Float64Array(Ymat.length);
    for (let j = 0; j < Ymat.length; j++) {
      let d2 = 0;
      const yj = Ymat[j]!;
      for (let k = 0; k < xi.length; k++) d2 += ((xi[k] ?? 0) - (yj[k] ?? 0)) ** 2;
      row[j] = Math.sqrt(d2);
    }
    return row;
  });
}

/**
 * Compute the Haversine distance between lat/lon pairs (in radians).
 * Useful for geographic data. Named haversineKernel to avoid conflicts.
 */
export function haversineKernel(
  X: Float64Array[],
  Y?: Float64Array[],
): Float64Array[] {
  const Ymat = Y ?? X;
  return X.map(xi => {
    const row = new Float64Array(Ymat.length);
    for (let j = 0; j < Ymat.length; j++) {
      const yj = Ymat[j]!;
      const lat1 = xi[0] ?? 0, lon1 = xi[1] ?? 0;
      const lat2 = yj[0] ?? 0, lon2 = yj[1] ?? 0;
      const dlat = lat2 - lat1;
      const dlon = lon2 - lon1;
      const a = Math.sin(dlat / 2) ** 2
        + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dlon / 2) ** 2;
      row[j] = 2 * Math.asin(Math.sqrt(a));
    }
    return row;
  });
}

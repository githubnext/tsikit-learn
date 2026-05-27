/**
 * Additional covariance estimators: OAS, LedoitWolfExt.
 * Mirrors sklearn.covariance extras.
 */

import { NotFittedError } from "../exceptions.js";

function computeSampleCov(X: Float64Array[]): {
  mean: Float64Array;
  cov: Float64Array[];
} {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  const mean = new Float64Array(p);
  for (const row of X) {
    for (let j = 0; j < p; j++) mean[j] = (mean[j] ?? 0) + (row[j] ?? 0);
  }
  for (let j = 0; j < p; j++) mean[j] = (mean[j] ?? 0) / n;

  const cov: Float64Array[] = Array.from({ length: p }, () => new Float64Array(p));
  for (const row of X) {
    for (let i = 0; i < p; i++) {
      for (let j = i; j < p; j++) {
        const v = ((row[i] ?? 0) - (mean[i] ?? 0)) * ((row[j] ?? 0) - (mean[j] ?? 0));
        cov[i]![j] = (cov[i]?.[j] ?? 0) + v;
        if (i !== j) cov[j]![i] = (cov[j]?.[i] ?? 0) + v;
      }
    }
  }
  const denom = n - 1 > 0 ? n - 1 : 1;
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) cov[i]![j] = (cov[i]?.[j] ?? 0) / denom;
  }
  return { mean, cov };
}

export class OAS {
  location_: Float64Array | null = null;
  covariance_: Float64Array[] | null = null;
  precision_: Float64Array[] | null = null;
  shrinkage_: number = 0;

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const { mean, cov } = computeSampleCov(X);
    this.location_ = mean;

    // OAS shrinkage coefficient
    let traceSq = 0;
    let traceSquared = 0;
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        traceSq += (cov[i]?.[j] ?? 0) ** 2;
      }
      traceSquared += (cov[i]?.[i] ?? 0);
    }
    traceSquared = traceSquared ** 2;

    const num = (1 - 2 / p) * traceSq + traceSquared;
    const denom2 = (n + 1 - 2 / p) * (traceSq - traceSquared / p);
    this.shrinkage_ = denom2 > 0 ? Math.min(1, num / denom2) : 1;

    const rho = this.shrinkage_;
    let traceS = 0;
    for (let i = 0; i < p; i++) traceS += cov[i]?.[i] ?? 0;
    const mu = traceS / p;

    this.covariance_ = Array.from({ length: p }, (_, i) =>
      Float64Array.from({ length: p }, (_, j) =>
        (1 - rho) * (cov[i]?.[j] ?? 0) + (i === j ? rho * mu : 0),
      ),
    );

    return this;
  }
}

export class LedoitWolfExt {
  location_: Float64Array | null = null;
  covariance_: Float64Array[] | null = null;
  shrinkage_: number = 0;

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const { mean, cov } = computeSampleCov(X);
    this.location_ = mean;

    // Ledoit-Wolf analytical shrinkage
    let mu = 0;
    for (let i = 0; i < p; i++) mu += cov[i]?.[i] ?? 0;
    mu /= p;

    let delta2 = 0;
    let beta2 = 0;
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        const Sij = cov[i]?.[j] ?? 0;
        const Fij = i === j ? mu : 0;
        delta2 += (Sij - Fij) ** 2;
      }
    }

    // Estimate beta
    for (const row of X) {
      const centered = new Float64Array(p);
      for (let j = 0; j < p; j++) centered[j] = (row[j] ?? 0) - (mean[j] ?? 0);
      for (let i = 0; i < p; i++) {
        for (let j = 0; j < p; j++) {
          const Xij = (centered[i] ?? 0) * (centered[j] ?? 0);
          const Sij = cov[i]?.[j] ?? 0;
          beta2 += (Xij - Sij) ** 2;
        }
      }
    }
    beta2 /= n ** 2;

    const rho = Math.min(1, beta2 / delta2);
    this.shrinkage_ = rho;

    this.covariance_ = Array.from({ length: p }, (_, i) =>
      Float64Array.from({ length: p }, (_, j) =>
        (1 - rho) * (cov[i]?.[j] ?? 0) + (i === j ? rho * mu : 0),
      ),
    );

    return this;
  }
}

export class ShrunkCovariance {
  shrinkage: number;
  location_: Float64Array | null = null;
  covariance_: Float64Array[] | null = null;

  constructor(shrinkage = 0.1) {
    this.shrinkage = shrinkage;
  }

  fit(X: Float64Array[]): this {
    const p = X[0]?.length ?? 0;
    const { mean, cov } = computeSampleCov(X);
    this.location_ = mean;

    let mu = 0;
    for (let i = 0; i < p; i++) mu += cov[i]?.[i] ?? 0;
    mu /= p;

    const rho = this.shrinkage;
    this.covariance_ = Array.from({ length: p }, (_, i) =>
      Float64Array.from({ length: p }, (_, j) =>
        (1 - rho) * (cov[i]?.[j] ?? 0) + (i === j ? rho * mu : 0),
      ),
    );
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.covariance_) throw new NotFittedError("ShrunkCovariance is not fitted");
    return X;
  }
}

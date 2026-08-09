/**
 * Empirical covariance estimators.
 * Mirrors scikit-learn's covariance.EmpiricalCovariance, LedoitWolf, OAS.
 */

function mean(
  X: Float64Array[],
  nSamples: number,
  nFeatures: number,
): Float64Array {
  const m = new Float64Array(nFeatures);
  for (const row of X) {
    for (let j = 0; j < nFeatures; j++)
      m[j] = (m[j] ?? 0) + (row[j] ?? 0) / nSamples;
  }
  return m;
}

function covMatrix(
  X: Float64Array[],
  mu: Float64Array,
  nSamples: number,
  nFeatures: number,
): Float64Array[] {
  const C: Float64Array[] = Array.from(
    { length: nFeatures },
    () => new Float64Array(nFeatures),
  );
  for (const row of X) {
    for (let i = 0; i < nFeatures; i++) {
      for (let j = 0; j < nFeatures; j++) {
        C[i]![j] =
          (C[i]![j] ?? 0) +
          (((row[i] ?? 0) - (mu[i] ?? 0)) * ((row[j] ?? 0) - (mu[j] ?? 0))) /
            nSamples;
      }
    }
  }
  return C;
}

export class EmpiricalCovariance {
  location_: Float64Array | null = null;
  covariance_: Float64Array[] | null = null;

  constructor(readonly assumeCentered = false) {}

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const mu = this.assumeCentered ? new Float64Array(p) : mean(X, n, p);
    this.location_ = mu;
    this.covariance_ = covMatrix(X, mu, n, p);
    return this;
  }

  mahalanobis(X: Float64Array[]): Float64Array {
    if (this.covariance_ === null || this.location_ === null) {
      throw new Error("EmpiricalCovariance must be fitted first");
    }
    // Simplified: diagonal approximation
    const diagInv = this.covariance_.map((row, i) => row[i] ?? 1);
    return Float64Array.from(X, (xi) => {
      let s = 0;
      for (let j = 0; j < xi.length; j++) {
        const diff = (xi[j] ?? 0) - (this.location_![j] ?? 0);
        s += (diff * diff) / (diagInv[j] ?? 1);
      }
      return Math.sqrt(s);
    });
  }

  score(XTest: Float64Array[], yTest?: unknown): number {
    void yTest;
    if (this.covariance_ === null) throw new Error("Not fitted");
    const n = XTest.length;
    const p = XTest[0]?.length ?? 0;
    const mu = mean(XTest, n, p);
    const testCov = covMatrix(XTest, mu, n, p);
    let s = 0;
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        const diff = (testCov[i]?.[j] ?? 0) - (this.covariance_[i]?.[j] ?? 0);
        s += diff * diff;
      }
    }
    return -Math.sqrt(s);
  }
}

/**
 * Ledoit-Wolf covariance estimator with analytic shrinkage.
 */
export class LedoitWolf extends EmpiricalCovariance {
  shrinkage_: number = 0;

  override fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const mu = mean(X, n, p);
    this.location_ = mu;
    const S = covMatrix(X, mu, n, p);

    // Ledoit-Wolf analytical formula
    let trS = 0;
    let trS2 = 0;
    let tr2S = 0;
    for (let i = 0; i < p; i++) {
      trS += S[i]?.[i] ?? 0;
      for (let j = 0; j < p; j++) trS2 += (S[i]?.[j] ?? 0) ** 2;
    }
    tr2S = trS * trS;

    // Oracle approximating shrinkage
    const mu1 = trS / p;
    const delta2 = (trS2 - tr2S / p) / p;
    const beta2 = Math.max(
      0,
      (trS2 / n - tr2S / (n * p)) / (trS2 - tr2S / p + 1e-10),
    );
    const shrinkage = Math.min(1, beta2);
    this.shrinkage_ = shrinkage;

    this.covariance_ = S.map((row, i) =>
      Float64Array.from(
        row,
        (v, j) => (1 - shrinkage) * v + (i === j ? shrinkage * mu1 : 0),
      ),
    );
    void delta2;
    return this;
  }
}

/**
 * Oracle Approximating Shrinkage (OAS) estimator.
 */
export class OAS extends EmpiricalCovariance {
  shrinkage_: number = 0;

  override fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const mu = mean(X, n, p);
    this.location_ = mu;
    const S = covMatrix(X, mu, n, p);

    let trS = 0;
    let trS2 = 0;
    for (let i = 0; i < p; i++) {
      trS += S[i]?.[i] ?? 0;
      for (let j = 0; j < p; j++) trS2 += (S[i]?.[j] ?? 0) ** 2;
    }

    // OAS formula
    const rho = (1 - 2 / p) * trS2 + trS * trS;
    const gamma = (n + 1 - 2 / p) * (trS2 - (trS * trS) / p);
    const shrinkage = Math.min(1, rho / (gamma + 1e-10));
    this.shrinkage_ = shrinkage;
    const mu1 = trS / p;

    this.covariance_ = S.map((row, i) =>
      Float64Array.from(
        row,
        (v, j) => (1 - shrinkage) * v + (i === j ? shrinkage * mu1 : 0),
      ),
    );
    return this;
  }
}

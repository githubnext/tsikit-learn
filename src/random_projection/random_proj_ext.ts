/**
 * Extended random projection utilities: SparseRandomProjection,
 * GaussianRandomProjection, and Johnson-Lindenstrauss lemma helpers.
 */

/** Johnson-Lindenstrauss minimum dimensions estimate. */
export function johnsonLindenstraussMinDim(
  nSamples: number,
  eps: number,
): number {
  const numerator = 4 * Math.log(nSamples);
  const denominator = eps ** 2 / 2 - eps ** 3 / 3;
  return Math.ceil(numerator / denominator);
}

/** Gaussian Random Projection. */
export class GaussianRandomProjection {
  nComponents: number | "auto";
  eps: number;
  components_?: Float64Array[];
  nFeaturesIn_?: number;
  nComponentsActual_?: number;

  constructor(nComponents: number | "auto" = "auto", eps = 0.1) {
    this.nComponents = nComponents;
    this.eps = eps;
  }

  fit(X: Float64Array[]): this {
    const d = X[0]?.length ?? 0;
    const n = X.length;
    this.nFeaturesIn_ = d;
    this.nComponentsActual_ =
      this.nComponents === "auto"
        ? Math.min(johnsonLindenstraussMinDim(n, this.eps), d)
        : (this.nComponents as number);

    const k = this.nComponentsActual_;
    const std = 1 / Math.sqrt(k);
    this.components_ = Array.from({ length: k }, () =>
      new Float64Array(d).map(() => gaussNormal(0, std))
    );
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_) throw new Error("Not fitted");
    const k = this.components_.length;
    return X.map((xi) =>
      new Float64Array(k).map((_, j) => {
        const comp = this.components_![j];
        if (comp === undefined) return 0;
        let dot = 0;
        for (let l = 0; l < xi.length; l++) dot += (xi[l] ?? 0) * (comp[l] ?? 0);
        return dot;
      })
    );
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

function gaussNormal(mean: number, std: number): number {
  const u1 = Math.random(), u2 = Math.random();
  return mean + std * Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
}

/** Sparse Random Projection (±1/sqrt(s) entries). */
export class SparseRandomProjection {
  nComponents: number | "auto";
  density: number | "auto";
  eps: number;
  components_?: Float64Array[];
  nFeaturesIn_?: number;
  nComponentsActual_?: number;

  constructor(nComponents: number | "auto" = "auto", density: number | "auto" = "auto", eps = 0.1) {
    this.nComponents = nComponents;
    this.density = density;
    this.eps = eps;
  }

  fit(X: Float64Array[]): this {
    const d = X[0]?.length ?? 0;
    const n = X.length;
    this.nFeaturesIn_ = d;
    this.nComponentsActual_ =
      this.nComponents === "auto"
        ? Math.min(johnsonLindenstraussMinDim(n, this.eps), d)
        : (this.nComponents as number);

    const dens = this.density === "auto" ? 1 / Math.sqrt(d) : (this.density as number);
    const k = this.nComponentsActual_;
    const scale = 1 / Math.sqrt(dens * k);
    this.components_ = Array.from({ length: k }, () =>
      new Float64Array(d).map(() => {
        const r = Math.random();
        if (r < dens / 2) return -scale;
        if (r < dens) return scale;
        return 0;
      })
    );
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_) throw new Error("Not fitted");
    const k = this.components_.length;
    return X.map((xi) =>
      new Float64Array(k).map((_, j) => {
        const comp = this.components_![j];
        if (comp === undefined) return 0;
        let dot = 0;
        for (let l = 0; l < xi.length; l++) dot += (xi[l] ?? 0) * (comp[l] ?? 0);
        return dot;
      })
    );
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

/**
 * Extended PLS utilities: PLSSVDExt.
 * Mirrors sklearn.cross_decomposition.PLSSVD.
 */

export interface PLSSVDOptions {
  nComponents?: number;
  scale?: boolean;
  copyData?: boolean;
}

/**
 * Partial Least Squares SVD.
 * Finds the directions of maximum covariance between X and Y.
 */
export class PLSSVDExt {
  nComponents: number;
  scale: boolean;

  xWeights_: Float64Array[] | null = null;
  yWeights_: Float64Array[] | null = null;
  xScores_: Float64Array[] | null = null;
  yScores_: Float64Array[] | null = null;
  xMean_: Float64Array | null = null;
  yMean_: Float64Array | null = null;
  xStd_: Float64Array | null = null;
  yStd_: Float64Array | null = null;
  nFeaturesFit_: number = 0;
  nTargetsFit_: number = 0;

  constructor(options: PLSSVDOptions = {}) {
    this.nComponents = options.nComponents ?? 2;
    this.scale = options.scale ?? true;
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const nSamples = X.length;
    const nFeatures = X[0]?.length ?? 0;
    const nTargets = Y[0]?.length ?? 0;
    this.nFeaturesFit_ = nFeatures;
    this.nTargetsFit_ = nTargets;

    // Center (and optionally scale)
    this.xMean_ = new Float64Array(nFeatures);
    this.yMean_ = new Float64Array(nTargets);
    for (const row of X)
      for (let j = 0; j < nFeatures; j++)
        this.xMean_[j] = (this.xMean_[j] ?? 0) + (row[j] ?? 0);
    for (const row of Y)
      for (let j = 0; j < nTargets; j++)
        this.yMean_[j] = (this.yMean_[j] ?? 0) + (row[j] ?? 0);
    for (let j = 0; j < nFeatures; j++)
      this.xMean_[j] = (this.xMean_[j] ?? 0) / nSamples;
    for (let j = 0; j < nTargets; j++)
      this.yMean_[j] = (this.yMean_[j] ?? 0) / nSamples;

    this.xStd_ = new Float64Array(nFeatures).fill(1);
    this.yStd_ = new Float64Array(nTargets).fill(1);
    if (this.scale) {
      for (const row of X)
        for (let j = 0; j < nFeatures; j++) {
          this.xStd_[j] =
            (this.xStd_[j] ?? 0) + ((row[j] ?? 0) - (this.xMean_[j] ?? 0)) ** 2;
        }
      for (let j = 0; j < nFeatures; j++)
        this.xStd_[j] = Math.sqrt((this.xStd_[j] ?? 0) / (nSamples - 1)) || 1;
      for (const row of Y)
        for (let j = 0; j < nTargets; j++) {
          this.yStd_[j] =
            (this.yStd_[j] ?? 0) + ((row[j] ?? 0) - (this.yMean_[j] ?? 0)) ** 2;
        }
      for (let j = 0; j < nTargets; j++)
        this.yStd_[j] = Math.sqrt((this.yStd_[j] ?? 0) / (nSamples - 1)) || 1;
    }

    // Center and scale X, Y
    const Xc = X.map((row) =>
      new Float64Array(nFeatures).map(
        (_, j) =>
          ((row[j] ?? 0) - (this.xMean_![j] ?? 0)) / (this.xStd_![j] ?? 1),
      ),
    );
    const Yc = Y.map((row) =>
      new Float64Array(nTargets).map(
        (_, j) =>
          ((row[j] ?? 0) - (this.yMean_![j] ?? 0)) / (this.yStd_![j] ?? 1),
      ),
    );

    // Compute cross-covariance matrix C = X^T Y
    const C: Float64Array[] = Array.from(
      { length: nFeatures },
      () => new Float64Array(nTargets),
    );
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        for (let k = 0; k < nTargets; k++) {
          C[j]![k] = (C[j]![k] ?? 0) + (Xc[i]?.[j] ?? 0) * (Yc[i]?.[k] ?? 0);
        }
      }
    }

    const k = Math.min(this.nComponents, nFeatures, nTargets);

    // SVD via power iteration
    const xWeights: Float64Array[] = [];
    const yWeights: Float64Array[] = [];

    let seed = 42;
    function rand(): number {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return ((seed >>> 0) / 0xffffffff) * 2 - 1;
    }

    for (let comp = 0; comp < k; comp++) {
      let u = new Float64Array(nFeatures).map(() => rand());
      let normU = Math.sqrt(u.reduce((s, v) => s + v ** 2, 0)) || 1;
      for (let j = 0; j < nFeatures; j++) u[j] = (u[j] ?? 0) / normU;

      for (let iter = 0; iter < 10; iter++) {
        // v = C^T u
        const v = new Float64Array(nTargets);
        for (let j = 0; j < nFeatures; j++)
          for (let l = 0; l < nTargets; l++)
            v[l] += (C[j]?.[l] ?? 0) * (u[j] ?? 0);
        const normV = Math.sqrt(v.reduce((s, v2) => s + v2 ** 2, 0)) || 1;
        for (let l = 0; l < nTargets; l++) v[l] = (v[l] ?? 0) / normV;

        // u = C v
        const uNew = new Float64Array(nFeatures);
        for (let j = 0; j < nFeatures; j++)
          for (let l = 0; l < nTargets; l++)
            uNew[j] += (C[j]?.[l] ?? 0) * (v[l] ?? 0);

        // Orthogonalize against previous
        for (const pu of xWeights) {
          let dot = 0;
          for (let j = 0; j < nFeatures; j++)
            dot += (uNew[j] ?? 0) * (pu[j] ?? 0);
          for (let j = 0; j < nFeatures; j++)
            uNew[j] = (uNew[j] ?? 0) - dot * (pu[j] ?? 0);
        }

        normU = Math.sqrt(uNew.reduce((s, v2) => s + v2 ** 2, 0)) || 1;
        u = new Float64Array(uNew.map((v2) => v2 / normU));
      }

      // Final v
      const v = new Float64Array(nTargets);
      for (let j = 0; j < nFeatures; j++)
        for (let l = 0; l < nTargets; l++)
          v[l] += (C[j]?.[l] ?? 0) * (u[j] ?? 0);
      const normV = Math.sqrt(v.reduce((s, v2) => s + v2 ** 2, 0)) || 1;
      for (let l = 0; l < nTargets; l++) v[l] = (v[l] ?? 0) / normV;

      xWeights.push(u);
      yWeights.push(v);
    }

    this.xWeights_ = xWeights;
    this.yWeights_ = yWeights;

    // Compute scores
    this.xScores_ = Xc.map(
      (row) =>
        new Float64Array(
          xWeights.map((w) => {
            let dot = 0;
            for (let j = 0; j < nFeatures; j++)
              dot += (row[j] ?? 0) * (w[j] ?? 0);
            return dot;
          }),
        ),
    );
    this.yScores_ = Yc.map(
      (row) =>
        new Float64Array(
          yWeights.map((w) => {
            let dot = 0;
            for (let j = 0; j < nTargets; j++)
              dot += (row[j] ?? 0) * (w[j] ?? 0);
            return dot;
          }),
        ),
    );

    return this;
  }

  transform(
    X: Float64Array[],
    Y?: Float64Array[],
  ): { xScores: Float64Array[]; yScores?: Float64Array[] } {
    if (!this.xWeights_ || !this.xMean_)
      throw new Error("PLSSVDExt not fitted");
    const nFeatures = this.nFeaturesFit_;
    const xScores = X.map(
      (row) =>
        new Float64Array(
          this.xWeights_!.map((w) => {
            let dot = 0;
            for (let j = 0; j < nFeatures; j++)
              dot +=
                (((row[j] ?? 0) - (this.xMean_![j] ?? 0)) /
                  (this.xStd_![j] ?? 1)) *
                (w[j] ?? 0);
            return dot;
          }),
        ),
    );

    if (Y) {
      const nTargets = this.nTargetsFit_;
      const yScores = Y.map(
        (row) =>
          new Float64Array(
            this.yWeights_!.map((w) => {
              let dot = 0;
              for (let j = 0; j < nTargets; j++)
                dot +=
                  (((row[j] ?? 0) - (this.yMean_![j] ?? 0)) /
                    (this.yStd_![j] ?? 1)) *
                  (w[j] ?? 0);
              return dot;
            }),
          ),
      );
      return { xScores, yScores };
    }
    return { xScores };
  }

  fitTransform(
    X: Float64Array[],
    Y: Float64Array[],
  ): { xScores: Float64Array[]; yScores: Float64Array[] } {
    this.fit(X, Y);
    return { xScores: this.xScores_!, yScores: this.yScores_! };
  }
}

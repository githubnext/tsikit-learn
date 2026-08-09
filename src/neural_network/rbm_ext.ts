/**
 * Restricted Boltzmann Machine extensions.
 * Mirrors scikit-learn's neural_network.BernoulliRBM with additional features.
 */

export interface RBMExtOptions {
  nComponents?: number;
  learningRate?: number;
  batchSize?: number;
  nIter?: number;
  randomState?: number;
  verbose?: boolean;
  cdSteps?: number;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export class BernoulliRBMExt {
  readonly nComponents: number;
  readonly learningRate: number;
  readonly batchSize: number;
  readonly nIter: number;
  readonly randomState: number;
  readonly cdSteps: number;

  components_: Float64Array[] | null = null; // W: nComponents x nFeatures
  hiddenBias_: Float64Array | null = null;   // c: nComponents
  visibleBias_: Float64Array | null = null;  // b: nFeatures

  pseudoLikelihood_: number[] = [];

  constructor(options: RBMExtOptions = {}) {
    this.nComponents = options.nComponents ?? 256;
    this.learningRate = options.learningRate ?? 0.1;
    this.batchSize = options.batchSize ?? 10;
    this.nIter = options.nIter ?? 10;
    this.randomState = options.randomState ?? 0;
    this.cdSteps = options.cdSteps ?? 1;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const nFeatures = X[0]?.length ?? 0;
    const H = this.nComponents;

    let seed = this.randomState;
    const rng = (): number => {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return (seed >>> 0) / 0xffffffff;
    };
    const randn = (): number => {
      // Box-Muller
      const u = rng();
      const v = rng();
      return Math.sqrt(-2 * Math.log(u + 1e-10)) * Math.cos(2 * Math.PI * v);
    };

    // Initialize weights
    const W: Float64Array[] = Array.from({ length: H }, () =>
      Float64Array.from({ length: nFeatures }, () => randn() * 0.01),
    );
    const hidBias = new Float64Array(H);
    const visBias = new Float64Array(nFeatures);

    // Pre-set visible bias to log(p/(1-p)) of data mean
    for (let j = 0; j < nFeatures; j++) {
      const p = X.reduce((s, xi) => s + (xi[j] ?? 0), 0) / n;
      const q = Math.max(0.01, Math.min(0.99, p));
      visBias[j] = Math.log(q / (1 - q));
    }

    for (let iter = 0; iter < this.nIter; iter++) {
      // Shuffle
      const perm = Array.from({ length: n }, (_, i) => i).sort(() => rng() - 0.5);
      for (let b = 0; b < n; b += this.batchSize) {
        const batch = perm.slice(b, b + this.batchSize).map((i) => X[i]!);
        const bSize = batch.length;

        // Positive phase: v -> h
        const posHidProb = batch.map((v) =>
          Float64Array.from({ length: H }, (_, j) => {
            let act = hidBias[j] ?? 0;
            for (let k = 0; k < nFeatures; k++) act += (W[j]?.[k] ?? 0) * (v[k] ?? 0);
            return sigmoid(act);
          }),
        );

        // Sample hidden states
        const posHidSample = posHidProb.map((ph) =>
          Float64Array.from(ph, (p) => (rng() < p ? 1 : 0)),
        );

        // CD-k: negative phase
        let negVisProb = batch;
        let negHidProb: Float64Array[] = posHidSample;

        for (let step = 0; step < this.cdSteps; step++) {
          negHidProb = negVisProb.map((v) =>
            Float64Array.from({ length: H }, (_, j) => {
              let act = hidBias[j] ?? 0;
              for (let k = 0; k < nFeatures; k++) act += (W[j]?.[k] ?? 0) * (v[k] ?? 0);
              return sigmoid(act);
            }),
          );
          negVisProb = negHidProb.map((h) =>
            Float64Array.from({ length: nFeatures }, (_, k) => {
              let act = visBias[k] ?? 0;
              for (let j = 0; j < H; j++) act += (W[j]?.[k] ?? 0) * (h[j] ?? 0);
              return sigmoid(act);
            }),
          );
        }

        // Update weights
        const lr = this.learningRate / bSize;
        for (let j = 0; j < H; j++) {
          for (let k = 0; k < nFeatures; k++) {
            let posGrad = 0, negGrad = 0;
            for (let i = 0; i < bSize; i++) {
              posGrad += (posHidProb[i]?.[j] ?? 0) * (batch[i]?.[k] ?? 0);
              negGrad += (negHidProb[i]?.[j] ?? 0) * (negVisProb[i]?.[k] ?? 0);
            }
            W[j]![k] = (W[j]![k] ?? 0) + lr * (posGrad - negGrad);
          }
          let dHid = 0;
          for (let i = 0; i < bSize; i++) {
            dHid += (posHidProb[i]?.[j] ?? 0) - (negHidProb[i]?.[j] ?? 0);
          }
          hidBias[j] = (hidBias[j] ?? 0) + lr * dHid;
        }
        for (let k = 0; k < nFeatures; k++) {
          let dVis = 0;
          for (let i = 0; i < bSize; i++) {
            dVis += (batch[i]?.[k] ?? 0) - (negVisProb[i]?.[k] ?? 0);
          }
          visBias[k] = (visBias[k] ?? 0) + lr * dVis;
        }
      }

      // Compute pseudo-likelihood
      const pl = this._pseudoLikelihood(X.slice(0, Math.min(100, n)), W, hidBias, visBias, nFeatures, H);
      this.pseudoLikelihood_.push(pl);
    }

    this.components_ = W;
    this.hiddenBias_ = hidBias;
    this.visibleBias_ = visBias;
    return this;
  }

  private _pseudoLikelihood(
    X: Float64Array[],
    W: Float64Array[],
    hidBias: Float64Array,
    visBias: Float64Array,
    nFeatures: number,
    H: number,
  ): number {
    let pl = 0;
    for (const xi of X) {
      const freeEnergy = (v: Float64Array): number => {
        let fe = 0;
        for (let k = 0; k < nFeatures; k++) fe -= (visBias[k] ?? 0) * (v[k] ?? 0);
        for (let j = 0; j < H; j++) {
          let act = hidBias[j] ?? 0;
          for (let k = 0; k < nFeatures; k++) act += (W[j]?.[k] ?? 0) * (v[k] ?? 0);
          fe -= Math.log(1 + Math.exp(act));
        }
        return fe;
      };

      // Randomly flip one bit
      const bit = Math.floor(Math.random() * nFeatures);
      const xiFlip = xi.slice() as Float64Array;
      xiFlip[bit] = 1 - (xi[bit] ?? 0);

      const fe = freeEnergy(xi);
      const feFlip = freeEnergy(xiFlip);
      pl += nFeatures * Math.log(sigmoid(feFlip - fe));
    }
    return pl / X.length;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.components_ === null || this.hiddenBias_ === null) {
      throw new Error("BernoulliRBMExt must be fitted first");
    }
    const W = this.components_;
    const hidBias = this.hiddenBias_;
    const H = this.nComponents;
    const nFeatures = X[0]?.length ?? 0;
    return X.map((v) =>
      Float64Array.from({ length: H }, (_, j) => {
        let act = hidBias[j] ?? 0;
        for (let k = 0; k < nFeatures; k++) act += (W[j]?.[k] ?? 0) * (v[k] ?? 0);
        return sigmoid(act);
      }),
    );
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

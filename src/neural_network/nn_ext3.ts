/**
 * Additional neural network utilities: EarlyStopping, LearningRateScheduler.
 * Mirrors sklearn.neural_network extras.
 */

export interface EarlyStoppingState {
  bestLoss: number;
  nItersNoChange: number;
  shouldStop: boolean;
}

export class EarlyStopping {
  patience: number;
  tol: number;
  restore: boolean;

  private state_: EarlyStoppingState = {
    bestLoss: Number.POSITIVE_INFINITY,
    nItersNoChange: 0,
    shouldStop: false,
  };

  constructor(
    options: {
      patience?: number;
      tol?: number;
      restore?: boolean;
    } = {},
  ) {
    this.patience = options.patience ?? 10;
    this.tol = options.tol ?? 1e-4;
    this.restore = options.restore ?? true;
  }

  update(loss: number): boolean {
    if (loss < this.state_.bestLoss - this.tol) {
      this.state_.bestLoss = loss;
      this.state_.nItersNoChange = 0;
    } else {
      this.state_.nItersNoChange++;
    }

    if (this.state_.nItersNoChange >= this.patience) {
      this.state_.shouldStop = true;
    }

    return this.state_.shouldStop;
  }

  get shouldStop(): boolean {
    return this.state_.shouldStop;
  }

  get bestLoss(): number {
    return this.state_.bestLoss;
  }

  reset(): void {
    this.state_ = {
      bestLoss: Number.POSITIVE_INFINITY,
      nItersNoChange: 0,
      shouldStop: false,
    };
  }
}

export type LRSchedule = "constant" | "invscaling" | "adaptive";

export class LearningRateScheduler {
  initialLr: number;
  schedule: LRSchedule;
  powerT: number;
  private currentLr_: number;
  private iter_: number = 0;
  private bestLoss_: number = Number.POSITIVE_INFINITY;

  constructor(
    options: {
      initialLr?: number;
      schedule?: LRSchedule;
      powerT?: number;
    } = {},
  ) {
    this.initialLr = options.initialLr ?? 0.1;
    this.schedule = options.schedule ?? "constant";
    this.powerT = options.powerT ?? 0.5;
    this.currentLr_ = this.initialLr;
  }

  step(loss?: number): number {
    this.iter_++;
    switch (this.schedule) {
      case "constant":
        break;
      case "invscaling":
        this.currentLr_ = this.initialLr / Math.pow(this.iter_, this.powerT);
        break;
      case "adaptive":
        if (loss !== undefined && loss <= this.bestLoss_ - 1e-4) {
          this.bestLoss_ = loss;
        } else if (loss !== undefined) {
          this.currentLr_ /= 5;
        }
        break;
    }
    return this.currentLr_;
  }

  get currentLr(): number {
    return this.currentLr_;
  }

  reset(): void {
    this.currentLr_ = this.initialLr;
    this.iter_ = 0;
    this.bestLoss_ = Number.POSITIVE_INFINITY;
  }
}

export function initWeights(
  layerSizes: number[],
  randomState = 0,
  activation: "relu" | "tanh" | "logistic" = "relu",
): Array<{ weights: Float64Array[]; biases: Float64Array }> {
  let rng = randomState;
  const nextRand = (): number => {
    rng = (rng * 1664525 + 1013904223) >>> 0;
    return (rng / 4294967296) * 2 - 1;
  };

  const layers: Array<{ weights: Float64Array[]; biases: Float64Array }> = [];
  for (let l = 0; l < layerSizes.length - 1; l++) {
    const nIn = layerSizes[l] ?? 1;
    const nOut = layerSizes[l + 1] ?? 1;
    // Glorot initialization
    const limit = Math.sqrt(6 / (nIn + nOut));
    const weights: Float64Array[] = Array.from({ length: nIn }, () => {
      const w = new Float64Array(nOut);
      for (let j = 0; j < nOut; j++) w[j] = nextRand() * limit;
      return w;
    });
    const biases = new Float64Array(nOut);
    layers.push({ weights, biases });
  }
  return layers;
}

export function relu(x: Float64Array): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = Math.max(0, x[i] ?? 0);
  return out;
}

export function tanhActivation(x: Float64Array): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = Math.tanh(x[i] ?? 0);
  return out;
}

export function logisticActivation(x: Float64Array): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = 1 / (1 + Math.exp(-(x[i] ?? 0)));
  return out;
}

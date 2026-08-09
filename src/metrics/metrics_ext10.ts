/**
 * Online metrics — streaming/incremental computation of evaluation metrics.
 */

export class RunningMean {
  private sum: number = 0;
  private count: number = 0;

  update(value: number, weight = 1): void {
    this.sum += value * weight;
    this.count += weight;
  }

  value(): number {
    return this.count > 0 ? this.sum / this.count : 0;
  }

  reset(): void {
    this.sum = 0;
    this.count = 0;
  }
}

export class RunningVariance {
  private mean: number = 0;
  private m2: number = 0;
  private count: number = 0;

  update(value: number): void {
    this.count++;
    const delta = value - this.mean;
    this.mean += delta / this.count;
    const delta2 = value - this.mean;
    this.m2 += delta * delta2;
  }

  variance(): number {
    return this.count < 2 ? 0 : this.m2 / (this.count - 1);
  }

  std(): number {
    return Math.sqrt(this.variance());
  }

  mean_(): number {
    return this.mean;
  }

  reset(): void {
    this.mean = 0;
    this.m2 = 0;
    this.count = 0;
  }
}

export class OnlineR2Score {
  private n: number = 0;
  private yMean: number = 0;
  private ssRes: number = 0;
  private ssTot: number = 0;

  update(yTrue: number, yPred: number): void {
    this.n++;
    const delta = yTrue - this.yMean;
    this.yMean += delta / this.n;
    this.ssRes += (yTrue - yPred) ** 2;
    this.ssTot += delta * (yTrue - this.yMean);
  }

  score(): number {
    return this.ssTot === 0 ? 0 : 1 - this.ssRes / this.ssTot;
  }

  reset(): void {
    this.n = 0;
    this.yMean = 0;
    this.ssRes = 0;
    this.ssTot = 0;
  }
}

export class OnlineAccuracy {
  private correct: number = 0;
  private total: number = 0;

  update(yTrue: number, yPred: number): void {
    this.total++;
    if (yTrue === yPred) this.correct++;
  }

  score(): number {
    return this.total > 0 ? this.correct / this.total : 0;
  }

  reset(): void {
    this.correct = 0;
    this.total = 0;
  }
}

export class OnlineConfusionMatrix {
  private matrix: Map<string, number> = new Map();
  private classes: Set<number> = new Set();

  update(yTrue: number, yPred: number): void {
    this.classes.add(yTrue);
    this.classes.add(yPred);
    const key = `${yTrue},${yPred}`;
    this.matrix.set(key, (this.matrix.get(key) ?? 0) + 1);
  }

  getMatrix(): { classes: number[]; matrix: number[][] } {
    const classes = Array.from(this.classes).sort((a, b) => a - b);
    const k = classes.length;
    const mat = Array.from({ length: k }, () => new Array<number>(k).fill(0));
    for (const [key, count] of this.matrix) {
      const [r, c] = key.split(",").map(Number);
      const ri = classes.indexOf(r ?? 0);
      const ci = classes.indexOf(c ?? 0);
      if (ri >= 0 && ci >= 0) mat[ri]![ci] = count;
    }
    return { classes, matrix: mat };
  }

  reset(): void {
    this.matrix.clear();
    this.classes.clear();
  }
}

export class ExponentialMovingAverage {
  private value: number = 0;
  private initialized: boolean = false;
  alpha: number;

  constructor(alpha = 0.1) {
    this.alpha = alpha;
  }

  update(x: number): number {
    if (!this.initialized) {
      this.value = x;
      this.initialized = true;
    } else {
      this.value = this.alpha * x + (1 - this.alpha) * this.value;
    }
    return this.value;
  }

  current(): number {
    return this.value;
  }

  reset(): void {
    this.value = 0;
    this.initialized = false;
  }
}

export class OnlinePrecisionRecall {
  private tp: number = 0;
  private fp: number = 0;
  private fn: number = 0;

  update(yTrue: number, yPred: number): void {
    if (yTrue === 1 && yPred === 1) this.tp++;
    else if (yTrue === 0 && yPred === 1) this.fp++;
    else if (yTrue === 1 && yPred === 0) this.fn++;
  }

  precision(): number {
    return this.tp + this.fp > 0 ? this.tp / (this.tp + this.fp) : 0;
  }

  recall(): number {
    return this.tp + this.fn > 0 ? this.tp / (this.tp + this.fn) : 0;
  }

  f1(): number {
    const p = this.precision(), r = this.recall();
    return p + r > 0 ? 2 * p * r / (p + r) : 0;
  }

  reset(): void {
    this.tp = 0;
    this.fp = 0;
    this.fn = 0;
  }
}

export class StreamingMAE {
  private errors: number[] = [];
  private windowSize: number;

  constructor(windowSize = 100) {
    this.windowSize = windowSize;
  }

  update(yTrue: number, yPred: number): void {
    this.errors.push(Math.abs(yTrue - yPred));
    if (this.errors.length > this.windowSize) this.errors.shift();
  }

  score(): number {
    if (this.errors.length === 0) return 0;
    return this.errors.reduce((a, b) => a + b, 0) / this.errors.length;
  }

  reset(): void {
    this.errors = [];
  }
}

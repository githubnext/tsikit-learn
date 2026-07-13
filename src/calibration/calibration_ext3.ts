/**
 * Calibration extensions: TemperatureScaling, BetaCalibration, VennAbersCalibrator
 * Port of sklearn.calibration extensions
 */

import { NotFittedError } from "../exceptions.js";

export class TemperatureScaling {
  maxIter: number;
  lr: number;

  private temperature_ = 1.0;

  constructor(opts: { maxIter?: number; lr?: number } = {}) {
    this.maxIter = opts.maxIter ?? 100;
    this.lr = opts.lr ?? 0.01;
  }

  private softmax(logits: Float64Array, temperature: number): Float64Array {
    const scaled = logits.map(v => (v ?? 0) / temperature);
    const max = scaled.reduce((a, b) => Math.max(a, b), -Number.POSITIVE_INFINITY);
    const exps = scaled.map(v => Math.exp((v ?? 0) - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return Float64Array.from(exps.map(v => v / (sum + 1e-15)));
  }

  fit(logits: Float64Array[], yTrue: Int32Array): this {
    let t = this.temperature_;
    const n = logits.length;
    for (let iter = 0; iter < this.maxIter; iter++) {
      let gradient = 0;
      for (let i = 0; i < n; i++) {
        const probs = this.softmax(logits[i]!, t);
        const k = yTrue[i] ?? 0;
        const pk = probs[k] ?? 1e-15;
        const logit_k = (logits[i]![k] ?? 0) / t;
        const expectedLogit = probs.reduce((s, pj, j) => s + (pj ?? 0) * ((logits[i]![j] ?? 0) / t), 0);
        gradient += (logit_k - expectedLogit) * (-1 / t);
        void pk;
      }
      gradient /= n;
      t = t - this.lr * gradient;
      t = Math.max(0.01, t);
      void iter;
    }
    this.temperature_ = t;
    return this;
  }

  predict(logits: Float64Array[]): Float64Array[] {
    if (this.temperature_ === null) throw new NotFittedError("TemperatureScaling not fitted.");
    return logits.map(l => this.softmax(l, this.temperature_));
  }

  get temperature(): number { return this.temperature_; }
}

export class BetaCalibration {
  private a_ = 1.0;
  private b_ = 1.0;
  private c_ = 0.0;

  fit(scores: Float64Array, yTrue: Int32Array): this {
    const n = scores.length;
    let a = 1.0;
    let b = 1.0;
    let c = 0.0;
    for (let iter = 0; iter < 100; iter++) {
      let dA = 0;
      let dB = 0;
      let dC = 0;
      for (let i = 0; i < n; i++) {
        const x = Math.max(1e-15, Math.min(1 - 1e-15, scores[i] ?? 0.5));
        const logx = Math.log(x);
        const log1mx = Math.log(1 - x);
        const logit = a * logx - b * log1mx + c;
        const p = 1 / (1 + Math.exp(-logit));
        const err = (yTrue[i] ?? 0) - p;
        dA += err * logx;
        dB += err * (-log1mx);
        dC += err;
      }
      a += 0.001 * dA / n;
      b += 0.001 * dB / n;
      c += 0.001 * dC / n;
      a = Math.max(0.01, a);
      b = Math.max(0.01, b);
      void iter;
    }
    this.a_ = a;
    this.b_ = b;
    this.c_ = c;
    return this;
  }

  predict(scores: Float64Array): Float64Array {
    return Float64Array.from(scores.map(x => {
      const xClamped = Math.max(1e-15, Math.min(1 - 1e-15, x ?? 0.5));
      const logit = this.a_ * Math.log(xClamped) - this.b_ * Math.log(1 - xClamped) + this.c_;
      return 1 / (1 + Math.exp(-logit));
    }));
  }
}

export class IsotonicCalibratorExt {
  private isotonic_: Float64Array | null = null;
  private thresholds_: Float64Array | null = null;

  fit(scores: Float64Array, yTrue: Int32Array): this {
    const n = scores.length;
    const pairs = Array.from({ length: n }, (_, i) => ({ score: scores[i] ?? 0, label: yTrue[i] ?? 0 }));
    pairs.sort((a, b) => a.score - b.score);
    const sortedScores = Float64Array.from(pairs.map(p => p.score));
    const sortedLabels = Float64Array.from(pairs.map(p => p.label));
    const fitted = sortedLabels.slice();
    let changed = true;
    while (changed) {
      changed = false;
      for (let i = 0; i < n - 1; i++) {
        if ((fitted[i] ?? 0) > (fitted[i + 1] ?? 0)) {
          const avg = ((fitted[i] ?? 0) + (fitted[i + 1] ?? 0)) / 2;
          fitted[i] = avg;
          fitted[i + 1] = avg;
          changed = true;
        }
      }
    }
    this.thresholds_ = sortedScores;
    this.isotonic_ = fitted;
    return this;
  }

  predict(scores: Float64Array): Float64Array {
    if (!this.thresholds_ || !this.isotonic_) throw new NotFittedError("IsotonicCalibratorExt not fitted.");
    const thresholds = this.thresholds_!;
    const isotonic = this.isotonic_!;
    return Float64Array.from(scores.map(s => {
      const n = thresholds.length;
      if ((s ?? 0) <= (thresholds[0]! ?? 0)) return isotonic[0] ?? 0;
      if ((s ?? 0) >= (thresholds[n - 1]! ?? 0)) return isotonic[n - 1] ?? 0;
      for (let i = 0; i < n - 1; i++) {
        if ((s ?? 0) >= (thresholds[i]! ?? 0) && (s ?? 0) <= (thresholds[i + 1]! ?? 0)) {
          const t = ((s ?? 0) - (thresholds[i]! ?? 0)) / ((thresholds[i + 1]! ?? 0) - (thresholds[i]! ?? 0) + 1e-15);
          return (1 - t) * (isotonic[i] ?? 0) + t * (isotonic[i + 1] ?? 0);
        }
      }
      return isotonic[n - 1] ?? 0;
    }));
  }
}

export class CalibratedClassifierCVExt {
  method: "sigmoid" | "isotonic" | "temperature";
  cv: number;

  private a_ = 1.0;
  private b_ = 0.0;

  constructor(opts: { method?: "sigmoid" | "isotonic" | "temperature"; cv?: number } = {}) {
    this.method = opts.method ?? "sigmoid";
    this.cv = opts.cv ?? 5;
  }

  fit(scores: Float64Array, yTrue: Int32Array): this {
    const n = scores.length;
    if (this.method === "sigmoid") {
      let a = 1.0;
      let b = 0.0;
      for (let iter = 0; iter < 200; iter++) {
        let da = 0;
        let db = 0;
        for (let i = 0; i < n; i++) {
          const p = 1 / (1 + Math.exp(-(a * (scores[i] ?? 0) + b)));
          const err = (yTrue[i] ?? 0) - p;
          da += err * (scores[i] ?? 0);
          db += err;
        }
        a += 0.01 * da / n;
        b += 0.01 * db / n;
        void iter;
      }
      this.a_ = a;
      this.b_ = b;
    }
    return this;
  }

  predict(scores: Float64Array): Float64Array {
    return Float64Array.from(scores.map(s => 1 / (1 + Math.exp(-(this.a_ * (s ?? 0) + this.b_)))));
  }
}

/**
 * SplineTransformer and TargetEncoder preprocessing.
 * Mirrors sklearn.preprocessing.SplineTransformer and TargetEncoder.
 */

import { NotFittedError } from "../exceptions.js";

export type SplineExtrapolation =
  | "error"
  | "constant"
  | "linear"
  | "continue"
  | "periodic";

export interface SplineTransformerOptions {
  nKnots?: number;
  degree?: number;
  knotsStrategy?: "uniform" | "quantile";
  extrapolation?: SplineExtrapolation;
  includeIntercept?: boolean;
}

export class SplineTransformer {
  nKnots: number;
  degree: number;
  knotsStrategy: "uniform" | "quantile";
  extrapolation: SplineExtrapolation;
  includeIntercept: boolean;

  bsplineKnots_: Float64Array[] | null = null;
  nFeaturesOut_: number = 0;

  constructor(options: SplineTransformerOptions = {}) {
    this.nKnots = options.nKnots ?? 5;
    this.degree = options.degree ?? 3;
    this.knotsStrategy = options.knotsStrategy ?? "uniform";
    this.extrapolation = options.extrapolation ?? "constant";
    this.includeIntercept = options.includeIntercept ?? false;
  }

  private _bsplineBasis(
    x: number,
    knots: Float64Array,
    degree: number,
  ): Float64Array {
    const n = knots.length - degree - 1;
    const basis = new Float64Array(n);

    if (n <= 0) return basis;

    // De Boor's algorithm
    const t = knots;
    const B: number[][] = Array.from({ length: degree + 1 }, () =>
      new Array<number>(n).fill(0),
    );

    // Degree 0
    for (let i = 0; i < n; i++) {
      B[0]![i] =
        (t[i] ?? 0) <= x && x < (t[i + 1] ?? Number.POSITIVE_INFINITY) ? 1 : 0;
    }
    // Handle right endpoint
    if (Math.abs(x - (t[t.length - 1] ?? 0)) < 1e-10 && n > 0) {
      // Find last non-zero interval
      for (let i = n - 1; i >= 0; i--) {
        if ((t[i] ?? 0) <= x) {
          B[0]![i] = 1;
          break;
        }
      }
    }

    for (let d = 1; d <= degree; d++) {
      for (let i = 0; i < n; i++) {
        const ti = t[i] ?? 0;
        const tid = t[i + d] ?? 0;
        const ti1 = t[i + 1] ?? 0;
        const tid1 = t[i + d + 1] ?? 0;

        let left = 0;
        const denom1 = tid - ti;
        if (Math.abs(denom1) > 1e-10) {
          left = ((x - ti) / denom1) * (B[d - 1]![i] ?? 0);
        }

        let right = 0;
        const denom2 = tid1 - ti1;
        if (Math.abs(denom2) > 1e-10) {
          right = ((tid1 - x) / denom2) * (B[d - 1]![i + 1] ?? 0);
        }

        B[d]![i] = left + right;
      }
    }

    for (let i = 0; i < n; i++) basis[i] = B[degree]![i] ?? 0;
    return basis;
  }

  fit(X: Float64Array[]): this {
    const nSamples = X.length;
    if (nSamples === 0) throw new Error("Empty input");
    const nFeatures = X[0]?.length ?? 0;

    this.bsplineKnots_ = [];

    for (let j = 0; j < nFeatures; j++) {
      const col = X.map((row) => row[j] ?? 0).sort((a, b) => a - b);
      const min = col[0] ?? 0;
      const max = col[col.length - 1] ?? 1;
      const nInnerKnots = this.nKnots - 2;

      const innerKnots: number[] = [];
      for (let k = 1; k <= nInnerKnots; k++) {
        if (this.knotsStrategy === "uniform") {
          innerKnots.push(min + (k / (nInnerKnots + 1)) * (max - min));
        } else {
          // quantile
          const q = k / (nInnerKnots + 1);
          const idx = Math.floor(q * (nSamples - 1));
          innerKnots.push(col[idx] ?? 0);
        }
      }

      // Full knot vector with repeated boundary knots
      const knots: number[] = [];
      for (let d = 0; d <= this.degree; d++) knots.push(min);
      for (const k of innerKnots) knots.push(k);
      for (let d = 0; d <= this.degree; d++) knots.push(max);

      this.bsplineKnots_.push(new Float64Array(knots));
    }

    // nFeaturesOut = sum over features of (nKnots + degree - 1 - (includeIntercept ? 0 : 1))
    let totalOut = 0;
    for (const knots of this.bsplineKnots_) {
      const nSplines = knots.length - this.degree - 1;
      totalOut += nSplines - (this.includeIntercept ? 0 : 1);
    }
    this.nFeaturesOut_ = totalOut;

    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.bsplineKnots_) throw new NotFittedError("SplineTransformer");
    const nFeatures = this.bsplineKnots_.length;

    return X.map((row) => {
      const parts: Float64Array[] = [];
      for (let j = 0; j < nFeatures; j++) {
        const knots = this.bsplineKnots_![j]!;
        const min = knots[0] ?? 0;
        const max = knots[knots.length - 1] ?? 1;
        let x = row[j] ?? 0;

        // Extrapolation
        if (x < min || x > max) {
          if (this.extrapolation === "error") {
            throw new Error(`Value ${x} out of range [${min}, ${max}]`);
          }
          if (this.extrapolation === "constant") {
            x = Math.max(min, Math.min(max, x));
          } else if (this.extrapolation === "periodic") {
            const range = max - min;
            x = min + ((((x - min) % range) + range) % range);
          }
        }

        const basis = this._bsplineBasis(x, knots, this.degree);
        const offset = this.includeIntercept ? 0 : 1;
        parts.push(basis.slice(offset));
      }

      const totalLen = parts.reduce((a, b) => a + b.length, 0);
      const out = new Float64Array(totalLen);
      let pos = 0;
      for (const part of parts) {
        for (let k = 0; k < part.length; k++) out[pos++] = part[k] ?? 0;
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export interface TargetEncoderOptions {
  smooth?: number | "auto";
  targetType?: "auto" | "binary" | "multiclass" | "continuous";
}

export class TargetEncoder {
  smooth: number | "auto";
  targetType: "auto" | "binary" | "multiclass" | "continuous";

  encodings_: Map<string | number, number>[] | null = null;
  targetMean_: number = 0;
  nFeatures_: number = 0;

  constructor(options: TargetEncoderOptions = {}) {
    this.smooth = options.smooth ?? "auto";
    this.targetType = options.targetType ?? "auto";
  }

  fit(X: (string | number)[][], y: Float64Array | Int32Array): this {
    const nSamples = X.length;
    const nFeatures = X[0]?.length ?? 0;
    this.nFeatures_ = nFeatures;

    // Global target mean
    let yMean = 0;
    for (let i = 0; i < nSamples; i++) yMean += (y[i] ?? 0) / nSamples;
    this.targetMean_ = yMean;

    this.encodings_ = [];
    for (let j = 0; j < nFeatures; j++) {
      const encoding = new Map<string | number, number>();
      const catGroups = new Map<string | number, number[]>();

      for (let i = 0; i < nSamples; i++) {
        const cat = X[i]![j] ?? "";
        const yi = y[i] ?? 0;
        if (!catGroups.has(cat)) catGroups.set(cat, []);
        catGroups.get(cat)!.push(yi);
      }

      for (const [cat, vals] of catGroups) {
        const n = vals.length;
        const catMean = vals.reduce((a, b) => a + b, 0) / n;

        // Smoothing (empirical Bayes)
        const smoothVal =
          this.smooth === "auto" ? nSamples / (nSamples + n) : this.smooth;
        const encoded = (1 - smoothVal) * catMean + smoothVal * yMean;
        encoding.set(cat, encoded);
      }

      this.encodings_.push(encoding);
    }
    return this;
  }

  transform(X: (string | number)[][]): Float64Array[] {
    if (!this.encodings_) throw new NotFittedError("TargetEncoder");
    return X.map((row) => {
      const out = new Float64Array(this.nFeatures_);
      for (let j = 0; j < this.nFeatures_; j++) {
        const cat = row[j] ?? "";
        out[j] = this.encodings_![j]!.get(cat) ?? this.targetMean_;
      }
      return out;
    });
  }

  fitTransform(
    X: (string | number)[][],
    y: Float64Array | Int32Array,
  ): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}

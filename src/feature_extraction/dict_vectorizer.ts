/**
 * Feature extraction: DictVectorizer and FeatureHasher.
 * Mirrors sklearn.feature_extraction.DictVectorizer and FeatureHasher.
 */

import { NotFittedError } from "../exceptions.js";

export interface DictVectorizerOptions {
  sparse?: boolean;
  dtype?: "float64" | "float32";
  separator?: string;
  sort?: boolean;
}

export class DictVectorizer {
  sparse: boolean;
  separator: string;
  sort: boolean;

  featureNames_: string[] | null = null;
  vocabulary_: Map<string, number> | null = null;

  constructor(options: DictVectorizerOptions = {}) {
    this.sparse = options.sparse ?? false;
    this.separator = options.separator ?? "=";
    this.sort = options.sort ?? true;
  }

  fit(X: Record<string, number | string>[]): this {
    const featureSet = new Set<string>();
    for (const sample of X) {
      for (const [key, value] of Object.entries(sample)) {
        if (typeof value === "number") {
          featureSet.add(key);
        } else {
          featureSet.add(`${key}${this.separator}${value}`);
        }
      }
    }
    let features = Array.from(featureSet);
    if (this.sort) features = features.sort();
    this.featureNames_ = features;
    this.vocabulary_ = new Map(features.map((f, i) => [f, i]));
    return this;
  }

  transform(X: Record<string, number | string>[]): Float64Array[] {
    if (!this.vocabulary_ || !this.featureNames_) throw new NotFittedError("DictVectorizer is not fitted.");
    const p = this.featureNames_.length;
    return X.map(sample => {
      const row = new Float64Array(p);
      for (const [key, value] of Object.entries(sample)) {
        let featureName: string;
        let featureVal: number;
        if (typeof value === "number") {
          featureName = key;
          featureVal = value;
        } else {
          featureName = `${key}${this.separator}${value}`;
          featureVal = 1;
        }
        const idx = this.vocabulary_!.get(featureName);
        if (idx !== undefined) row[idx] = featureVal;
      }
      return row;
    });
  }

  fitTransform(X: Record<string, number | string>[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Float64Array[]): Record<string, number>[] {
    if (!this.featureNames_) throw new NotFittedError("DictVectorizer is not fitted.");
    return X.map(row => {
      const result: Record<string, number> = {};
      for (let j = 0; j < row.length; j++) {
        const v = row[j] ?? 0;
        if (v !== 0) result[this.featureNames_![j] ?? `f${j}`] = v;
      }
      return result;
    });
  }

  getFeatureNames(): string[] {
    if (!this.featureNames_) throw new NotFittedError("DictVectorizer is not fitted.");
    return this.featureNames_;
  }
}

export interface FeatureHasherOptions {
  nFeatures?: number;
  inputType?: "dict" | "pair" | "string";
  dtype?: "float64" | "float32";
  alternateSign?: boolean;
}

export class FeatureHasher {
  nFeatures: number;
  alternateSign: boolean;

  constructor(options: FeatureHasherOptions = {}) {
    this.nFeatures = options.nFeatures ?? 1048576;
    this.alternateSign = options.alternateSign ?? true;
  }

  private _hash(s: string): number {
    let h = 5381;
    for (let i = 0; i < s.length; i++) {
      h = ((h << 5) + h + s.charCodeAt(i)) >>> 0;
    }
    return h;
  }

  transform(X: Record<string, number>[]): Float64Array[] {
    const p = this.nFeatures;
    return X.map(sample => {
      const row = new Float64Array(p);
      for (const [key, value] of Object.entries(sample)) {
        const h = this._hash(key);
        const idx = h % p;
        const sign = this.alternateSign ? ((h >>> 31) ? -1 : 1) : 1;
        row[idx] = (row[idx] ?? 0) + sign * value;
      }
      return row;
    });
  }

  fit(_X: Record<string, number>[]): this { return this; }

  fitTransform(X: Record<string, number>[]): Float64Array[] {
    return this.transform(X);
  }
}

/**
 * OneHotEncoder and OrdinalEncoder.
 * Mirrors sklearn.preprocessing.OneHotEncoder and OrdinalEncoder.
 */

import { NotFittedError } from "../exceptions.js";

export class OneHotEncoder {
  sparse: boolean;
  handleUnknown: string;

  categories_: Float64Array[] | null = null;
  featureNamesOut_: string[] | null = null;

  constructor(
    options: { sparse?: boolean; handleUnknown?: string } = {},
  ) {
    this.sparse = options.sparse ?? false;
    this.handleUnknown = options.handleUnknown ?? "error";
  }

  fit(X: Float64Array[]): this {
    const p = (X[0] ?? new Float64Array(0)).length;
    this.categories_ = [];
    for (let j = 0; j < p; j++) {
      const vals = Array.from(new Set(X.map((xi) => xi[j] ?? 0))).sort((a, b) => a - b);
      this.categories_.push(new Float64Array(vals));
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.categories_ === null) throw new NotFittedError("OneHotEncoder");
    const cats = this.categories_;

    return X.map((xi) => {
      const parts: number[] = [];
      for (let j = 0; j < xi.length; j++) {
        const cat = cats[j] ?? new Float64Array(0);
        const val = xi[j] ?? 0;
        for (let k = 0; k < cat.length; k++) {
          parts.push(cat[k] === val ? 1 : 0);
        }
      }
      return new Float64Array(parts);
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Float64Array[]): Float64Array[] {
    if (this.categories_ === null) throw new NotFittedError("OneHotEncoder");
    const cats = this.categories_;
    const p = cats.length;

    return X.map((xi) => {
      const result = new Float64Array(p);
      let offset = 0;
      for (let j = 0; j < p; j++) {
        const cat = cats[j] ?? new Float64Array(0);
        let maxVal = Number.NEGATIVE_INFINITY;
        let bestIdx = 0;
        for (let k = 0; k < cat.length; k++) {
          if ((xi[offset + k] ?? 0) > maxVal) {
            maxVal = xi[offset + k] ?? 0;
            bestIdx = k;
          }
        }
        result[j] = cat[bestIdx] ?? 0;
        offset += cat.length;
      }
      return result;
    });
  }
}

export class OrdinalEncoder {
  categories_: Float64Array[] | null = null;

  fit(X: Float64Array[]): this {
    const p = (X[0] ?? new Float64Array(0)).length;
    this.categories_ = [];
    for (let j = 0; j < p; j++) {
      const vals = Array.from(new Set(X.map((xi) => xi[j] ?? 0))).sort((a, b) => a - b);
      this.categories_.push(new Float64Array(vals));
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.categories_ === null) throw new NotFittedError("OrdinalEncoder");
    const cats = this.categories_;
    return X.map((xi) => {
      const result = new Float64Array(xi.length);
      for (let j = 0; j < xi.length; j++) {
        const cat = cats[j] ?? new Float64Array(0);
        const val = xi[j] ?? 0;
        const idx = Array.from(cat).indexOf(val);
        result[j] = idx >= 0 ? idx : 0;
      }
      return result;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Float64Array[]): Float64Array[] {
    if (this.categories_ === null) throw new NotFittedError("OrdinalEncoder");
    const cats = this.categories_;
    return X.map((xi) => {
      const result = new Float64Array(xi.length);
      for (let j = 0; j < xi.length; j++) {
        const cat = cats[j] ?? new Float64Array(0);
        const idx = Math.round(xi[j] ?? 0);
        result[j] = cat[Math.min(idx, cat.length - 1)] ?? 0;
      }
      return result;
    });
  }
}

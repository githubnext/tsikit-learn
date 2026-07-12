/**
 * MissingIndicator transformer.
 * Mirrors sklearn.impute.MissingIndicator.
 */

import { NotFittedError } from "../exceptions.js";

export interface MissingIndicatorOptions {
  missingValues?: number;
  features?: "missing-only" | "all";
  sparse?: boolean | "auto";
  errorOnNew?: boolean;
}

/**
 * Binary indicators for missing values.
 * Transforms a dataset to boolean indicator matrix for missing values.
 */
export class MissingIndicator {
  private missingValues: number;
  private features: "missing-only" | "all";
  private errorOnNew: boolean;

  features_?: Int32Array;
  nFeatures_?: number;

  constructor(options: MissingIndicatorOptions = {}) {
    this.missingValues = options.missingValues ?? Number.NaN;
    this.features = options.features ?? "missing-only";
    this.errorOnNew = options.errorOnNew ?? true;
  }

  private isMissing(val: number): boolean {
    if (Number.isNaN(this.missingValues)) return Number.isNaN(val);
    return val === this.missingValues;
  }

  fit(X: Float64Array[]): this {
    if (X.length === 0) {
      this.features_ = new Int32Array(0);
      this.nFeatures_ = 0;
      return this;
    }
    const nFeats = X[0]!.length;
    this.nFeatures_ = nFeats;

    if (this.features === "all") {
      this.features_ = new Int32Array(
        Array.from({ length: nFeats }, (_, i) => i),
      );
    } else {
      // "missing-only": only columns with at least one missing value
      const hasMissing: boolean[] = new Array(nFeats).fill(false) as boolean[];
      for (const row of X) {
        for (let j = 0; j < nFeats; j++) {
          if (this.isMissing(row[j] ?? 0)) hasMissing[j] = true;
        }
      }
      this.features_ = new Int32Array(
        hasMissing.map((v, i) => (v ? i : -1)).filter((v) => v >= 0),
      );
    }
    return this;
  }

  transform(X: Float64Array[]): Uint8Array[] {
    if (!this.features_)
      throw new NotFittedError("MissingIndicator is not fitted");
    const nOut = this.features_.length;

    return X.map((row) => {
      const indicator = new Uint8Array(nOut);
      for (let j = 0; j < nOut; j++) {
        const featIdx = this.features_![j] ?? 0;
        indicator[j] = this.isMissing(row[featIdx] ?? 0) ? 1 : 0;
      }
      return indicator;
    });
  }

  fitTransform(X: Float64Array[]): Uint8Array[] {
    return this.fit(X).transform(X);
  }

  getParams(): Record<string, unknown> {
    return {
      missingValues: this.missingValues,
      features: this.features,
      errorOnNew: this.errorOnNew,
    };
  }

  setParams(params: Record<string, unknown>): this {
    const p = params as {
      missingValues?: number;
      features?: "missing-only" | "all";
      errorOnNew?: boolean;
    };
    if (p.missingValues !== undefined) this.missingValues = p.missingValues;
    if (p.features !== undefined) this.features = p.features;
    if (p.errorOnNew !== undefined) this.errorOnNew = p.errorOnNew;
    return this;
  }
}

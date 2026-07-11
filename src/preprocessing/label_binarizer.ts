/**
 * LabelBinarizer and MultiLabelBinarizer.
 * Mirrors sklearn.preprocessing.LabelBinarizer and MultiLabelBinarizer.
 */

import { NotFittedError } from "../exceptions.js";

export interface LabelBinarizerOptions {
  negLabel?: number;
  posLabel?: number;
  sparseOutput?: boolean;
}

/**
 * Binarize labels in a one-vs-all fashion.
 * For binary classes produces a single column; multiclass produces n_classes columns.
 * Mirrors sklearn.preprocessing.LabelBinarizer.
 */
export class LabelBinarizer {
  negLabel: number;
  posLabel: number;

  classes_: string[] | null = null;
  yType_: "binary" | "multiclass" = "binary";
  sparseInput_: boolean = false;

  constructor(options: LabelBinarizerOptions = {}) {
    this.negLabel = options.negLabel ?? 0;
    this.posLabel = options.posLabel ?? 1;
  }

  fit(y: string[]): this {
    const unique = Array.from(new Set(y)).sort();
    this.classes_ = unique;
    this.yType_ = unique.length <= 2 ? "binary" : "multiclass";
    return this;
  }

  transform(y: string[]): Float64Array[] {
    if (!this.classes_)
      throw new NotFittedError("LabelBinarizer is not fitted yet.");
    const n = y.length;
    const k = this.classes_.length;

    if (this.yType_ === "binary") {
      // Single column: posLabel for positive class (index 1), negLabel otherwise
      const posClass = this.classes_[1] ?? this.classes_[0] ?? "";
      return Array.from({ length: n }, (_, i) => {
        const v = new Float64Array(1);
        v[0] = y[i] === posClass ? this.posLabel : this.negLabel;
        return v;
      });
    }

    return Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(k).fill(this.negLabel);
      const idx = this.classes_!.indexOf(y[i] ?? "");
      if (idx !== -1) row[idx] = this.posLabel;
      return row;
    });
  }

  fitTransform(y: string[]): Float64Array[] {
    return this.fit(y).transform(y);
  }

  inverseTransform(Y: Float64Array[]): string[] {
    if (!this.classes_)
      throw new NotFittedError("LabelBinarizer is not fitted yet.");
    const k = this.classes_.length;

    if (this.yType_ === "binary") {
      const posClass = this.classes_[1] ?? this.classes_[0] ?? "";
      const negClass = this.classes_[0] ?? "";
      return Y.map((row) => ((row[0] ?? 0) > 0 ? posClass : negClass));
    }

    return Y.map((row) => {
      let best = -1;
      let bestVal = -Number.POSITIVE_INFINITY;
      for (let j = 0; j < k; j++) {
        if ((row[j] ?? 0) > bestVal) {
          bestVal = row[j] ?? 0;
          best = j;
        }
      }
      return best !== -1
        ? (this.classes_![best] ?? "")
        : (this.classes_![0] ?? "");
    });
  }
}

export interface MultiLabelBinarizerOptions {
  classes?: string[];
}

/**
 * Transform between iterable of iterables and a multilabel format.
 * Mirrors sklearn.preprocessing.MultiLabelBinarizer.
 */
export class MultiLabelBinarizer {
  classesInput: string[] | null;

  classes_: string[] | null = null;

  constructor(options: MultiLabelBinarizerOptions = {}) {
    this.classesInput = options.classes ?? null;
  }

  fit(y: string[][]): this {
    if (this.classesInput) {
      this.classes_ = [...this.classesInput];
    } else {
      const unique = new Set<string>();
      for (const row of y) for (const label of row) unique.add(label);
      this.classes_ = Array.from(unique).sort();
    }
    return this;
  }

  transform(y: string[][]): Float64Array[] {
    if (!this.classes_)
      throw new NotFittedError("MultiLabelBinarizer is not fitted yet.");
    const k = this.classes_.length;
    return y.map((labels) => {
      const row = new Float64Array(k);
      for (const label of labels) {
        const idx = this.classes_!.indexOf(label);
        if (idx !== -1) row[idx] = 1;
      }
      return row;
    });
  }

  fitTransform(y: string[][]): Float64Array[] {
    return this.fit(y).transform(y);
  }

  inverseTransform(Y: Float64Array[]): string[][] {
    if (!this.classes_)
      throw new NotFittedError("MultiLabelBinarizer is not fitted yet.");
    return Y.map((row) => {
      const labels: string[] = [];
      for (let j = 0; j < this.classes_!.length; j++) {
        if ((row[j] ?? 0) !== 0) labels.push(this.classes_![j] ?? "");
      }
      return labels;
    });
  }
}

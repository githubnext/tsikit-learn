/**
 * Feature hashing (the hashing trick).
 * Mirrors scikit-learn's feature_extraction.FeatureHasher.
 */

export interface FeatureHasherOptions {
  nFeatures?: number;
  inputType?: "dict" | "pair" | "string";
  dtype?: "float32" | "float64";
  alternateSign?: boolean;
}

type InputDict = Record<string, number>;
type InputPair = [string, number];

/** MurmurHash-inspired 32-bit hash function */
function hashFeature(key: string, seed = 0): number {
  let h = seed;
  for (let i = 0; i < key.length; i++) {
    h = Math.imul(h ^ key.charCodeAt(i), 0x9e3779b9);
    h ^= h >>> 16;
  }
  return h >>> 0;
}

/**
 * Hash features from a dictionary, list of pairs, or string tokens into a fixed-size vector.
 */
export class FeatureHasherExt {
  readonly nFeatures: number;
  readonly inputType: "dict" | "pair" | "string";
  readonly alternateSign: boolean;

  constructor(options: FeatureHasherOptions = {}) {
    this.nFeatures = options.nFeatures ?? 1048576; // 2^20
    this.inputType = options.inputType ?? "dict";
    this.alternateSign = options.alternateSign ?? true;
  }

  transform(
    rawXIter: Iterable<InputDict | InputPair[] | string[]>,
  ): Float64Array[] {
    return Array.from(rawXIter, (rawX) => this._transformOne(rawX));
  }

  private _transformOne(
    rawX: InputDict | InputPair[] | string[],
  ): Float64Array {
    const row = new Float64Array(this.nFeatures);

    const process = (key: string, value: number): void => {
      const h = hashFeature(key);
      const idx = h % this.nFeatures;
      const sign = this.alternateSign ? (h & 1 ? 1 : -1) : 1;
      row[idx] = (row[idx] ?? 0) + sign * value;
    };

    if (this.inputType === "dict") {
      const d = rawX as InputDict;
      for (const [k, v] of Object.entries(d)) process(k, v);
    } else if (this.inputType === "pair") {
      for (const [k, v] of rawX as InputPair[]) process(k, v);
    } else {
      for (const token of rawX as string[]) process(token, 1);
    }

    return row;
  }
}

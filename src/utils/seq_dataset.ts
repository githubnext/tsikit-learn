/**
 * Sequential/dataset utilities for SGD solvers.
 * Port of sklearn.utils._seq_dataset
 */

export interface Dataset {
  nSamples: number;
  nFeatures: number;
  next(): [Float64Array, number, number];
  reset(): void;
  shuffle(seed: number): void;
}

/**
 * Sequential array dataset for SGD solvers.
 * Port of sklearn.utils._seq_dataset.ArrayDataset
 */
export class ArrayDataset implements Dataset {
  private X: Float64Array[];
  private y: Float64Array;
  private sampleWeight: Float64Array;
  private indices: Int32Array;
  private pos: number = 0;
  nSamples: number;
  nFeatures: number;

  constructor(
    X: Float64Array[],
    y: Float64Array,
    sampleWeight: Float64Array | null = null,
  ) {
    this.X = X;
    this.y = y;
    this.nSamples = X.length;
    this.nFeatures = X[0]?.length ?? 0;
    this.sampleWeight =
      sampleWeight ?? new Float64Array(this.nSamples).fill(1.0);
    this.indices = new Int32Array(
      Array.from({ length: this.nSamples }, (_, i) => i),
    );
  }

  next(): [Float64Array, number, number] {
    if (this.pos >= this.nSamples) this.pos = 0;
    const idx = this.indices[this.pos]!;
    this.pos++;
    return [this.X[idx]!, this.y[idx]!, this.sampleWeight[idx]!];
  }

  reset(): void {
    this.pos = 0;
  }

  shuffle(seed: number): void {
    let rng = seed;
    const rand = (): number => {
      rng = (rng * 1664525 + 1013904223) & 0xffffffff;
      return (rng >>> 0) / 0x100000000;
    };
    for (let i = this.nSamples - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const tmp = this.indices[i]!;
      this.indices[i] = this.indices[j]!;
      this.indices[j] = tmp;
    }
    this.pos = 0;
  }
}

/**
 * CSR matrix dataset for sparse SGD.
 * Port of sklearn.utils._seq_dataset.CSRDataset
 */
export class CSRDataset implements Dataset {
  private data: Float64Array;
  private indices: Int32Array;
  private indptr: Int32Array;
  private y: Float64Array;
  private sampleWeight: Float64Array;
  private sampleIndices: Int32Array;
  private pos: number = 0;
  nSamples: number;
  nFeatures: number;

  constructor(
    data: Float64Array,
    indices: Int32Array,
    indptr: Int32Array,
    y: Float64Array,
    sampleWeight: Float64Array | null,
    nFeatures: number,
  ) {
    this.data = data;
    this.indices = indices;
    this.indptr = indptr;
    this.y = y;
    this.nSamples = y.length;
    this.nFeatures = nFeatures;
    this.sampleWeight =
      sampleWeight ?? new Float64Array(this.nSamples).fill(1.0);
    this.sampleIndices = new Int32Array(
      Array.from({ length: this.nSamples }, (_, i) => i),
    );
  }

  next(): [Float64Array, number, number] {
    if (this.pos >= this.nSamples) this.pos = 0;
    const sampleIdx = this.sampleIndices[this.pos]!;
    this.pos++;

    // Expand sparse row to dense
    const row = new Float64Array(this.nFeatures);
    const start = this.indptr[sampleIdx]!;
    const end = this.indptr[sampleIdx + 1]!;
    for (let k = start; k < end; k++) {
      row[this.indices[k]!] = this.data[k]!;
    }
    return [row, this.y[sampleIdx]!, this.sampleWeight[sampleIdx]!];
  }

  reset(): void {
    this.pos = 0;
  }

  shuffle(seed: number): void {
    let rng = seed;
    const rand = (): number => {
      rng = (rng * 1664525 + 1013904223) & 0xffffffff;
      return (rng >>> 0) / 0x100000000;
    };
    for (let i = this.nSamples - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const tmp = this.sampleIndices[i]!;
      this.sampleIndices[i] = this.sampleIndices[j]!;
      this.sampleIndices[j] = tmp;
    }
    this.pos = 0;
  }
}

/** Create appropriate dataset based on data type */
export function makeDataset(
  X: Float64Array[],
  y: Float64Array,
  sampleWeight: Float64Array | null = null,
  randomState = 42,
): ArrayDataset {
  const ds = new ArrayDataset(X, y, sampleWeight);
  ds.shuffle(randomState);
  return ds;
}

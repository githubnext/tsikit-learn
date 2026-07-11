/**
 * RCV1 dataset utilities and sparse text dataset helpers.
 * Mirrors sklearn.datasets.rcv1 and related sparse dataset loaders.
 */
import type { SparseMatrix } from "../utils/sparsefuncs.js";

export interface RCV1DatasetInfo {
  nSamples: number;
  nFeatures: number;
  nCategories: number;
  description: string;
}

/** Metadata about the RCV1 corpus. */
export const RCV1_INFO: RCV1DatasetInfo = {
  nSamples: 804414,
  nFeatures: 47236,
  nCategories: 103,
  description:
    "RCV1 — Reuters Corpus Volume 1. A collection of 804,414 news articles " +
    "annotated with 103 topic categories. Features are TF-IDF weighted bag-of-words.",
};

export interface TextDataset {
  data: SparseMatrix;
  target: Int32Array;
  targetNames: string[];
  featureNames: string[];
  description: string;
}

/**
 * Build a sparse TF-IDF matrix from an array of tokenized documents.
 * Each document is an array of term strings.
 */
export function buildTfIdf(
  documents: string[][],
  options: {
    maxFeatures?: number;
    sublinearTf?: boolean;
    smoothIdf?: boolean;
  } = {},
): {
  matrix: SparseMatrix;
  vocabulary: Map<string, number>;
  idf: Float64Array;
} {
  const { maxFeatures, sublinearTf = false, smoothIdf = true } = options;
  const nDocs = documents.length;

  // Build vocabulary
  const df = new Map<string, number>();
  for (const doc of documents) {
    const seen = new Set<string>();
    for (const term of doc) {
      if (!seen.has(term)) {
        df.set(term, (df.get(term) ?? 0) + 1);
        seen.add(term);
      }
    }
  }

  // Sort by df descending, take top maxFeatures
  let vocab = [...df.entries()].sort((a, b) => b[1] - a[1]);
  if (maxFeatures !== undefined) vocab = vocab.slice(0, maxFeatures);
  const termToIdx = new Map<string, number>(vocab.map(([t], i) => [t, i]));
  const nTerms = termToIdx.size;

  // IDF
  const idf = new Float64Array(nTerms);
  for (const [term, idx] of termToIdx) {
    const dfi = df.get(term) ?? 0;
    idf[idx] =
      Math.log(((smoothIdf ? 1 : 0) + nDocs) / ((smoothIdf ? 1 : 0) + dfi)) + 1;
  }

  // Build CSR TF-IDF matrix
  const dataArr: number[] = [];
  const indicesArr: number[] = [];
  const indptrArr: number[] = [0];

  for (const doc of documents) {
    const tf = new Map<number, number>();
    for (const term of doc) {
      const idx = termToIdx.get(term);
      if (idx !== undefined) tf.set(idx, (tf.get(idx) ?? 0) + 1);
    }
    const docLen = doc.length;
    const entries = [...tf.entries()].sort((a, b) => a[0] - b[0]);
    for (const [idx, count] of entries) {
      const tfVal = sublinearTf ? 1 + Math.log(count) : count / docLen;
      const val = tfVal * (idf[idx] ?? 0);
      if (val !== 0) {
        dataArr.push(val);
        indicesArr.push(idx);
      }
    }
    indptrArr.push(dataArr.length);
  }

  const matrix: SparseMatrix = {
    data: new Float64Array(dataArr),
    indices: new Int32Array(indicesArr),
    indptr: new Int32Array(indptrArr),
    shape: [nDocs, nTerms],
  };

  return { matrix, vocabulary: termToIdx, idf };
}

/**
 * Generate a synthetic sparse text dataset for testing.
 * Returns documents drawn from `nCategories` topics with `nFeatures` vocabulary.
 */
export function makeSparseTextDataset(
  options: {
    nSamples?: number;
    nFeatures?: number;
    nCategories?: number;
    avgTermsPerDoc?: number;
    randomState?: number;
  } = {},
): {
  X: SparseMatrix;
  y: Int32Array;
  featureNames: string[];
  categoryNames: string[];
} {
  const {
    nSamples = 200,
    nFeatures = 500,
    nCategories = 5,
    avgTermsPerDoc = 20,
    randomState = 42,
  } = options;

  let seed = randomState | 0;
  const rng = (): number => {
    seed = (seed ^ (seed << 13)) >>> 0;
    seed = (seed ^ (seed >>> 17)) >>> 0;
    seed = (seed ^ (seed << 5)) >>> 0;
    return (seed >>> 0) / 0xffffffff;
  };

  const featureNames = Array.from({ length: nFeatures }, (_, i) => `word_${i}`);
  const categoryNames = Array.from(
    { length: nCategories },
    (_, i) => `category_${i}`,
  );

  const data: number[] = [];
  const indices: number[] = [];
  const indptr: number[] = [0];
  const y = new Int32Array(nSamples);

  for (let i = 0; i < nSamples; i++) {
    const cat = Math.floor(rng() * nCategories);
    y[i] = cat;
    const nTerms = Math.max(1, Math.round(avgTermsPerDoc * (0.5 + rng())));
    const tfMap = new Map<number, number>();
    for (let t = 0; t < nTerms; t++) {
      // Category-biased term selection
      const bias = rng() < 0.3 ? cat * Math.floor(nFeatures / nCategories) : 0;
      const termIdx =
        (Math.floor(rng() * Math.floor(nFeatures / nCategories)) + bias) %
        nFeatures;
      tfMap.set(termIdx, (tfMap.get(termIdx) ?? 0) + 1);
    }
    const entries = [...tfMap.entries()].sort((a, b) => a[0] - b[0]);
    for (const [idx, count] of entries) {
      data.push(count);
      indices.push(idx);
    }
    indptr.push(data.length);
  }

  const X: SparseMatrix = {
    data: new Float64Array(data),
    indices: new Int32Array(indices),
    indptr: new Int32Array(indptr),
    shape: [nSamples, nFeatures],
  };

  return { X, y, featureNames, categoryNames };
}

/**
 * Text feature extraction: CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer.
 * Mirrors sklearn.feature_extraction.text.
 */

import { NotFittedError } from "../exceptions.js";

/** Tokenize text by splitting on non-word characters (lowercase). */
function tokenize(text: string): string[] {
  return text.toLowerCase().match(/\b[a-z]+\b/g) ?? [];
}

/** Options for CountVectorizer. */
export interface CountVectorizerOptions {
  minDf?: number;
  maxDf?: number;
  maxFeatures?: number | null;
  ngramRange?: [number, number];
  lowercase?: boolean;
  analyzer?: "word" | "char";
}

/** Options for HashingVectorizer. */
export interface HashingVectorizerOptions {
  nFeatures?: number;
  alternate_sign?: boolean;
  lowercase?: boolean;
  ngramRange?: [number, number];
}

/** Options for TfidfTransformer. */
export interface TfidfTransformerOptions {
  norm?: "l1" | "l2" | null;
  useIdf?: boolean;
  smoothIdf?: boolean;
  sublinearTf?: boolean;
}

/** Simple string hash. */
function murmurhash(str: string): number {
  let h = 0xdeadbeef;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 0x9e3779b9);
    h = (h << 13) | (h >>> 19);
  }
  return (h ^ (h >>> 16)) >>> 0;
}

/**
 * Convert a collection of text documents to a matrix of token counts.
 * Mirrors sklearn.feature_extraction.text.CountVectorizer.
 */
export class CountVectorizer {
  minDf: number;
  maxDf: number;
  maxFeatures: number | null;
  ngramRange: [number, number];
  lowercase: boolean;
  analyzer: "word" | "char";

  vocabulary_: Map<string, number> | null = null;
  featureNames_: string[] | null = null;

  constructor(options: CountVectorizerOptions = {}) {
    this.minDf = options.minDf ?? 1;
    this.maxDf = options.maxDf ?? 1.0;
    this.maxFeatures = options.maxFeatures ?? null;
    this.ngramRange = options.ngramRange ?? [1, 1];
    this.lowercase = options.lowercase ?? true;
    this.analyzer = options.analyzer ?? "word";
  }

  private _analyze(doc: string): string[] {
    const text = this.lowercase ? doc.toLowerCase() : doc;
    const tokens = this.analyzer === "word"
      ? (text.match(/\b[a-z0-9]+\b/g) ?? [])
      : Array.from(text);
    const [minN, maxN] = this.ngramRange;
    if (minN === 1 && maxN === 1) return tokens;
    const ngrams: string[] = [];
    for (let n = minN; n <= maxN; n++) {
      for (let i = 0; i <= tokens.length - n; i++) {
        ngrams.push(tokens.slice(i, i + n).join(" "));
      }
    }
    return ngrams;
  }

  fit(docs: string[]): this {
    const termDocFreq = new Map<string, number>();
    const n = docs.length;
    for (const doc of docs) {
      const seen = new Set<string>();
      for (const term of this._analyze(doc)) {
        if (!seen.has(term)) {
          seen.add(term);
          termDocFreq.set(term, (termDocFreq.get(term) ?? 0) + 1);
        }
      }
    }
    const minDfAbs = this.minDf < 1 ? Math.floor(this.minDf * n) : this.minDf;
    const maxDfAbs = this.maxDf <= 1.0 ? Math.ceil(this.maxDf * n) : this.maxDf;
    let terms = [...termDocFreq.entries()]
      .filter(([, df]) => df >= minDfAbs && df <= maxDfAbs)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([t]) => t);
    if (this.maxFeatures !== null) {
      terms = terms.slice(0, this.maxFeatures);
    }
    this.vocabulary_ = new Map(terms.map((t, i) => [t, i]));
    this.featureNames_ = terms;
    return this;
  }

  transform(docs: string[]): Float64Array[] {
    if (this.vocabulary_ === null) throw new NotFittedError();
    const vocab = this.vocabulary_;
    const nFeatures = vocab.size;
    return docs.map((doc) => {
      const row = new Float64Array(nFeatures);
      for (const term of this._analyze(doc)) {
        const idx = vocab.get(term);
        if (idx !== undefined) row[idx] = (row[idx] ?? 0) + 1;
      }
      return row;
    });
  }

  fitTransform(docs: string[]): Float64Array[] {
    return this.fit(docs).transform(docs);
  }

  getFeatureNames(): string[] {
    if (this.featureNames_ === null) throw new NotFittedError();
    return this.featureNames_;
  }
}

/**
 * Transform a count matrix to a normalized TF or TF-IDF representation.
 * Mirrors sklearn.feature_extraction.text.TfidfTransformer.
 */
export class TfidfTransformer {
  norm: "l1" | "l2" | null;
  useIdf: boolean;
  smoothIdf: boolean;
  sublinearTf: boolean;

  idf_: Float64Array | null = null;

  constructor(options: TfidfTransformerOptions = {}) {
    this.norm = options.norm ?? "l2";
    this.useIdf = options.useIdf ?? true;
    this.smoothIdf = options.smoothIdf ?? true;
    this.sublinearTf = options.sublinearTf ?? false;
  }

  fit(X: Float64Array[]): this {
    if (!this.useIdf) {
      this.idf_ = null;
      return this;
    }
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const df = new Float64Array(p);
    for (const row of X) {
      for (let j = 0; j < p; j++) {
        if ((row[j] ?? 0) > 0) df[j] = (df[j] ?? 0) + 1;
      }
    }
    const smooth = this.smoothIdf ? 1 : 0;
    this.idf_ = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      this.idf_[j] = Math.log((n + smooth) / ((df[j] ?? 0) + smooth)) + 1;
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const p = (X[0] ?? new Float64Array(0)).length;
    return X.map((row) => {
      const out = new Float64Array(p);
      for (let j = 0; j < p; j++) {
        let tf = row[j] ?? 0;
        if (this.sublinearTf && tf > 0) tf = 1 + Math.log(tf);
        const idfVal = this.idf_ !== null ? (this.idf_[j] ?? 1) : 1;
        out[j] = tf * idfVal;
      }
      if (this.norm === "l2") {
        let norm = 0;
        for (let j = 0; j < p; j++) norm += (out[j] ?? 0) ** 2;
        norm = Math.sqrt(norm);
        if (norm > 0) for (let j = 0; j < p; j++) out[j] = (out[j] ?? 0) / norm;
      } else if (this.norm === "l1") {
        let norm = 0;
        for (let j = 0; j < p; j++) norm += Math.abs(out[j] ?? 0);
        if (norm > 0) for (let j = 0; j < p; j++) out[j] = (out[j] ?? 0) / norm;
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

/**
 * Convert a collection of raw documents to a matrix of TF-IDF features.
 * Mirrors sklearn.feature_extraction.text.TfidfVectorizer.
 */
export class TfidfVectorizer {
  private cv: CountVectorizer;
  private tfidf: TfidfTransformer;

  vocabulary_: Map<string, number> | null = null;
  featureNames_: string[] | null = null;

  constructor(
    cvOptions: CountVectorizerOptions = {},
    tfidfOptions: TfidfTransformerOptions = {},
  ) {
    this.cv = new CountVectorizer(cvOptions);
    this.tfidf = new TfidfTransformer(tfidfOptions);
  }

  fit(docs: string[]): this {
    const counts = this.cv.fit(docs).transform(docs);
    this.tfidf.fit(counts);
    this.vocabulary_ = this.cv.vocabulary_;
    this.featureNames_ = this.cv.featureNames_;
    return this;
  }

  transform(docs: string[]): Float64Array[] {
    const counts = this.cv.transform(docs);
    return this.tfidf.transform(counts);
  }

  fitTransform(docs: string[]): Float64Array[] {
    return this.fit(docs).transform(docs);
  }

  getFeatureNames(): string[] {
    if (this.featureNames_ === null) throw new NotFittedError();
    return this.featureNames_;
  }
}

/**
 * Convert a collection of text documents to a matrix of token occurrences using a hash trick.
 * Mirrors sklearn.feature_extraction.text.HashingVectorizer.
 */
export class HashingVectorizer {
  nFeatures: number;
  alternateSign: boolean;
  lowercase: boolean;
  ngramRange: [number, number];

  constructor(options: HashingVectorizerOptions = {}) {
    this.nFeatures = options.nFeatures ?? 2 ** 20;
    this.alternateSign = options.alternate_sign ?? true;
    this.lowercase = options.lowercase ?? true;
    this.ngramRange = options.ngramRange ?? [1, 1];
  }

  private _analyze(doc: string): string[] {
    const text = this.lowercase ? doc.toLowerCase() : doc;
    const tokens = text.match(/\b[a-z0-9]+\b/g) ?? [];
    const [minN, maxN] = this.ngramRange;
    if (minN === 1 && maxN === 1) return tokens;
    const ngrams: string[] = [];
    for (let n = minN; n <= maxN; n++) {
      for (let i = 0; i <= tokens.length - n; i++) {
        ngrams.push(tokens.slice(i, i + n).join(" "));
      }
    }
    return ngrams;
  }

  transform(docs: string[]): Float64Array[] {
    return docs.map((doc) => {
      const row = new Float64Array(this.nFeatures);
      for (const term of this._analyze(doc)) {
        const h = murmurhash(term);
        const idx = h % this.nFeatures;
        const sign = this.alternateSign ? (h & 1 ? 1 : -1) : 1;
        row[idx] = (row[idx] ?? 0) + sign;
      }
      return row;
    });
  }
}

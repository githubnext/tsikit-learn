/**
 * Advanced feature extraction: hashing vectorizer, TF-IDF extensions, image feature extraction.
 */

function murmurhash3(key: string): number {
  let h = 0xdeadbeef;
  for (let i = 0; i < key.length; i++) {
    const c = key.charCodeAt(i);
    h = Math.imul(h ^ c, 0x85ebca6b);
    h ^= h >>> 13;
  }
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
  h ^= h >>> 13;
  return (h >>> 0) % 0x80000000;
}

export class HashingVectorizer {
  constructor(private nFeatures = 1024, private ngram: [number, number] = [1, 1]) {}

  transform(documents: string[]): Float64Array[] {
    return documents.map(doc => {
      const tokens = doc.toLowerCase().split(/\s+/).filter(t => t.length > 0);
      const vec = new Float64Array(this.nFeatures);
      const ngrams = this._getNgrams(tokens);
      for (const ng of ngrams) {
        const h = murmurhash3(ng) % this.nFeatures;
        const sign = murmurhash3(`s${ng}`) % 2 === 0 ? 1 : -1;
        vec[h] = (vec[h] ?? 0) + sign;
      }
      return vec;
    });
  }

  private _getNgrams(tokens: string[]): string[] {
    const ngrams: string[] = [];
    for (let n = this.ngram[0]; n <= this.ngram[1]; n++) {
      for (let i = 0; i <= tokens.length - n; i++) {
        ngrams.push(tokens.slice(i, i + n).join(' '));
      }
    }
    return ngrams;
  }
}

export class TfIdfVectorizerExt {
  private vocabulary_!: Map<string, number>;
  private idf_!: Float64Array;
  private fitted_ = false;

  constructor(
    private maxFeatures = 1000,
    private minDf = 1,
    private maxDf = 1.0,
    private sublinearTf = false,
    private ngram: [number, number] = [1, 1]
  ) {}

  fit(documents: string[]): this {
    const n = documents.length;
    const tokenizedDocs = documents.map(d => this._tokenize(d));
    // Count document frequencies
    const dfCounts = new Map<string, number>();
    for (const tokens of tokenizedDocs) {
      for (const t of new Set(tokens)) dfCounts.set(t, (dfCounts.get(t) ?? 0) + 1);
    }
    // Filter by df thresholds and select top features by df
    const minDfAbs = typeof this.minDf === 'number' && this.minDf < 1 ? Math.ceil(this.minDf * n) : this.minDf;
    const maxDfAbs = typeof this.maxDf === 'number' && this.maxDf <= 1 ? Math.floor(this.maxDf * n) : this.maxDf;
    const filtered = Array.from(dfCounts.entries())
      .filter(([, df]) => df >= minDfAbs && df <= maxDfAbs)
      .sort((a, b) => b[1] - a[1])
      .slice(0, this.maxFeatures);
    this.vocabulary_ = new Map(filtered.map(([t], idx) => [t, idx]));
    this.idf_ = new Float64Array(this.vocabulary_.size).map((_, idx) => {
      const term = filtered[idx]![0]!;
      const df = dfCounts.get(term) ?? 1;
      return Math.log((n + 1) / (df + 1)) + 1;
    });
    this.fitted_ = true;
    void minDfAbs; void maxDfAbs;
    return this;
  }

  transform(documents: string[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return documents.map(doc => {
      const tokens = this._tokenize(doc);
      const tf = new Float64Array(this.vocabulary_.size);
      for (const t of tokens) {
        const idx = this.vocabulary_.get(t);
        if (idx !== undefined) tf[idx] = (tf[idx] ?? 0) + 1;
      }
      const vec = new Float64Array(tf.map((v, j) => (this.sublinearTf ? (v > 0 ? 1 + Math.log(v) : 0) : v) * (this.idf_[j] ?? 0)));
      const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
      return norm > 0 ? new Float64Array(vec.map(v => v / norm)) : vec;
    });
  }

  fitTransform(documents: string[]): Float64Array[] {
    return this.fit(documents).transform(documents);
  }

  private _tokenize(doc: string): string[] {
    const tokens = doc.toLowerCase().split(/\s+/).filter(t => t.length > 0);
    const ngrams: string[] = [];
    for (let n = this.ngram[0]; n <= this.ngram[1]; n++) {
      for (let i = 0; i <= tokens.length - n; i++) ngrams.push(tokens.slice(i, i + n).join('_'));
    }
    return ngrams;
  }

  get vocabulary(): Map<string, number> { return this.vocabulary_; }
}

export function extractImagePatches(
  image: Float64Array[],
  patchSize: [number, number] = [8, 8],
  stride = 4
): Float64Array[] {
  const [pH, pW] = patchSize;
  const nRows = image.length, nCols = image[0]?.length ?? 0;
  const patches: Float64Array[] = [];
  for (let r = 0; r + pH <= nRows; r += stride) {
    for (let c = 0; c + pW <= nCols; c += stride) {
      const patch = new Float64Array(pH * pW);
      for (let dr = 0; dr < pH; dr++) {
        for (let dc = 0; dc < pW; dc++) {
          patch[dr * pW + dc] = image[r + dr]![c + dc] ?? 0;
        }
      }
      patches.push(patch);
    }
  }
  return patches;
}

export function hogFeatures(image: Float64Array[], cellSize = 8, nBins = 9): Float64Array {
  const height = image.length, width = image[0]?.length ?? 0;
  const nCellsH = Math.floor(height / cellSize), nCellsW = Math.floor(width / cellSize);
  const histograms: Float64Array[] = [];
  for (let cy = 0; cy < nCellsH; cy++) {
    for (let cx = 0; cx < nCellsW; cx++) {
      const hist = new Float64Array(nBins);
      for (let dy = 0; dy < cellSize; dy++) {
        for (let dx = 0; dx < cellSize; dx++) {
          const y = cy * cellSize + dy, x = cx * cellSize + dx;
          const gx = (image[y]![(x + 1) < width ? x + 1 : x] ?? 0) - (image[y]![x > 0 ? x - 1 : 0] ?? 0);
          const gy = (image[(y + 1) < height ? y + 1 : y]![x] ?? 0) - (image[y > 0 ? y - 1 : 0]![x] ?? 0);
          const mag = Math.sqrt(gx ** 2 + gy ** 2);
          const angle = (Math.atan2(gy, gx) + Math.PI) / Math.PI * nBins;
          const bin = Math.floor(angle) % nBins;
          hist[bin] = (hist[bin] ?? 0) + mag;
        }
      }
      histograms.push(hist);
    }
  }
  const features = new Float64Array(histograms.flatMap(h => Array.from(h)));
  const norm = Math.sqrt(features.reduce((s, v) => s + v * v, 0));
  return norm > 0 ? new Float64Array(features.map(v => v / norm)) : features;
}

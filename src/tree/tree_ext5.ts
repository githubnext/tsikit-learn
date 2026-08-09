/**
 * Tree extensions: ObliqueDecisionTree, ExtraObliqueTree, RandomPatches
 * Port of sklearn.tree extensions
 */

import { NotFittedError } from "../exceptions.js";

interface ObliqueNode {
  isLeaf: boolean;
  weights?: Float64Array;
  threshold?: number;
  left?: ObliqueNode;
  right?: ObliqueNode;
  value?: number;
  classProbs?: Float64Array;
}

function giniImpurityArr(labels: number[], nClasses: number): number {
  const counts = new Int32Array(nClasses);
  for (const l of labels) counts[Math.min(l, nClasses - 1)] = (counts[Math.min(l, nClasses - 1)] ?? 0) + 1;
  let g = 1;
  const n = labels.length;
  for (let k = 0; k < nClasses; k++) g -= ((counts[k] ?? 0) / n) ** 2;
  return g;
}

function buildObliqueTree(
  X: Float64Array[],
  y: Int32Array,
  nClasses: number,
  depth: number,
  maxDepth: number,
  rng: () => number,
  nOblique: number
): ObliqueNode {
  const n = X.length;
  if (n === 0) return { isLeaf: true, value: 0, classProbs: new Float64Array(nClasses) };
  const counts = new Int32Array(nClasses);
  for (let i = 0; i < n; i++) counts[y[i] ?? 0] = (counts[y[i] ?? 0] ?? 0) + 1;
  let majority = 0;
  for (let k = 0; k < nClasses; k++) if ((counts[k] ?? 0) > (counts[majority] ?? 0)) majority = k;
  const classProbs = Float64Array.from({ length: nClasses }, (_, k) => (counts[k] ?? 0) / n);
  if (depth >= maxDepth || n <= 1 || (counts[majority] ?? 0) === n) {
    return { isLeaf: true, value: majority, classProbs };
  }
  const p = X[0]?.length ?? 0;
  let bestGini = Number.POSITIVE_INFINITY;
  let bestWeights: Float64Array = new Float64Array(p);
  let bestThreshold = 0;
  for (let t = 0; t < nOblique; t++) {
    const weights = new Float64Array(p);
    const nNonZero = Math.max(1, Math.floor(rng() * Math.min(5, p)));
    for (let k = 0; k < nNonZero; k++) {
      const j = Math.floor(rng() * p);
      weights[j] = rng() * 2 - 1;
    }
    const projections = X.map(xi => xi.reduce((s, v, j) => s + (v ?? 0) * (weights[j] ?? 0), 0));
    projections.sort((a, b) => a - b);
    const candidates = projections.slice(1).map((v, i) => (v + (projections[i] ?? 0)) / 2);
    for (const thresh of candidates.slice(0, 10)) {
      const leftY: number[] = [];
      const rightY: number[] = [];
      for (let i = 0; i < n; i++) {
        const proj = X[i]!.reduce((s, v, j) => s + (v ?? 0) * (weights[j] ?? 0), 0);
        if (proj <= thresh) leftY.push(y[i] ?? 0);
        else rightY.push(y[i] ?? 0);
      }
      if (leftY.length === 0 || rightY.length === 0) continue;
      const gini = (leftY.length / n) * giniImpurityArr(leftY, nClasses) + (rightY.length / n) * giniImpurityArr(rightY, nClasses);
      if (gini < bestGini) { bestGini = gini; bestWeights = weights.slice(); bestThreshold = thresh; }
    }
  }
  const leftX: Float64Array[] = [];
  const leftY: number[] = [];
  const rightX: Float64Array[] = [];
  const rightY: number[] = [];
  for (let i = 0; i < n; i++) {
    const proj = X[i]!.reduce((s, v, j) => s + (v ?? 0) * (bestWeights[j] ?? 0), 0);
    if (proj <= bestThreshold) { leftX.push(X[i]!); leftY.push(y[i] ?? 0); }
    else { rightX.push(X[i]!); rightY.push(y[i] ?? 0); }
  }
  if (leftX.length === 0 || rightX.length === 0) return { isLeaf: true, value: majority, classProbs };
  return {
    isLeaf: false,
    weights: bestWeights,
    threshold: bestThreshold,
    classProbs,
    left: buildObliqueTree(leftX, Int32Array.from(leftY), nClasses, depth + 1, maxDepth, rng, nOblique),
    right: buildObliqueTree(rightX, Int32Array.from(rightY), nClasses, depth + 1, maxDepth, rng, nOblique),
  };
}

function predictObliqueNode(node: ObliqueNode, x: Float64Array): number {
  if (node.isLeaf) return node.value ?? 0;
  const proj = (node.weights ?? new Float64Array(0)).reduce((s, w, j) => s + (w ?? 0) * (x[j] ?? 0), 0);
  return proj <= (node.threshold ?? 0) ? predictObliqueNode(node.left!, x) : predictObliqueNode(node.right!, x);
}

function predictObliqueProbas(node: ObliqueNode, x: Float64Array): Float64Array {
  if (node.isLeaf) return node.classProbs ?? new Float64Array(0);
  const proj = (node.weights ?? new Float64Array(0)).reduce((s, w, j) => s + (w ?? 0) * (x[j] ?? 0), 0);
  return proj <= (node.threshold ?? 0) ? predictObliqueProbas(node.left!, x) : predictObliqueProbas(node.right!, x);
}

export class ObliqueDecisionTreeClassifier {
  maxDepth: number;
  nOblique: number;
  randomState: number;

  private root_: ObliqueNode | null = null;
  classes_: Int32Array | null = null;

  constructor(opts: { maxDepth?: number; nOblique?: number; randomState?: number } = {}) {
    this.maxDepth = opts.maxDepth ?? 5;
    this.nOblique = opts.nOblique ?? 10;
    this.randomState = opts.randomState ?? 42;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const classSet = new Set<number>();
    for (let i = 0; i < y.length; i++) classSet.add(y[i] ?? 0);
    this.classes_ = Int32Array.from([...classSet].sort((a, b) => a - b));
    const nClasses = this.classes_.length;
    const classMap = new Map(Array.from(this.classes_).map((c, i) => [c, i]));
    const yMapped = Int32Array.from(y.map(yi => classMap.get(yi ?? 0) ?? 0));
    let seed = this.randomState;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    this.root_ = buildObliqueTree(X, yMapped, nClasses, 0, this.maxDepth, rng, this.nOblique);
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.root_ || !this.classes_) throw new NotFittedError("ObliqueDecisionTreeClassifier not fitted.");
    return Int32Array.from(X.map(xi => this.classes_![predictObliqueNode(this.root_!, xi)] ?? 0));
  }

  predictProba(X: Float64Array[]): Float64Array[] {
    if (!this.root_) throw new NotFittedError("ObliqueDecisionTreeClassifier not fitted.");
    return X.map(xi => predictObliqueProbas(this.root_!, xi));
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) correct++;
    return correct / y.length;
  }
}

export class PatchExtractor {
  patchSize: [number, number];
  maxPatches: number;
  randomState: number;

  constructor(opts: { patchSize?: [number, number]; maxPatches?: number; randomState?: number } = {}) {
    this.patchSize = opts.patchSize ?? [8, 8];
    this.maxPatches = opts.maxPatches ?? 50;
    this.randomState = opts.randomState ?? 0;
  }

  transform(images: Float64Array[], imageShape: [number, number]): Float64Array[] {
    const [rows, cols] = imageShape;
    const [ph, pw] = this.patchSize;
    const result: Float64Array[] = [];
    let seed = this.randomState;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    for (const img of images) {
      const nPatches = this.maxPatches;
      for (let p = 0; p < nPatches; p++) {
        const r = Math.floor(rng() * (rows - ph + 1));
        const c = Math.floor(rng() * (cols - pw + 1));
        const patch = new Float64Array(ph * pw);
        for (let i = 0; i < ph; i++) for (let j = 0; j < pw; j++) {
          patch[i * pw + j] = img[(r + i) * cols + c + j] ?? 0;
        }
        result.push(patch);
      }
    }
    return result;
  }
}

export class RandomPatchesClassifier {
  nEstimators: number;
  maxSamples: number;
  maxFeatures: number;
  maxDepth: number;
  randomState: number;

  private estimators_: ObliqueDecisionTreeClassifier[] | null = null;
  private featureSets_: number[][] | null = null;
  private sampleSets_: number[][] | null = null;
  classes_: Int32Array | null = null;

  constructor(opts: { nEstimators?: number; maxSamples?: number; maxFeatures?: number; maxDepth?: number; randomState?: number } = {}) {
    this.nEstimators = opts.nEstimators ?? 10;
    this.maxSamples = opts.maxSamples ?? 0.5;
    this.maxFeatures = opts.maxFeatures ?? 0.5;
    this.maxDepth = opts.maxDepth ?? 5;
    this.randomState = opts.randomState ?? 0;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const classSet = new Set<number>();
    for (let i = 0; i < y.length; i++) classSet.add(y[i] ?? 0);
    this.classes_ = Int32Array.from([...classSet].sort((a, b) => a - b));
    let seed = this.randomState;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    this.estimators_ = [];
    this.featureSets_ = [];
    this.sampleSets_ = [];
    for (let t = 0; t < this.nEstimators; t++) {
      const nSamp = Math.max(1, Math.floor(n * this.maxSamples));
      const nFeat = Math.max(1, Math.floor(p * this.maxFeatures));
      const samples = Array.from({ length: nSamp }, () => Math.floor(rng() * n));
      const features = Array.from({ length: p }, (_, i) => i);
      for (let i = features.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); const tmp = features[i]!; features[i] = features[j]!; features[j] = tmp; }
      const selectedFeats = features.slice(0, nFeat);
      const subX = samples.map(i => Float64Array.from(selectedFeats.map(f => X[i]![f] ?? 0)));
      const subY = Int32Array.from(samples.map(i => y[i] ?? 0));
      const est = new ObliqueDecisionTreeClassifier({ maxDepth: this.maxDepth, randomState: seed + t });
      est.fit(subX, subY);
      this.estimators_.push(est);
      this.featureSets_.push(selectedFeats);
      this.sampleSets_.push(samples);
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.estimators_ || !this.classes_ || !this.featureSets_) throw new NotFittedError("RandomPatchesClassifier not fitted.");
    const nClasses = this.classes_.length;
    return Int32Array.from(X.map(xi => {
      const votes = new Float64Array(nClasses);
      for (let t = 0; t < this.estimators_!.length; t++) {
        const subXi = Float64Array.from(this.featureSets_![t]!.map(f => xi[f] ?? 0));
        const proba = this.estimators_![t]!.predictProba([subXi])[0];
        if (proba) for (let k = 0; k < Math.min(nClasses, proba.length); k++) votes[k] = (votes[k] ?? 0) + (proba[k] ?? 0);
      }
      let bestK = 0;
      for (let k = 1; k < nClasses; k++) if ((votes[k] ?? 0) > (votes[bestK] ?? 0)) bestK = k;
      return this.classes_![bestK] ?? 0;
    }));
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) correct++;
    return correct / y.length;
  }
}

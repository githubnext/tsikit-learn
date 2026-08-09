/**
 * Ensemble extensions: RotationForestClassifier, ExtraTreesEnsemble, AdaptiveBoosting
 * Port of sklearn-compatible ensemble methods
 */

import { NotFittedError } from "../exceptions.js";

function pca2d(X: Float64Array[]): { components: Float64Array[]; mean: Float64Array } {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  const mean = new Float64Array(p);
  for (const xi of X) for (let j = 0; j < p; j++) mean[j] = (mean[j] ?? 0) + (xi[j] ?? 0) / n;
  const centered = X.map(xi => { const r = new Float64Array(p); for (let j = 0; j < p; j++) r[j] = (xi[j] ?? 0) - (mean[j] ?? 0); return r; });
  const cov = Array.from({ length: p }, () => new Float64Array(p));
  for (const xi of centered) {
    for (let j = 0; j < p; j++) for (let k = 0; k < p; k++) {
      cov[j]![k] = (cov[j]![k] ?? 0) + (xi[j] ?? 0) * (xi[k] ?? 0) / n;
    }
  }
  const components: Float64Array[] = Array.from({ length: Math.min(p, 2) }, (_, i) => {
    const v = new Float64Array(p);
    v[i % p] = 1;
    for (let iter = 0; iter < 50; iter++) {
      const nv = new Float64Array(p);
      for (let j = 0; j < p; j++) for (let k = 0; k < p; k++) nv[j] = (nv[j] ?? 0) + (cov[j]![k] ?? 0) * (v[k] ?? 0);
      let norm = 0;
      for (let j = 0; j < p; j++) norm += (nv[j] ?? 0) ** 2;
      norm = Math.sqrt(norm) + 1e-15;
      for (let j = 0; j < p; j++) v[j] = (nv[j] ?? 0) / norm;
    }
    return v;
  });
  return { components, mean };
}

interface SimpleTreeNode {
  feature?: number;
  threshold?: number;
  left?: SimpleTreeNode;
  right?: SimpleTreeNode;
  value?: number;
  isLeaf: boolean;
}

function buildTree(X: Float64Array[], y: Int32Array, depth: number, maxDepth: number, rng: () => number): SimpleTreeNode {
  const n = X.length;
  if (n === 0) return { isLeaf: true, value: 0 };
  const counts: Record<number, number> = {};
  for (let i = 0; i < n; i++) counts[y[i] ?? 0] = (counts[y[i] ?? 0] ?? 0) + 1;
  let majorityClass = 0;
  let maxCount = 0;
  for (const [k, v] of Object.entries(counts)) {
    if (v > maxCount) { maxCount = v; majorityClass = Number(k); }
  }
  if (depth >= maxDepth || n <= 1 || maxCount === n) return { isLeaf: true, value: majorityClass };
  const p = X[0]?.length ?? 0;
  let bestGini = Number.POSITIVE_INFINITY;
  let bestFeat = 0;
  let bestThresh = 0;
  const featsToTry = Math.max(1, Math.floor(Math.sqrt(p)));
  const feats = Array.from({ length: featsToTry }, () => Math.floor(rng() * p));
  for (const f of feats) {
    const vals = X.map(xi => xi[f] ?? 0).sort((a, b) => a - b);
    for (let t = 0; t < vals.length - 1; t++) {
      const thresh = ((vals[t] ?? 0) + (vals[t + 1] ?? 0)) / 2;
      const leftY: number[] = [];
      const rightY: number[] = [];
      for (let i = 0; i < n; i++) {
        if ((X[i]![f] ?? 0) <= thresh) leftY.push(y[i] ?? 0);
        else rightY.push(y[i] ?? 0);
      }
      const gini = (leftY.length / n) * giniImpurity(leftY) + (rightY.length / n) * giniImpurity(rightY);
      if (gini < bestGini) { bestGini = gini; bestFeat = f; bestThresh = thresh; }
    }
  }
  const leftX: Float64Array[] = [];
  const leftY: number[] = [];
  const rightX: Float64Array[] = [];
  const rightY: number[] = [];
  for (let i = 0; i < n; i++) {
    if ((X[i]![bestFeat] ?? 0) <= bestThresh) { leftX.push(X[i]!); leftY.push(y[i] ?? 0); }
    else { rightX.push(X[i]!); rightY.push(y[i] ?? 0); }
  }
  if (leftX.length === 0 || rightX.length === 0) return { isLeaf: true, value: majorityClass };
  return {
    isLeaf: false,
    feature: bestFeat,
    threshold: bestThresh,
    left: buildTree(leftX, Int32Array.from(leftY), depth + 1, maxDepth, rng),
    right: buildTree(rightX, Int32Array.from(rightY), depth + 1, maxDepth, rng),
  };
}

function giniImpurity(y: number[]): number {
  if (y.length === 0) return 0;
  const counts: Record<number, number> = {};
  for (const v of y) counts[v] = (counts[v] ?? 0) + 1;
  let g = 1;
  for (const v of Object.values(counts)) g -= (v / y.length) ** 2;
  return g;
}

function predictTree(node: SimpleTreeNode, x: Float64Array): number {
  if (node.isLeaf) return node.value ?? 0;
  const f = node.feature ?? 0;
  if ((x[f] ?? 0) <= (node.threshold ?? 0)) return predictTree(node.left!, x);
  return predictTree(node.right!, x);
}

export class RotationForestClassifier {
  nEstimators: number;
  maxDepth: number;
  nGroups: number;
  randomState: number;

  private trees_: SimpleTreeNode[] | null = null;
  private rotations_: Array<{ components: Float64Array[]; mean: Float64Array; feats: number[] }> | null = null;
  private classes_: Int32Array | null = null;

  constructor(opts: { nEstimators?: number; maxDepth?: number; nGroups?: number; randomState?: number } = {}) {
    this.nEstimators = opts.nEstimators ?? 10;
    this.maxDepth = opts.maxDepth ?? 5;
    this.nGroups = opts.nGroups ?? 3;
    this.randomState = opts.randomState ?? 42;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const classSet = new Set<number>();
    for (let i = 0; i < n; i++) classSet.add(y[i] ?? 0);
    this.classes_ = Int32Array.from([...classSet].sort((a, b) => a - b));
    let seed = this.randomState;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    this.trees_ = [];
    this.rotations_ = [];
    for (let t = 0; t < this.nEstimators; t++) {
      const groupSize = Math.max(1, Math.floor(p / this.nGroups));
      const feats = Array.from({ length: p }, (_, i) => i);
      for (let i = feats.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); const tmp = feats[i]!; feats[i] = feats[j]!; feats[j] = tmp; }
      const allComponents: Float64Array[] = [];
      const usedFeats: number[] = [];
      for (let g = 0; g < this.nGroups; g++) {
        const gFeats = feats.slice(g * groupSize, (g + 1) * groupSize);
        const subX = X.map(xi => { const r = new Float64Array(gFeats.length); for (let j = 0; j < gFeats.length; j++) r[j] = xi[gFeats[j]!] ?? 0; return r; });
        const { components, mean } = pca2d(subX);
        for (const comp of components) { allComponents.push(comp); usedFeats.push(...gFeats); }
        void mean;
      }
      const { mean } = pca2d(X);
      this.rotations_.push({ components: allComponents, mean, feats: usedFeats });
      const rotX = X.map(xi => {
        const r = new Float64Array(allComponents.length);
        for (let j = 0; j < allComponents.length; j++) {
          const comp = allComponents[j];
          if (!comp) continue;
          const featsForComp = usedFeats.slice(0, comp.length);
          let val = 0;
          for (let k = 0; k < featsForComp.length; k++) val += (comp[k] ?? 0) * ((xi[featsForComp[k]!] ?? 0) - (mean[featsForComp[k]!] ?? 0));
          r[j] = val;
        }
        return r;
      });
      const bootIdx = Array.from({ length: n }, () => Math.floor(rng() * n));
      const bootX = bootIdx.map(i => rotX[i]!);
      const bootY = Int32Array.from(bootIdx.map(i => y[i] ?? 0));
      this.trees_.push(buildTree(bootX, bootY, 0, this.maxDepth, rng));
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.trees_ || !this.rotations_ || !this.classes_) throw new NotFittedError("RotationForestClassifier not fitted.");
    const labels = new Int32Array(X.length);
    for (let i = 0; i < X.length; i++) {
      const xi = X[i]!;
      const votes: Record<number, number> = {};
      for (let t = 0; t < this.trees_.length; t++) {
        const rot = this.rotations_[t]!;
        const rotX = new Float64Array(rot.components.length);
        for (let j = 0; j < rot.components.length; j++) {
          const comp = rot.components[j];
          if (!comp) continue;
          const featsForComp = rot.feats.slice(0, comp.length);
          let val = 0;
          for (let k = 0; k < featsForComp.length; k++) val += (comp[k] ?? 0) * ((xi[featsForComp[k]!] ?? 0) - (rot.mean[featsForComp[k]!] ?? 0));
          rotX[j] = val;
        }
        const pred = predictTree(this.trees_[t]!, rotX);
        votes[pred] = (votes[pred] ?? 0) + 1;
      }
      let bestClass = 0;
      let bestVotes = -1;
      for (const [k, v] of Object.entries(votes)) {
        if (v > bestVotes) { bestVotes = v; bestClass = Number(k); }
      }
      labels[i] = bestClass;
    }
    return labels;
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) correct++;
    return correct / y.length;
  }
}

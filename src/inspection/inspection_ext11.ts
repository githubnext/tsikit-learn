/**
 * Inspection extensions: Explainability, SHAP approximations, ICE plots
 */

export interface ExplanationResult {
  shapValues: Float64Array[];
  baseValue: number;
  feature: number;
}

export function kernelSHAPExt(
  model: { predict: (X: Float64Array[]) => Float64Array },
  X: Float64Array[],
  background: Float64Array[],
  nSamples: number = 100,
  randomState: number = 42
): Float64Array[] {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  let rng = randomState;
  const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 0xffffffff; };

  const basePred = model.predict(background);
  const baseValue = basePred.reduce((s, v) => s + v, 0) / basePred.length;

  return X.map(instance => {
    const shapValues = new Float64Array(p);
    // Sample random coalitions
    for (let s = 0; s < nSamples; s++) {
      const mask = new Uint8Array(p).map(() => rand() > 0.5 ? 1 : 0);
      const bgIdx = Math.floor(rand() * background.length);
      const maskedWith = new Float64Array(p).map((_, j) => (mask[j] ?? 0) ? (instance[j] ?? 0) : (background[bgIdx]?.[j] ?? 0));
      const pred = model.predict([maskedWith])[0] ?? 0;

      // Weight by kernel
      const sz = mask.reduce((ss, v) => ss + v, 0);
      const weight = sz > 0 && sz < p ? 1 / (sz * (p - sz)) : 0.01;

      for (let j = 0; j < p; j++) {
        if (mask[j]) shapValues[j] = (shapValues[j] ?? 0) + weight * (pred - baseValue) / nSamples;
      }
    }
    return shapValues;
  });
}

export function treeSHAPApprox(
  featureImportances: Float64Array,
  predictions: Float64Array,
  X: Float64Array[],
  baseValue: number
): Float64Array[] {
  const n = X.length, p = X[0]?.length ?? 0;
  const totalImportance = featureImportances.reduce((s, v) => s + v, 0) + 1e-10;
  const normalizedImp = featureImportances.map(v => v / totalImportance);

  return X.map((row, i) => {
    const delta = (predictions[i] ?? 0) - baseValue;
    return new Float64Array(p).map((_, j) => delta * (normalizedImp[j] ?? 0));
  });
}

export function individualConditionalExpectation(
  model: { predict: (X: Float64Array[]) => Float64Array },
  X: Float64Array[],
  featureIdx: number,
  gridPoints: number = 50
): { gridValues: Float64Array; iceLines: Float64Array[] } {
  const vals = X.map(row => row[featureIdx] ?? 0);
  const minV = Math.min(...vals), maxV = Math.max(...vals);
  const gridValues = new Float64Array(gridPoints).map((_, i) => minV + (i / (gridPoints - 1)) * (maxV - minV));

  const iceLines = X.map(row => {
    const preds = model.predict(Array.from(gridValues, v => {
      const r = row.slice();
      r[featureIdx] = v;
      return r;
    }));
    return preds;
  });

  return { gridValues, iceLines };
}

export function partialDependencePlot(
  model: { predict: (X: Float64Array[]) => Float64Array },
  X: Float64Array[],
  featureIdx: number,
  gridPoints: number = 50
): { gridValues: Float64Array; averagePredictions: Float64Array } {
  const { gridValues, iceLines } = individualConditionalExpectation(model, X, featureIdx, gridPoints);
  const averagePredictions = new Float64Array(gridPoints).map((_, g) => iceLines.reduce((s, line) => s + (line[g] ?? 0), 0) / iceLines.length);
  return { gridValues, averagePredictions };
}

export function partialDependence2D(
  model: { predict: (X: Float64Array[]) => Float64Array },
  X: Float64Array[],
  features: [number, number],
  gridPoints: number = 20
): { grid0: Float64Array; grid1: Float64Array; averagePredictions: Float64Array[] } {
  const [f0, f1] = features;
  const vals0 = X.map(row => row[f0] ?? 0);
  const vals1 = X.map(row => row[f1] ?? 0);
  const [min0, max0] = [Math.min(...vals0), Math.max(...vals0)];
  const [min1, max1] = [Math.min(...vals1), Math.max(...vals1)];
  const grid0 = new Float64Array(gridPoints).map((_, i) => min0 + (i / (gridPoints - 1)) * (max0 - min0));
  const grid1 = new Float64Array(gridPoints).map((_, i) => min1 + (i / (gridPoints - 1)) * (max1 - min1));

  const averagePredictions = Array.from({ length: gridPoints }, (_, i) =>
    new Float64Array(gridPoints).map((_, j) => {
      const preds = model.predict(X.map(row => {
        const r = row.slice();
        r[f0] = grid0[i] ?? 0;
        r[f1] = grid1[j] ?? 0;
        return r;
      }));
      return preds.reduce((s, v) => s + v, 0) / preds.length;
    })
  );
  return { grid0, grid1, averagePredictions };
}

export function hStatisticExt(
  model: { predict: (X: Float64Array[]) => Float64Array },
  X: Float64Array[],
  features: [number, number],
  gridPoints: number = 20
): number {
  const [f0, f1] = features;
  const { grid0, grid1, averagePredictions: pdp2d } = partialDependence2D(model, X, [f0, f1], gridPoints);
  const pdp0 = partialDependencePlot(model, X, f0, gridPoints);
  const pdp1 = partialDependencePlot(model, X, f1, gridPoints);

  let numerator = 0, denominator = 0;
  for (let i = 0; i < gridPoints; i++) {
    for (let j = 0; j < gridPoints; j++) {
      const joint = pdp2d[i]?.[j] ?? 0;
      const marg0 = pdp0.averagePredictions[i] ?? 0;
      const marg1 = pdp1.averagePredictions[j] ?? 0;
      numerator += (joint - marg0 - marg1) ** 2;
      denominator += joint ** 2;
    }
  }
  return denominator > 0 ? numerator / denominator : 0;
}

export function globalSurrogateModel(
  model: { predict: (X: Float64Array[]) => Float64Array },
  X: Float64Array[],
  maxDepth: number = 4
): { predict: (X: Float64Array[]) => Float64Array; fidelity: number; treeStructure: SurrogateNode } {
  const predictions = model.predict(X);

  // Build a simple regression tree as surrogate
  const tree = buildSurrogateTree(X, predictions, 0, maxDepth);
  const surrogatePredict = (Xpred: Float64Array[]) => new Float64Array(Xpred.map(row => predictSurrogate(tree, row)));

  // Compute fidelity (R2 between black-box and surrogate)
  const surrogPreds = surrogatePredict(X);
  const mean = predictions.reduce((s, v) => s + v, 0) / predictions.length;
  const ss = predictions.reduce((s, v, i) => s + ((surrogPreds[i] ?? 0) - v) ** 2, 0);
  const st = predictions.reduce((s, v) => s + (v - mean) ** 2, 0);
  const fidelity = 1 - ss / (st + 1e-10);

  return { predict: surrogatePredict, fidelity, treeStructure: tree };
}

export interface SurrogateNode {
  leaf: boolean;
  value?: number;
  feature?: number;
  threshold?: number;
  left?: SurrogateNode;
  right?: SurrogateNode;
}

function buildSurrogateTree(X: Float64Array[], y: Float64Array, depth: number, maxDepth: number): SurrogateNode {
  const n = X.length;
  const mean = y.reduce((s, v) => s + v, 0) / n;
  if (depth >= maxDepth || n < 5) return { leaf: true, value: mean };

  const p = X[0]?.length ?? 0;
  let bestFeature = 0, bestThreshold = 0, bestMSE = Number.POSITIVE_INFINITY;

  for (let j = 0; j < p; j++) {
    const vals = X.map((r, i) => ({ v: r[j] ?? 0, y: y[i] ?? 0 })).sort((a, b) => a.v - b.v);
    for (let k = 1; k < vals.length; k++) {
      const threshold = ((vals[k - 1]?.v ?? 0) + (vals[k]?.v ?? 0)) / 2;
      const left = vals.slice(0, k), right = vals.slice(k);
      const mseL = left.reduce((s, e) => s + (e.y - left.reduce((ss, ee) => ss + ee.y, 0) / left.length) ** 2, 0);
      const mseR = right.reduce((s, e) => s + (e.y - right.reduce((ss, ee) => ss + ee.y, 0) / right.length) ** 2, 0);
      if (mseL + mseR < bestMSE) { bestMSE = mseL + mseR; bestFeature = j; bestThreshold = threshold; }
    }
  }

  const leftIdx: number[] = [], rightIdx: number[] = [];
  for (let i = 0; i < n; i++) (X[i]![bestFeature]! <= bestThreshold ? leftIdx : rightIdx).push(i);
  if (leftIdx.length === 0 || rightIdx.length === 0) return { leaf: true, value: mean };

  return {
    leaf: false, feature: bestFeature, threshold: bestThreshold,
    left: buildSurrogateTree(leftIdx.map(i => X[i]!), new Float64Array(leftIdx.map(i => y[i] ?? 0)), depth + 1, maxDepth),
    right: buildSurrogateTree(rightIdx.map(i => X[i]!), new Float64Array(rightIdx.map(i => y[i] ?? 0)), depth + 1, maxDepth)
  };
}

function predictSurrogate(node: SurrogateNode, row: Float64Array): number {
  if (node.leaf) return node.value ?? 0;
  if ((row[node.feature ?? 0] ?? 0) <= (node.threshold ?? 0)) return predictSurrogate(node.left!, row);
  return predictSurrogate(node.right!, row);
}

/**
 * Feature importance analysis and interaction detection.
 */

export interface ModelWithFeatureImportance {
  predict(X: Float64Array[]): Float64Array | Int32Array;
  featureImportances?: Float64Array;
}

export function permutationImportance(
  model: ModelWithFeatureImportance,
  X: Float64Array[],
  y: Float64Array | Int32Array,
  scorer: (yTrue: Float64Array | Int32Array, yPred: Float64Array | Int32Array) => number,
  nRepeats = 5
): { importances: Float64Array; importancesStd: Float64Array } {
  const n = X.length, p = X[0]?.length ?? 0;
  const baseScore = scorer(y, model.predict(X));
  const importances = new Float64Array(p);
  const importancesStd = new Float64Array(p);

  for (let j = 0; j < p; j++) {
    const scores = new Float64Array(nRepeats);
    for (let rep = 0; rep < nRepeats; rep++) {
      const Xperm = X.map(row => new Float64Array(row));
      // Shuffle feature j
      const vals = X.map(row => row[j] ?? 0);
      for (let i = vals.length - 1; i > 0; i--) {
        const k = Math.floor(Math.random() * (i + 1));
        [vals[i], vals[k]] = [vals[k]!, vals[i]!];
      }
      for (let i = 0; i < n; i++) Xperm[i]![j] = vals[i] ?? 0;
      scores[rep] = baseScore - scorer(y, model.predict(Xperm));
    }
    importances[j] = scores.reduce((s, v) => s + v, 0) / nRepeats;
    const mean = importances[j]!;
    importancesStd[j] = Math.sqrt(scores.reduce((s, v) => s + (v - mean) ** 2, 0) / nRepeats);
  }
  return { importances, importancesStd };
}

export function pairwiseInteractionStrengths(
  model: ModelWithFeatureImportance,
  X: Float64Array[],
  featurePairs: Array<[number, number]>
): Float64Array {
  const p = X[0]?.length ?? 0;
  const basePred = model.predict(X) as Float64Array;
  return new Float64Array(featurePairs.length).map((_, pi) => {
    const [f1, f2] = featurePairs[pi]!;
    // H-statistic: how much the joint effect differs from sum of marginal effects
    const marginal1 = model.predict(X.map(row => {
      const r = new Float64Array(p).fill(0);
      r[f1] = row[f1] ?? 0;
      return r;
    })) as Float64Array;
    const marginal2 = model.predict(X.map(row => {
      const r = new Float64Array(p).fill(0);
      r[f2] = row[f2] ?? 0;
      return r;
    })) as Float64Array;
    const joint = model.predict(X.map(row => {
      const r = new Float64Array(p).fill(0);
      r[f1] = row[f1] ?? 0;
      r[f2] = row[f2] ?? 0;
      return r;
    })) as Float64Array;
    const hStat = joint.reduce((s, v, i) => s + (v - (marginal1[i] ?? 0) - (marginal2[i] ?? 0)) ** 2, 0);
    const total = basePred.reduce((s, v) => s + v * v, 0);
    return total > 0 ? hStat / total : 0;
  });
}

export class ICEPlot {
  private gridValues: Float64Array;
  private predictions_!: Float64Array[][];
  private fitted_ = false;

  constructor(private featureIdx: number, private nGridPoints = 20) {
    this.gridValues = new Float64Array(nGridPoints);
  }

  compute(model: ModelWithFeatureImportance, X: Float64Array[]): this {
    const featureVals = X.map(row => row[this.featureIdx] ?? 0);
    const min = Math.min(...featureVals), max = Math.max(...featureVals);
    this.gridValues = new Float64Array(this.nGridPoints).map((_, k) => min + k * (max - min) / (this.nGridPoints - 1));
    const p = X[0]?.length ?? 0;
    this.predictions_ = X.map(xi => {
      return Array.from(this.gridValues).map(gv => {
        const xmod = new Float64Array(xi);
        xmod[this.featureIdx] = gv;
        const pred = model.predict([xmod]) as Float64Array;
        return new Float64Array([pred[0] ?? 0]);
      });
    });
    this.fitted_ = true;
    void p;
    return this;
  }

  get pdpValues(): Float64Array {
    if (!this.fitted_) throw new Error('Not computed');
    const nGrid = this.gridValues.length;
    return new Float64Array(nGrid).map((_, k) =>
      this.predictions_.reduce((s, preds) => s + (preds[k]![0] ?? 0), 0) / this.predictions_.length
    );
  }

  get grid(): Float64Array { return this.gridValues; }
}

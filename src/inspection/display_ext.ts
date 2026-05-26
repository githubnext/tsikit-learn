/**
 * Inspection display extensions: LIME explainer, SHAP display utilities.
 */

export interface Explanation {
  instance: Float64Array;
  featureImportances: Float64Array;
  featureNames: string[];
  intercept: number;
  prediction: number;
  score: number;
}

export class LIMEExplainer {
  constructor(
    private readonly predict: (X: Float64Array[]) => Float64Array,
    private readonly featureNames?: string[],
    private readonly nSamples = 500,
    private readonly kernelWidth = 0.75,
    private readonly seed = 42
  ) {}

  explain(instance: Float64Array): Explanation {
    const nF = instance.length;
    const rng = this._seededRng(this.seed);
    // Generate perturbations
    const perturbations: Float64Array[] = Array.from({ length: this.nSamples }, () => {
      const p = new Float64Array(nF);
      for (let f = 0; f < nF; f++) {
        p[f] = (instance[f] ?? 0) + (rng() - 0.5) * this.kernelWidth * 2;
      }
      return p;
    });
    // Get predictions
    const predictions = this.predict(perturbations);
    // Compute kernel weights
    const weights = perturbations.map((p) => {
      let d = 0;
      for (let f = 0; f < nF; f++) d += ((p[f] ?? 0) - (instance[f] ?? 0)) ** 2;
      return Math.exp(-d / (2 * this.kernelWidth ** 2));
    });
    // Weighted linear regression
    const importances = this._weightedLinearRegression(perturbations, predictions, weights);
    const intercept = predictions.reduce((a, b, i) => a + b * (weights[i] ?? 1), 0) / weights.reduce((a, b) => a + b, 0);
    const predValue = this.predict([instance])[0] ?? 0;
    // Score as R² of the local model
    const score = this._computeScore(perturbations, predictions, importances, weights);
    return {
      instance,
      featureImportances: importances,
      featureNames: this.featureNames ?? Array.from({ length: nF }, (_, i) => `feature_${i}`),
      intercept,
      prediction: predValue,
      score,
    };
  }

  private _weightedLinearRegression(X: Float64Array[], y: Float64Array, weights: number[]): Float64Array {
    const n = X.length;
    const nF = X[0]?.length ?? 1;
    // XtWX * beta = XtWy
    const XtW = Array.from({ length: nF }, (_, f) => {
      const row = new Float64Array(n);
      for (let i = 0; i < n; i++) row[i] = (X[i]?.[f] ?? 0) * (weights[i] ?? 1);
      return row;
    });
    const XtWy = new Float64Array(nF);
    for (let f = 0; f < nF; f++) for (let i = 0; i < n; i++) XtWy[f] = (XtWy[f] ?? 0) + (XtW[f]?.[i] ?? 0) * (y[i] ?? 0);
    const XtWX = Array.from({ length: nF }, (_, f1) => new Float64Array(nF).map((_, f2) => {
      let s = 0;
      for (let i = 0; i < n; i++) s += (XtW[f1]?.[i] ?? 0) * (X[i]?.[f2] ?? 0);
      return s;
    }));
    // Solve via pseudo-inverse (diagonal approx)
    const beta = new Float64Array(nF);
    for (let f = 0; f < nF; f++) {
      const d = XtWX[f]?.[f] ?? 1;
      beta[f] = d > 1e-10 ? (XtWy[f] ?? 0) / d : 0;
    }
    return beta;
  }

  private _computeScore(X: Float64Array[], y: Float64Array, beta: Float64Array, weights: number[]): number {
    let ssRes = 0, ssTot = 0;
    const wMean = y.reduce((a, b, i) => a + b * (weights[i] ?? 1), 0) / weights.reduce((a, b) => a + b, 0);
    for (let i = 0; i < X.length; i++) {
      const xi = X[i]!;
      let pred = 0;
      for (let f = 0; f < xi.length; f++) pred += (xi[f] ?? 0) * (beta[f] ?? 0);
      ssRes += (weights[i] ?? 1) * ((y[i] ?? 0) - pred) ** 2;
      ssTot += (weights[i] ?? 1) * ((y[i] ?? 0) - wMean) ** 2;
    }
    return ssTot > 0 ? 1 - ssRes / ssTot : 0;
  }

  private _seededRng(seed: number): () => number {
    let s = seed;
    return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
  }
}

export class SHAPDisplayUtility {
  static waterfall(explanation: Explanation): string {
    const lines: string[] = [`Prediction: ${explanation.prediction.toFixed(4)}`, `Intercept: ${explanation.intercept.toFixed(4)}`];
    const pairs = explanation.featureNames.map((name, i) => ({ name, value: explanation.featureImportances[i] ?? 0 }));
    pairs.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    for (const { name, value } of pairs.slice(0, 10)) {
      const bar = value >= 0 ? `+${"█".repeat(Math.round(Math.abs(value) * 20))}` : `-${"█".repeat(Math.round(Math.abs(value) * 20))}`;
      lines.push(`${name.padEnd(20)} ${value.toFixed(4).padStart(8)} ${bar}`);
    }
    return lines.join("\n");
  }

  static forceplot(explanation: Explanation): string {
    const positive = explanation.featureNames
      .map((name, i) => ({ name, value: explanation.featureImportances[i] ?? 0 }))
      .filter((p) => p.value > 0).sort((a, b) => b.value - a.value).slice(0, 3).map((p) => `${p.name}:+${p.value.toFixed(3)}`).join(", ");
    const negative = explanation.featureNames
      .map((name, i) => ({ name, value: explanation.featureImportances[i] ?? 0 }))
      .filter((p) => p.value < 0).sort((a, b) => a.value - b.value).slice(0, 3).map((p) => `${p.name}:${p.value.toFixed(3)}`).join(", ");
    return `[${negative}] → ${explanation.prediction.toFixed(4)} ← [${positive}]`;
  }
}

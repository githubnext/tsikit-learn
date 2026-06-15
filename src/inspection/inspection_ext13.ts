/**
 * SHAP-like explainability and model calibration inspection.
 */

export interface Explainer {
  explain(X: Float64Array[]): Float64Array[];
}

export class KernelSHAP implements Explainer {
  private baseValue_!: number;
  private fitted_ = false;

  constructor(
    private model: { predict(X: Float64Array[]): Float64Array },
    private nSamples = 100
  ) {}

  fit(XBackground: Float64Array[]): this {
    const preds = this.model.predict(XBackground) as Float64Array;
    this.baseValue_ = preds.reduce((s, v) => s + v, 0) / preds.length;
    this.fitted_ = true;
    return this;
  }

  explain(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    const p = X[0]?.length ?? 0;
    return X.map(x => {
      const shapValues = new Float64Array(p);
      // Approximate SHAP via random coalition sampling
      for (let sample = 0; sample < this.nSamples; sample++) {
        const subset = Array.from({ length: p }, (_, j) => ({ j, include: Math.random() < 0.5 }));
        const masked = new Float64Array(p).map((_, j) => subset[j]!.include ? (x[j] ?? 0) : 0);
        const pred = this.model.predict([masked])[0] ?? 0;
        for (const { j, include } of subset) {
          if (include) shapValues[j] = (shapValues[j] ?? 0) + (pred - this.baseValue_) / this.nSamples;
        }
      }
      // Normalize to sum to prediction - base
      const modelPred = this.model.predict([x])[0] ?? 0;
      const shapSum = shapValues.reduce((s, v) => s + v, 0);
      const scale = shapSum !== 0 ? (modelPred - this.baseValue_) / shapSum : 1;
      return new Float64Array(shapValues.map(v => v * scale));
    });
  }

  get baseValue(): number { return this.baseValue_; }
}

export class TreeSHAP {
  constructor(private treePredictor: {
    predict(X: Float64Array[]): Float64Array;
    featureImportances?: Float64Array;
  }) {}

  explain(X: Float64Array[]): Float64Array[] {
    const p = X[0]?.length ?? 0;
    const importances = this.treePredictor.featureImportances;
    if (!importances) {
      // Fallback: use perturbation
      const preds = this.treePredictor.predict(X) as Float64Array;
      return X.map((x, i) => {
        const shapVals = new Float64Array(p);
        const predI = preds[i] ?? 0;
        for (let j = 0; j < p; j++) {
          const xMasked = new Float64Array(x).map((v, k) => k === j ? 0 : v);
          const predMasked = (this.treePredictor.predict([xMasked]) as Float64Array)[0] ?? 0;
          shapVals[j] = predI - predMasked;
        }
        return shapVals;
      });
    }
    // Use feature importances scaled to prediction magnitude
    return X.map(x => {
      const pred = (this.treePredictor.predict([x]) as Float64Array)[0] ?? 0;
      const importanceSum = importances.reduce((s, v) => s + v, 0);
      return new Float64Array(importances.map(v => pred * v / (importanceSum + 1e-10)));
    });
  }
}

export class LIMEExplainer implements Explainer {
  private fitted_ = false;

  constructor(
    private model: { predict(X: Float64Array[]): Float64Array },
    private kernelWidth = 0.75,
    private nSamples = 200,
    private nFeatures = 5
  ) {}

  fit(_X: Float64Array[]): this { this.fitted_ = true; return this; }

  explain(X: Float64Array[]): Float64Array[] {
    const p = X[0]?.length ?? 0;
    return X.map(x => {
      // Generate perturbed samples around x
      const perturbedX = Array.from({ length: this.nSamples }, () =>
        new Float64Array(p).map((_, j) => (x[j] ?? 0) + (Math.random() - 0.5) * 0.1)
      );
      const perturbedPreds = this.model.predict(perturbedX) as Float64Array;
      // Compute kernel weights based on distance
      const weights = perturbedX.map(px => {
        const dist = Math.sqrt(px.reduce((s, v, j) => s + (v - (x[j] ?? 0)) ** 2, 0));
        return Math.exp(-(dist ** 2) / this.kernelWidth ** 2);
      });
      // Weighted linear regression
      const W = new Float64Array(this.nSamples).map((_, i) => weights[i] ?? 0);
      const WX = perturbedX.map((px, i) => new Float64Array([...px, 1]).map(v => v * (W[i] ?? 0)));
      const pAug = p + 1;
      const XtWX = Array.from({ length: pAug }, (_, a) =>
        new Float64Array(pAug).map((_, b) => WX.reduce((s, wx, i) => s + (wx[a] ?? 0) * (perturbedX[i]![b < p ? b : 0] ?? (b === p ? 1 : 0)), 0))
      );
      const XtWy = new Float64Array(pAug).map((_, a) => WX.reduce((s, wx, i) => s + (wx[a] ?? 0) * (perturbedPreds[i] ?? 0), 0));
      for (let a = 0; a < pAug; a++) XtWX[a]![a] = (XtWX[a]![a] ?? 0) + 1e-6;
      const aug = XtWX.map((row, a) => [...Array.from(row), XtWy[a] ?? 0]);
      for (let col = 0; col < pAug; col++) {
        const piv = aug[col]![col] ?? 1;
        for (let j = col; j <= pAug; j++) aug[col]![j] = (aug[col]![j] ?? 0) / piv;
        for (let row = 0; row < pAug; row++) {
          if (row === col) continue;
          const f = aug[row]![col] ?? 0;
          for (let j = col; j <= pAug; j++) aug[row]![j] = (aug[row]![j] ?? 0) - f * (aug[col]![j] ?? 0);
        }
      }
      return new Float64Array(p).map((_, j) => aug[j]![pAug] ?? 0);
    });
  }
}

/**
 * Multi-output extensions: RegressorChain.
 */

export class RegressorChain {
  private estimators: Array<{ fit: (X: Float64Array[], y: Float64Array) => void; predict: (X: Float64Array[]) => Float64Array }> = [];
  private nOutputs = 0;
  private order: number[];

  constructor(
    private readonly baseRegressorFactory: () => { fit: (X: Float64Array[], y: Float64Array) => void; predict: (X: Float64Array[]) => Float64Array }
  ) {
    this.order = [];
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    this.nOutputs = Y[0]?.length ?? 0;
    this.order = Array.from({ length: this.nOutputs }, (_, i) => i);
    this.estimators = [];
    const augX: Float64Array[] = X.map((x) => new Float64Array(x));
    for (const k of this.order) {
      const yk = new Float64Array(Y.map((row) => row[k] ?? 0));
      const reg = this.baseRegressorFactory();
      reg.fit(augX, yk);
      this.estimators.push(reg);
      const preds = reg.predict(augX);
      for (let i = 0; i < augX.length; i++) {
        const old = augX[i]!;
        const newX = new Float64Array(old.length + 1);
        newX.set(old);
        newX[old.length] = preds[i] ?? 0;
        augX[i] = newX;
      }
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array[] {
    const result: Float64Array[] = Array.from({ length: X.length }, () => new Float64Array(this.nOutputs));
    let augX: Float64Array[] = X.map((x) => new Float64Array(x));
    for (let ki = 0; ki < this.order.length; ki++) {
      const k = this.order[ki]!;
      const preds = this.estimators[ki]?.predict(augX) ?? new Float64Array(augX.length);
      for (let i = 0; i < augX.length; i++) {
        const r = result[i];
        if (r !== undefined) r[k] = preds[i] ?? 0;
        const old = augX[i]!;
        const newX = new Float64Array(old.length + 1);
        newX.set(old);
        newX[old.length] = preds[i] ?? 0;
        augX[i] = newX;
      }
    }
    return result;
  }
}

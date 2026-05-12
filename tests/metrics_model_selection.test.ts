import { describe, expect, it } from "bun:test";
import {
  accuracy_score,
  confusion_matrix,
  f1_score,
  precision_score,
  recall_score,
} from "../src/metrics/classification.ts";
import {
  mean_absolute_error,
  mean_squared_error,
  r2_score,
} from "../src/metrics/regression.ts";
import { KFold, train_test_split } from "../src/model_selection/split.ts";

describe("Regression metrics", () => {
  it("MSE is 0 for perfect prediction", () => {
    const y = new Float64Array([1, 2, 3]);
    expect(mean_squared_error(y, y)).toBe(0);
  });

  it("MAE is 0 for perfect prediction", () => {
    const y = new Float64Array([1, 2, 3]);
    expect(mean_absolute_error(y, y)).toBe(0);
  });

  it("R² is 1 for perfect prediction", () => {
    const y = new Float64Array([1, 2, 3]);
    expect(r2_score(y, y)).toBe(1);
  });

  it("MSE is correct", () => {
    const yTrue = new Float64Array([1, 2, 3]);
    const yPred = new Float64Array([2, 3, 4]); // all off by 1
    expect(mean_squared_error(yTrue, yPred)).toBe(1);
  });
});

describe("Classification metrics", () => {
  it("accuracy is 1 for perfect prediction", () => {
    const y = new Int32Array([0, 1, 2]);
    expect(accuracy_score(y, y)).toBe(1);
  });

  it("accuracy counts correct predictions", () => {
    const yTrue = new Int32Array([0, 1, 1, 0]);
    const yPred = new Int32Array([0, 1, 0, 0]);
    expect(accuracy_score(yTrue, yPred)).toBe(0.75);
  });

  it("confusion matrix is correct for binary", () => {
    const yTrue = new Int32Array([0, 1, 0, 1, 0]);
    const yPred = new Int32Array([0, 1, 1, 1, 0]);
    const cm = confusion_matrix(yTrue, yPred);
    // [[TN, FP], [FN, TP]]
    expect((cm[0] as number[])[0]).toBe(2); // TN
    expect((cm[0] as number[])[1]).toBe(1); // FP
    expect((cm[1] as number[])[0]).toBe(0); // FN
    expect((cm[1] as number[])[1]).toBe(2); // TP
  });

  it("f1 is 1 for perfect predictions", () => {
    const y = new Int32Array([0, 1, 0, 1]);
    expect(f1_score(y, y)).toBeCloseTo(1);
  });
});

describe("train_test_split", () => {
  it("splits data correctly", () => {
    const X = Array.from({ length: 100 }, (_, i) => new Float64Array([i]));
    const y = new Float64Array(Array.from({ length: 100 }, (_, i) => i));
    const { XTrain, XTest, yTrain, yTest } = train_test_split(X, y, {
      testSize: 0.2,
    });
    expect(XTrain.length).toBe(80);
    expect(XTest.length).toBe(20);
    expect(yTrain.length).toBe(80);
    expect(yTest.length).toBe(20);
  });

  it("is reproducible with randomState", () => {
    const X = Array.from({ length: 20 }, (_, i) => new Float64Array([i]));
    const y = new Float64Array(Array.from({ length: 20 }, (_, i) => i));
    const r1 = train_test_split(X, y, { randomState: 42 });
    const r2 = train_test_split(X, y, { randomState: 42 });
    expect(Array.from(r1.yTest)).toEqual(Array.from(r2.yTest));
  });
});

describe("KFold", () => {
  it("generates k folds", () => {
    const X = Array.from({ length: 10 }, (_, i) => new Float64Array([i]));
    const kf = new KFold({ nSplits: 5 });
    const folds = [...kf.split(X)];
    expect(folds.length).toBe(5);
    for (const fold of folds) {
      expect(fold.trainIndex.length).toBe(8);
      expect(fold.testIndex.length).toBe(2);
    }
  });

  it("covers all samples exactly once", () => {
    const X = Array.from({ length: 9 }, (_, i) => new Float64Array([i]));
    const kf = new KFold({ nSplits: 3 });
    const allTest = new Set<number>();
    for (const fold of kf.split(X)) {
      for (const idx of fold.testIndex) allTest.add(idx);
    }
    expect(allTest.size).toBe(9);
  });
});

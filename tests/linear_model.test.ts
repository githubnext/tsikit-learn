import { describe, it, expect } from "bun:test";
import { LinearRegression } from "../src/linear_model/linear_regression.ts";
import { Ridge } from "../src/linear_model/ridge.ts";

describe("LinearRegression", () => {
  it("fits a simple 1D linear relationship", () => {
    const X = [
      new Float64Array([1]),
      new Float64Array([2]),
      new Float64Array([3]),
      new Float64Array([4]),
      new Float64Array([5]),
    ];
    const y = new Float64Array([2, 4, 6, 8, 10]);
    const reg = new LinearRegression();
    reg.fit(X, y);

    expect(reg.coef_).toBeDefined();
    expect(Math.abs((reg.coef_ as Float64Array)[0]! - 2)).toBeLessThan(1e-6);
    expect(Math.abs((reg.intercept_ as number))).toBeLessThan(1e-6);
  });

  it("fits with intercept", () => {
    const X = [
      new Float64Array([0]),
      new Float64Array([1]),
      new Float64Array([2]),
    ];
    const y = new Float64Array([1, 3, 5]); // y = 2x + 1
    const reg = new LinearRegression();
    reg.fit(X, y);

    expect(Math.abs((reg.coef_ as Float64Array)[0]! - 2)).toBeLessThan(1e-6);
    expect(Math.abs((reg.intercept_ as number) - 1)).toBeLessThan(1e-6);
  });

  it("fits without intercept", () => {
    const X = [
      new Float64Array([1]),
      new Float64Array([2]),
      new Float64Array([3]),
    ];
    const y = new Float64Array([3, 6, 9]); // y = 3x
    const reg = new LinearRegression({ fit_intercept: false });
    reg.fit(X, y);

    expect(Math.abs((reg.coef_ as Float64Array)[0]! - 3)).toBeLessThan(1e-6);
    expect(reg.intercept_).toBe(0);
  });

  it("predicts correctly", () => {
    const X = [new Float64Array([1]), new Float64Array([2])];
    const y = new Float64Array([1, 2]);
    const reg = new LinearRegression();
    reg.fit(X, y);

    const pred = reg.predict([new Float64Array([3])]);
    expect(Math.abs(pred[0]! - 3)).toBeLessThan(1e-4);
  });

  it("fits multiple features", () => {
    // y = 1*x1 + 2*x2
    const X = [
      new Float64Array([1, 2]),
      new Float64Array([2, 1]),
      new Float64Array([3, 3]),
      new Float64Array([4, 2]),
    ];
    const y = new Float64Array([5, 4, 9, 8]);
    const reg = new LinearRegression({ fit_intercept: false });
    reg.fit(X, y);

    const pred = reg.predict([new Float64Array([1, 2])]);
    expect(Math.abs(pred[0]! - 5)).toBeLessThan(0.1);
  });

  it("computes R² score", () => {
    const X = [
      new Float64Array([1]),
      new Float64Array([2]),
      new Float64Array([3]),
      new Float64Array([4]),
    ];
    const y = new Float64Array([2, 4, 6, 8]);
    const reg = new LinearRegression();
    reg.fit(X, y);

    const score = reg.score(X, y);
    expect(score).toBeCloseTo(1.0, 5);
  });

  it("returns R² close to 1 for perfect linear data", () => {
    const X = Array.from({ length: 20 }, (_, i) =>
      new Float64Array([i, i * 2]));
    const y = new Float64Array(Array.from({ length: 20 }, (_, i) => i * 3 + 1));
    const reg = new LinearRegression();
    reg.fit(X, y);
    expect(reg.score(X, y)).toBeGreaterThan(0.999);
  });

  it("throws NotFittedError when predicting before fit", () => {
    const reg = new LinearRegression();
    expect(() => reg.predict([new Float64Array([1])])).toThrow();
  });

  it("get_params returns all params", () => {
    const reg = new LinearRegression({ alpha: 0 } as never);
    const params = reg.get_params();
    expect("fit_intercept" in params).toBe(true);
  });
});

describe("Ridge", () => {
  it("fits a simple linear relationship with regularization", () => {
    const X = [
      new Float64Array([1]),
      new Float64Array([2]),
      new Float64Array([3]),
      new Float64Array([4]),
      new Float64Array([5]),
    ];
    const y = new Float64Array([2, 4, 6, 8, 10]);
    const reg = new Ridge({ alpha: 0.0001 });
    reg.fit(X, y);

    // With tiny alpha, should be close to OLS
    expect(Math.abs((reg.coef_ as Float64Array)[0]! - 2)).toBeLessThan(0.01);
  });

  it("shrinks coefficients with large alpha", () => {
    const X = [
      new Float64Array([1, 0]),
      new Float64Array([0, 1]),
      new Float64Array([1, 1]),
    ];
    const y = new Float64Array([2, 3, 5]);

    const regLowAlpha = new Ridge({ alpha: 0.001 });
    const regHighAlpha = new Ridge({ alpha: 100.0 });
    regLowAlpha.fit(X, y);
    regHighAlpha.fit(X, y);

    const normLow = Array.from(regLowAlpha.coef_ as Float64Array)
      .reduce((a, b) => a + b * b, 0);
    const normHigh = Array.from(regHighAlpha.coef_ as Float64Array)
      .reduce((a, b) => a + b * b, 0);

    // Higher alpha → smaller coefficients
    expect(normHigh).toBeLessThan(normLow);
  });

  it("predicts correctly", () => {
    const X = [new Float64Array([1]), new Float64Array([2]), new Float64Array([3])];
    const y = new Float64Array([1, 2, 3]);
    const reg = new Ridge({ alpha: 0.001 });
    reg.fit(X, y);

    const pred = reg.predict([new Float64Array([4])]);
    expect(Math.abs(pred[0]! - 4)).toBeLessThan(0.1);
  });

  it("score is R²", () => {
    const X = Array.from({ length: 20 }, (_, i) => new Float64Array([i]));
    const y = new Float64Array(Array.from({ length: 20 }, (_, i) => i * 2 + 1));
    const reg = new Ridge({ alpha: 0.001 });
    reg.fit(X, y);
    expect(reg.score(X, y)).toBeGreaterThan(0.99);
  });

  it("throws NotFittedError when predicting before fit", () => {
    const reg = new Ridge();
    expect(() => reg.predict([new Float64Array([1])])).toThrow();
  });
});

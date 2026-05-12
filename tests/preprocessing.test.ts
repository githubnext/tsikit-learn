import { describe, it, expect } from "bun:test";
import { StandardScaler } from "../src/preprocessing/standard_scaler.ts";
import { MinMaxScaler } from "../src/preprocessing/minmax_scaler.ts";
import { LabelEncoder } from "../src/preprocessing/label_encoder.ts";
import { Normalizer } from "../src/preprocessing/normalizer.ts";
import { NotFittedError } from "../src/exceptions.ts";

describe("StandardScaler", () => {
  const X = [
    new Float64Array([1, 2]),
    new Float64Array([3, 4]),
    new Float64Array([5, 6]),
  ];

  it("computes mean and std correctly", () => {
    const scaler = new StandardScaler();
    scaler.fit(X);
    expect(scaler.mean_).toBeDefined();
    expect(Math.abs((scaler.mean_ as Float64Array)[0]! - 3)).toBeLessThan(1e-10);
    expect(Math.abs((scaler.mean_ as Float64Array)[1]! - 4)).toBeLessThan(1e-10);
  });

  it("transforms to zero mean", () => {
    const scaler = new StandardScaler();
    const Xt = scaler.fit_transform(X);
    const mean0 = Xt.reduce((a, r) => a + (r[0] ?? 0), 0) / Xt.length;
    expect(Math.abs(mean0)).toBeLessThan(1e-10);
  });

  it("inverse_transform recovers original", () => {
    const scaler = new StandardScaler();
    const Xt = scaler.fit_transform(X);
    const Xr = scaler.inverse_transform(Xt);
    for (let i = 0; i < X.length; i++) {
      for (let j = 0; j < (X[i] as Float64Array).length; j++) {
        expect(Math.abs((Xr[i] as Float64Array)[j]! - (X[i] as Float64Array)[j]!)).toBeLessThan(1e-8);
      }
    }
  });

  it("throws when not fitted", () => {
    const scaler = new StandardScaler();
    expect(() => scaler.transform(X)).toThrow(NotFittedError);
  });
});

describe("MinMaxScaler", () => {
  const X = [
    new Float64Array([0, 2]),
    new Float64Array([5, 4]),
    new Float64Array([10, 6]),
  ];

  it("scales to [0, 1] by default", () => {
    const scaler = new MinMaxScaler();
    const Xt = scaler.fit_transform(X);
    expect((Xt[0] as Float64Array)[0]).toBeCloseTo(0, 8);
    expect((Xt[2] as Float64Array)[0]).toBeCloseTo(1, 8);
  });

  it("scales to custom range", () => {
    const scaler = new MinMaxScaler({ feature_range: [-1, 1] });
    const Xt = scaler.fit_transform(X);
    expect((Xt[0] as Float64Array)[0]).toBeCloseTo(-1, 6);
    expect((Xt[2] as Float64Array)[0]).toBeCloseTo(1, 6);
  });

  it("inverse_transform recovers original", () => {
    const scaler = new MinMaxScaler();
    const Xt = scaler.fit_transform(X);
    const Xr = scaler.inverse_transform(Xt);
    for (let i = 0; i < X.length; i++) {
      for (let j = 0; j < (X[i] as Float64Array).length; j++) {
        expect(Math.abs((Xr[i] as Float64Array)[j]! - (X[i] as Float64Array)[j]!)).toBeLessThan(1e-8);
      }
    }
  });
});

describe("LabelEncoder", () => {
  it("encodes labels", () => {
    const le = new LabelEncoder();
    const y = new Int32Array([3, 1, 2, 1, 3]);
    const encoded = le.fit_transform(y);
    expect(Array.from(encoded)).toEqual([2, 0, 1, 0, 2]);
  });

  it("inverse_transform recovers original", () => {
    const le = new LabelEncoder();
    const y = new Int32Array([10, 20, 30]);
    const encoded = le.fit_transform(y);
    const decoded = le.inverse_transform(encoded);
    expect(Array.from(decoded)).toEqual([10, 20, 30]);
  });

  it("throws on unseen labels", () => {
    const le = new LabelEncoder();
    le.fit(new Int32Array([1, 2, 3]));
    expect(() => le.transform(new Int32Array([4]))).toThrow();
  });
});

describe("Normalizer", () => {
  it("normalizes to unit L2 norm", () => {
    const norm = new Normalizer({ norm: "l2" });
    const X = [new Float64Array([3, 4])]; // 3² + 4² = 25, norm = 5
    const Xt = norm.transform(X);
    expect((Xt[0] as Float64Array)[0]).toBeCloseTo(0.6, 8);
    expect((Xt[0] as Float64Array)[1]).toBeCloseTo(0.8, 8);
  });

  it("normalizes to unit L1 norm", () => {
    const norm = new Normalizer({ norm: "l1" });
    const X = [new Float64Array([1, 3])]; // sum = 4
    const Xt = norm.transform(X);
    expect((Xt[0] as Float64Array)[0]).toBeCloseTo(0.25, 8);
    expect((Xt[0] as Float64Array)[1]).toBeCloseTo(0.75, 8);
  });
});

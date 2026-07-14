import { describe, expect, it } from "bun:test";
import {
  EmpiricalCovariance,
  LedoitWolf,
  ShrunkCovariance,
} from "../src/covariance/covariance.ts";
import { PLSRegression, PLSSVD } from "../src/cross_decomposition/pls.ts";
import {
  FactorAnalysis,
  IncrementalPCA,
  KernelPCA,
} from "../src/decomposition/advanced.ts";
import {
  CountVectorizer,
  HashingVectorizer,
  TfidfTransformer,
  TfidfVectorizer,
} from "../src/feature_extraction/text.ts";
import {
  AdditiveChi2Sampler,
  Nystroem,
  RBFSampler,
} from "../src/kernel_approximation/rbf_sampler.ts";
import {
  Binarizer,
  FunctionTransformer,
  PowerTransformer,
  QuantileTransformer,
} from "../src/preprocessing/power_transformer.ts";

const DOCS = [
  "the cat sat on the mat",
  "the dog sat on the log",
  "cats and dogs are pets",
  "i love my cat and my dog",
];

describe("CountVectorizer", () => {
  it("fits and transforms documents", () => {
    const cv = new CountVectorizer({ minDf: 1, maxFeatures: 10 });
    const X = cv.fitTransform(DOCS);
    expect(X.length).toBe(DOCS.length);
    const features = cv.getFeatureNames();
    expect(features.length).toBeGreaterThan(0);
    // 'the' should appear in most docs
    const theIdx = features.indexOf("the");
    if (theIdx >= 0) {
      expect(X[0]![theIdx] ?? 0).toBeGreaterThan(0);
    }
  });

  it("respects minDf filter", () => {
    const cv = new CountVectorizer({ minDf: 2 });
    cv.fit(DOCS);
    const features = cv.getFeatureNames();
    // Only terms appearing in >= 2 docs
    expect(features.length).toBeGreaterThan(0);
    for (const f of features) {
      const count = DOCS.filter(
        (d) =>
          d
            .toLowerCase()
            .match(/\b[a-z0-9]+\b/g)
            ?.includes(f) ?? false,
      ).length;
      expect(count).toBeGreaterThanOrEqual(2);
    }
  });

  it("throws NotFittedError before fit", () => {
    const cv = new CountVectorizer();
    expect(() => cv.transform(DOCS)).toThrow();
  });
});

describe("TfidfTransformer", () => {
  it("transforms count matrix to TF-IDF", () => {
    const cv = new CountVectorizer();
    const counts = cv.fitTransform(DOCS);
    const tfidf = new TfidfTransformer();
    const X = tfidf.fitTransform(counts);
    expect(X.length).toBe(DOCS.length);
    // After L2 norm, each row should have approximately unit length
    for (const row of X) {
      const norm = Math.sqrt(Array.from(row).reduce((s, x) => s + x * x, 0));
      if (norm > 0) expect(Math.abs(norm - 1)).toBeLessThan(1e-10);
    }
  });
});

describe("TfidfVectorizer", () => {
  it("combines CountVectorizer and TfidfTransformer", () => {
    const tv = new TfidfVectorizer({ minDf: 1 });
    const X = tv.fitTransform(DOCS);
    expect(X.length).toBe(DOCS.length);
    const features = tv.getFeatureNames();
    expect(features.length).toBeGreaterThan(0);
  });
});

describe("HashingVectorizer", () => {
  it("transforms documents without fitting", () => {
    const hv = new HashingVectorizer({ nFeatures: 256 });
    const X = hv.transform(DOCS);
    expect(X.length).toBe(DOCS.length);
    expect(X[0]!.length).toBe(256);
    // Non-empty documents should have non-zero features
    const total = Array.from(X[0]!).reduce((s, x) => s + Math.abs(x), 0);
    expect(total).toBeGreaterThan(0);
  });
});

describe("RBFSampler", () => {
  const X = [
    new Float64Array([1, 0]),
    new Float64Array([0, 1]),
    new Float64Array([1, 1]),
    new Float64Array([0, 0]),
  ];

  it("transforms to correct dimension", () => {
    const rbf = new RBFSampler({ nComponents: 10, gamma: 1.0 });
    const Xt = rbf.fitTransform(X);
    expect(Xt.length).toBe(4);
    expect(Xt[0]!.length).toBe(10);
  });

  it("throws before fitting", () => {
    const rbf = new RBFSampler();
    expect(() => rbf.transform(X)).toThrow();
  });
});

describe("Nystroem", () => {
  const X = [
    new Float64Array([1, 0]),
    new Float64Array([0, 1]),
    new Float64Array([1, 1]),
    new Float64Array([0, 0]),
    new Float64Array([0.5, 0.5]),
  ];

  it("transforms with rbf kernel", () => {
    const ny = new Nystroem({ kernel: "rbf", nComponents: 3 });
    const Xt = ny.fitTransform(X);
    expect(Xt.length).toBe(5);
    expect(Xt[0]!.length).toBe(3);
  });

  it("transforms with linear kernel", () => {
    const ny = new Nystroem({ kernel: "linear", nComponents: 3 });
    const Xt = ny.fitTransform(X);
    expect(Xt.length).toBe(5);
  });
});

describe("AdditiveChi2Sampler", () => {
  const X = [new Float64Array([0.5, 0.3]), new Float64Array([0.2, 0.8])];

  it("transforms to higher dimension", () => {
    const sampler = new AdditiveChi2Sampler({ sampleSteps: 2 });
    const Xt = sampler.fitTransform(X);
    expect(Xt.length).toBe(2);
    expect(Xt[0]!.length).toBe(2 * (2 * 2 + 1)); // p * (2 * steps + 1)
  });
});

describe("EmpiricalCovariance", () => {
  const X = [
    new Float64Array([1, 2]),
    new Float64Array([2, 3]),
    new Float64Array([3, 4]),
    new Float64Array([4, 5]),
    new Float64Array([5, 6]),
  ];

  it("computes covariance matrix", () => {
    const ec = new EmpiricalCovariance();
    ec.fit(X);
    expect(ec.covariance_).toBeDefined();
    expect(ec.location_).toBeDefined();
    expect(ec.location_![0] ?? 0).toBeCloseTo(3, 5);
    expect(ec.location_![1] ?? 0).toBeCloseTo(4, 5);
  });

  it("computes mahalanobis distances", () => {
    const ec = new EmpiricalCovariance();
    ec.fit(X);
    const dists = ec.mahalanobis(X);
    expect(dists.length).toBe(5);
    for (let i = 0; i < 5; i++) expect(dists[i] ?? 0).toBeGreaterThanOrEqual(0);
  });
});

describe("ShrunkCovariance", () => {
  const X = [
    new Float64Array([1, 2, 3]),
    new Float64Array([2, 3, 4]),
    new Float64Array([3, 4, 5]),
    new Float64Array([4, 5, 6]),
  ];

  it("applies shrinkage to off-diagonal", () => {
    const sc = new ShrunkCovariance({ shrinkage: 0.5 });
    sc.fit(X);
    expect(sc.covariance_).toBeDefined();
    const emp = new EmpiricalCovariance();
    emp.fit(X);
    // Off-diagonal elements should be smaller
    const off01_sc = Math.abs(sc.covariance_![0]![1] ?? 0);
    const off01_emp = Math.abs(emp.covariance_![0]![1] ?? 0);
    expect(off01_sc).toBeLessThanOrEqual(off01_emp + 1e-10);
  });
});

describe("LedoitWolf", () => {
  const X = [
    new Float64Array([1, 2]),
    new Float64Array([2, 3]),
    new Float64Array([3, 2]),
    new Float64Array([1, 3]),
    new Float64Array([2, 1]),
  ];

  it("fits and returns a covariance matrix", () => {
    const lw = new LedoitWolf();
    lw.fit(X);
    expect(lw.covariance_).toBeDefined();
    expect(lw.shrinkage_).toBeDefined();
    expect(lw.shrinkage_!).toBeGreaterThanOrEqual(0);
  });
});

describe("PLSRegression", () => {
  const X = [
    new Float64Array([1, 2]),
    new Float64Array([2, 3]),
    new Float64Array([3, 4]),
    new Float64Array([4, 5]),
    new Float64Array([5, 6]),
  ];
  const Y = [
    new Float64Array([1]),
    new Float64Array([2]),
    new Float64Array([3]),
    new Float64Array([4]),
    new Float64Array([5]),
  ];

  it("fits and predicts", () => {
    const pls = new PLSRegression({ nComponents: 1 });
    pls.fit(X, Y);
    const pred = pls.predict(X);
    expect(pred.length).toBe(5);
    // Should predict something close to the actual Y (linear relationship)
    for (let i = 0; i < 5; i++) {
      expect(Math.abs((pred[i]![0] ?? 0) - (Y[i]![0] ?? 0))).toBeLessThan(1);
    }
  });

  it("transforms to latent space", () => {
    const pls = new PLSRegression({ nComponents: 2 });
    pls.fit(X, Y);
    const Xt = pls.transform(X);
    expect(Xt.length).toBe(5);
    expect(Xt[0]!.length).toBe(2);
  });

  it("throws before fitting", () => {
    const pls = new PLSRegression();
    expect(() => pls.predict(X)).toThrow();
  });
});

describe("PLSSVD", () => {
  const X = [
    new Float64Array([1, 2]),
    new Float64Array([2, 3]),
    new Float64Array([3, 4]),
    new Float64Array([4, 5]),
  ];
  const Y = [
    new Float64Array([1, 0]),
    new Float64Array([2, 1]),
    new Float64Array([3, 2]),
    new Float64Array([4, 3]),
  ];

  it("extracts latent components", () => {
    const plssvd = new PLSSVD({ nComponents: 2 });
    const [xScores, yScores] = plssvd.fitTransform(X, Y);
    expect(xScores.length).toBe(4);
    expect(xScores[0]!.length).toBe(2);
    expect(yScores.length).toBe(4);
  });
});

describe("PowerTransformer", () => {
  const X = [
    new Float64Array([1, 2]),
    new Float64Array([4, 8]),
    new Float64Array([16, 32]),
    new Float64Array([64, 128]),
  ];

  it("yeo-johnson transform", () => {
    const pt = new PowerTransformer({
      method: "yeo-johnson",
      standardize: true,
    });
    const Xt = pt.fitTransform(X);
    expect(Xt.length).toBe(4);
    expect(Xt[0]!.length).toBe(2);
    // Standardized output should be roughly centered
    let sum0 = 0;
    for (const row of Xt) sum0 += row[0] ?? 0;
    expect(Math.abs(sum0 / 4)).toBeLessThan(5); // rough check
  });
});

describe("QuantileTransformer", () => {
  const X = Array.from(
    { length: 20 },
    (_, i) => new Float64Array([i + 1, 20 - i]),
  );

  it("uniform output", () => {
    const qt = new QuantileTransformer({
      nQuantiles: 10,
      outputDistribution: "uniform",
    });
    const Xt = qt.fitTransform(X);
    expect(Xt.length).toBe(20);
    for (const row of Xt) {
      expect(row[0] ?? 0).toBeGreaterThanOrEqual(-1e-6);
      expect(row[0] ?? 0).toBeLessThanOrEqual(1 + 1e-6);
    }
  });

  it("normal output", () => {
    const qt = new QuantileTransformer({
      nQuantiles: 10,
      outputDistribution: "normal",
    });
    const Xt = qt.fitTransform(X);
    expect(Xt.length).toBe(20);
  });
});

describe("Binarizer", () => {
  const X = [
    new Float64Array([0.5, 1.5, -0.5]),
    new Float64Array([0.0, 2.0, 1.0]),
  ];

  it("binarizes with threshold 0", () => {
    const b = new Binarizer({ threshold: 0 });
    const Xt = b.transform(X);
    expect(Xt[0]![0]).toBe(1);
    expect(Xt[0]![1]).toBe(1);
    expect(Xt[0]![2]).toBe(0);
  });

  it("binarizes with threshold 1", () => {
    const b = new Binarizer({ threshold: 1 });
    const Xt = b.transform(X);
    expect(Xt[0]![0]).toBe(0);
    expect(Xt[0]![1]).toBe(1);
    expect(Xt[1]![1]).toBe(1);
  });
});

describe("FunctionTransformer", () => {
  const X = [new Float64Array([1, 4]), new Float64Array([9, 16])];

  it("applies custom function", () => {
    const ft = new FunctionTransformer({
      func: (X) => X.map((xi) => Float64Array.from(xi, Math.sqrt)),
    });
    const Xt = ft.fitTransform(X);
    expect(Math.abs((Xt[0]![0] ?? 0) - 1)).toBeLessThan(1e-10);
    expect(Math.abs((Xt[0]![1] ?? 0) - 2)).toBeLessThan(1e-10);
    expect(Math.abs((Xt[1]![0] ?? 0) - 3)).toBeLessThan(1e-10);
  });

  it("identity when no func", () => {
    const ft = new FunctionTransformer();
    const Xt = ft.transform(X);
    expect(Xt[0]![0]).toBe(1);
  });
});

describe("IncrementalPCA", () => {
  const X = Array.from(
    { length: 20 },
    (_, i) => new Float64Array([i, i * 2, i * 3]),
  );

  it("fits and transforms", () => {
    const ipca = new IncrementalPCA({ nComponents: 2, batchSize: 5 });
    const Xt = ipca.fitTransform(X);
    expect(Xt.length).toBe(20);
    expect(Xt[0]!.length).toBe(2);
  });

  it("partialFit accumulates samples", () => {
    const ipca = new IncrementalPCA({ nComponents: 2 });
    ipca.partialFit(X.slice(0, 10));
    ipca.partialFit(X.slice(10, 20));
    expect(ipca.nSamplesSeen_).toBe(20);
  });
});

describe("KernelPCA", () => {
  const X = [
    new Float64Array([0, 0]),
    new Float64Array([1, 0]),
    new Float64Array([0, 1]),
    new Float64Array([1, 1]),
    new Float64Array([0.5, 0.5]),
  ];

  it("rbf kernel projection", () => {
    const kpca = new KernelPCA({ nComponents: 2, kernel: "rbf", gamma: 1 });
    const Xt = kpca.fitTransform(X);
    expect(Xt.length).toBe(5);
    expect(Xt[0]!.length).toBe(2);
  });

  it("polynomial kernel", () => {
    const kpca = new KernelPCA({ nComponents: 2, kernel: "poly" });
    const Xt = kpca.fitTransform(X);
    expect(Xt.length).toBe(5);
  });
});

describe("FactorAnalysis", () => {
  const X = Array.from(
    { length: 15 },
    (_, i) => new Float64Array([Math.sin(i), Math.cos(i), i * 0.1]),
  );

  it("extracts factors", () => {
    const fa = new FactorAnalysis({ nComponents: 2, maxIter: 20 });
    const Xt = fa.fitTransform(X);
    expect(Xt.length).toBe(15);
    expect(Xt[0]!.length).toBe(2);
  });

  it("noise variance is positive", () => {
    const fa = new FactorAnalysis({ nComponents: 1, maxIter: 10 });
    fa.fit(X);
    expect(fa.noiseVariance_).toBeDefined();
    for (let i = 0; i < 3; i++) {
      expect(fa.noiseVariance_![i] ?? 0).toBeGreaterThan(0);
    }
  });
});

import { describe, it, expect } from "bun:test";
import { BaseEstimator, ClassifierMixin, RegressorMixin, clone, check_is_fitted } from "../src/base.ts";
import { NotFittedError } from "../src/exceptions.ts";

class DummyEstimator extends BaseEstimator {
  alpha: number;
  beta: string;
  fitted_?: boolean;

  constructor(alpha = 1.0, beta = "test") {
    super();
    this.alpha = alpha;
    this.beta = beta;
  }

  fit(): this {
    this.fitted_ = true;
    return this;
  }
}

describe("BaseEstimator", () => {
  it("get_params returns constructor params", () => {
    const est = new DummyEstimator(2.0, "hello");
    const params = est.get_params();
    expect(params["alpha"]).toBe(2.0);
    expect(params["beta"]).toBe("hello");
  });

  it("set_params updates params", () => {
    const est = new DummyEstimator();
    est.set_params({ alpha: 5.0 });
    expect(est.alpha).toBe(5.0);
  });

  it("check_is_fitted throws NotFittedError when not fitted", () => {
    const est = new DummyEstimator();
    expect(() => est.fit()._check_is_fitted(["fitted_"])).not.toThrow();
    const est2 = new DummyEstimator();
    expect(() => est2["_check_is_fitted"](["fitted_"])).toThrow(NotFittedError);
  });
});

describe("clone", () => {
  it("creates a new instance with same params", () => {
    const est = new DummyEstimator(3.0, "foo");
    const cloned = clone(est);
    expect(cloned).not.toBe(est);
    expect(cloned.alpha).toBe(3.0);
    expect(cloned.beta).toBe("foo");
  });
});

describe("Exceptions", () => {
  it("NotFittedError has correct name", () => {
    const err = new NotFittedError();
    expect(err.name).toBe("NotFittedError");
  });

  it("NotFittedError is an Error", () => {
    expect(new NotFittedError()).toBeInstanceOf(Error);
  });
});

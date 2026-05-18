/**
 * Parameter validation utilities — ported from sklearn.utils._param_validation
 */

/** Constraint representing a closed/open numeric interval */
export class Interval {
  constructor(
    public readonly type: "numeric" | "integer",
    public readonly left: number | null,
    public readonly right: number | null,
    public readonly closed: "left" | "right" | "both" | "neither",
  ) {}

  /** Test whether a value satisfies this interval constraint */
  isValid(value: unknown): boolean {
    if (typeof value !== "number" || Number.isNaN(value)) return false;
    if (this.type === "integer" && !Number.isInteger(value)) return false;
    if (this.left !== null) {
      const leftOk = this.closed === "left" || this.closed === "both"
        ? value >= this.left
        : value > this.left;
      if (!leftOk) return false;
    }
    if (this.right !== null) {
      const rightOk = this.closed === "right" || this.closed === "both"
        ? value <= this.right
        : value < this.right;
      if (!rightOk) return false;
    }
    return true;
  }

  toString(): string {
    const l = this.left === null ? "-inf" : String(this.left);
    const r = this.right === null ? "inf" : String(this.right);
    const lp = this.closed === "left" || this.closed === "both" ? "[" : "(";
    const rp = this.closed === "right" || this.closed === "both" ? "]" : ")";
    return `${lp}${l}, ${r}${rp}`;
  }
}

/** Constraint representing a set of valid string values */
export class StrOptions {
  constructor(public readonly options: ReadonlySet<string>) {}

  isValid(value: unknown): boolean {
    return typeof value === "string" && this.options.has(value);
  }

  toString(): string {
    return `{${Array.from(this.options).map(s => `'${s}'`).join(", ")}}`;
  }
}

/** Constraint requiring value to be one of a set of objects (including null) */
export class Options {
  constructor(public readonly options: ReadonlySet<unknown>) {}

  isValid(value: unknown): boolean {
    return this.options.has(value);
  }

  toString(): string {
    return `{${Array.from(this.options).map(v => JSON.stringify(v)).join(", ")}}`;
  }
}

/** Constraint requiring value to be callable */
export class Callable {
  isValid(value: unknown): boolean {
    return typeof value === "function";
  }

  toString(): string {
    return "callable";
  }
}

/** Constraint requiring value to be an array/typed array */
export class ArrayLike {
  isValid(value: unknown): boolean {
    return Array.isArray(value)
      || value instanceof Float64Array
      || value instanceof Int32Array
      || value instanceof Float32Array;
  }

  toString(): string {
    return "array-like";
  }
}

/** Union of all constraint types */
export type Constraint = Interval | StrOptions | Options | Callable | ArrayLike;

/** Map of parameter names to arrays of valid constraints */
export type ParamConstraints = Record<string, Constraint[]>;

/** Error thrown when a parameter fails validation */
export class InvalidParameterError extends Error {
  constructor(
    public readonly paramName: string,
    public readonly value: unknown,
    public readonly constraints: Constraint[],
    estimatorName?: string,
  ) {
    const constraintStr = constraints.map(c => c.toString()).join(" or ");
    const prefix = estimatorName ? `${estimatorName}: ` : "";
    super(
      `${prefix}Parameter '${paramName}' must be ${constraintStr}; got ${JSON.stringify(value)} instead.`
    );
    this.name = "InvalidParameterError";
  }
}

/**
 * Validate estimator parameters against their constraints.
 * Throws InvalidParameterError on first violation found.
 */
export function validateParams(
  params: Record<string, unknown>,
  constraints: ParamConstraints,
  estimatorName?: string,
): void {
  for (const [name, constraintList] of Object.entries(constraints)) {
    if (!(name in params)) continue;
    const value = params[name];
    const valid = constraintList.some(c => c.isValid(value));
    if (!valid) {
      throw new InvalidParameterError(name, value, constraintList, estimatorName);
    }
  }
}

/** Convenience factory for a real-valued closed interval */
export function realInterval(left: number | null, right: number | null, closed: "left" | "right" | "both" | "neither" = "both"): Interval {
  return new Interval("numeric", left, right, closed);
}

/** Convenience factory for an integer interval */
export function intInterval(left: number | null, right: number | null, closed: "left" | "right" | "both" | "neither" = "both"): Interval {
  return new Interval("integer", left, right, closed);
}

/** Convenience factory for a set of string options */
export function strOptions(...values: string[]): StrOptions {
  return new StrOptions(new Set(values));
}

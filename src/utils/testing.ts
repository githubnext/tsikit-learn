/**
 * Testing utilities.
 * Mirrors scikit-learn's utils.testing and sklearn.utils._testing.
 */

/** Assert two arrays are element-wise equal within a tolerance. */
export function assertArrayAlmostEqual(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  decimal = 6,
): void {
  const tol = 10 ** -decimal;
  if (actual.length !== expected.length) {
    throw new Error(`Length mismatch: ${actual.length} != ${expected.length}`);
  }
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i] ?? 0;
    const e = expected[i] ?? 0;
    if (Math.abs(a - e) > tol) {
      throw new Error(
        `Arrays not almost equal at index ${i}: ${a} != ${e} (diff ${Math.abs(a - e)} > ${tol})`,
      );
    }
  }
}

/** Assert two arrays are element-wise exactly equal. */
export function assertArrayEqual(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
): void {
  if (actual.length !== expected.length) {
    throw new Error(`Length mismatch: ${actual.length} != ${expected.length}`);
  }
  for (let i = 0; i < actual.length; i++) {
    if (actual[i] !== expected[i]) {
      throw new Error(
        `Arrays differ at index ${i}: ${actual[i]} != ${expected[i]}`,
      );
    }
  }
}

/** Assert a value is approximately equal to another. */
export function assertAlmostEqual(
  actual: number,
  expected: number,
  decimal = 7,
): void {
  const tol = 10 ** -decimal;
  if (Math.abs(actual - expected) > tol) {
    throw new Error(`${actual} != ${expected} within ${decimal} decimals`);
  }
}

/** Assert that a function raises an error matching the given pattern. */
export function assertRaises(
  fn: () => unknown,
  errorClass: new (...args: unknown[]) => Error,
  msgPattern?: RegExp,
): void {
  let threw = false;
  try {
    fn();
  } catch (e) {
    threw = true;
    if (!(e instanceof errorClass)) {
      throw new Error(
        `Expected ${errorClass.name} but got ${(e as Error).constructor.name}`,
      );
    }
    if (msgPattern !== undefined && !msgPattern.test((e as Error).message)) {
      throw new Error(
        `Error message "${(e as Error).message}" does not match ${msgPattern}`,
      );
    }
  }
  if (!threw) {
    throw new Error(
      `Expected ${errorClass.name} to be raised but no error was thrown`,
    );
  }
}

/** Create a simple mock object for testing. */
export function createMock<T extends object>(
  defaults: Partial<T> = {},
): T & { _calls: Record<string, unknown[][]> } {
  const calls: Record<string, unknown[][]> = {};
  const handler: ProxyHandler<object> = {
    get(target, prop) {
      if (prop === "_calls") return calls;
      if (prop in target)
        return (target as Record<string | symbol, unknown>)[prop];
      return (...args: unknown[]) => {
        const key = String(prop);
        if (calls[key] === undefined) calls[key] = [];
        calls[key].push(args);
      };
    },
  };
  return new Proxy(defaults as object, handler) as T & {
    _calls: Record<string, unknown[][]>;
  };
}

/** Ignore warnings during a function call. */
export function ignoreWarnings<T>(fn: () => T): T {
  return fn();
}

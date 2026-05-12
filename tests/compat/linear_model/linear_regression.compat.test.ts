/**
 * Parity tests for LinearRegression — validates TypeScript port against
 * scikit-learn reference values stored in fixtures/linear_regression.json.
 */

import { LinearRegression } from '../../../src/sklearn/linear_model/LinearRegression';
import { fromArrays } from '../../../src/sklearn/dataframe/tsessebeAdapter';
import fixtures from './fixtures/linear_regression.json';

const EPS = 1e-6;

function approxEqual(a: number, b: number, eps = EPS): boolean {
  return Math.abs(a - b) <= eps;
}

function approxArrayEqual(a: number[], b: number[], eps = EPS): boolean {
  return a.length === b.length && a.every((v, i) => approxEqual(v, b[i], eps));
}

describe('LinearRegression — parity with scikit-learn fixtures', () => {
  for (const tc of fixtures.cases) {
    it(tc.name, () => {
      const lr = new LinearRegression({ fit_intercept: tc.fit_intercept });

      const nFeatures = tc.X_train[0].length;
      const colNames = Array.from({ length: nFeatures }, (_, i) => `x${i}`);

      // Transpose X_train into column-oriented arrays
      const trainCols = colNames.map((_, j) => tc.X_train.map((row) => row[j]));
      const XTrain = fromArrays(colNames, trainCols);
      lr.fit(XTrain, tc.y_train);

      // coef check
      if (tc.expected_coef) {
        expect(approxArrayEqual(lr.coef_, tc.expected_coef, 1e-4)).toBe(true);
      }

      // intercept check
      if (tc.expected_intercept !== undefined) {
        expect(approxEqual(lr.intercept_, tc.expected_intercept, 1e-4)).toBe(true);
      }

      // predict check
      const testCols = colNames.map((_, j) => tc.X_test.map((row) => row[j]));
      const XTest = fromArrays(colNames, testCols);
      const yPred = lr.predict(XTest);

      // R² on training set
      const trainPred = lr.predict(XTrain);
      const r2 = computeR2(tc.y_train, trainPred);

      if (tc.expected_r2 !== undefined) {
        expect(approxEqual(r2, tc.expected_r2, 1e-4)).toBe(true);
      } else if ((tc as any).r2_threshold !== undefined) {
        expect(r2).toBeGreaterThanOrEqual((tc as any).r2_threshold);
      }

      // Individual predictions (for exact cases)
      if (tc.expected_coef && tc.expected_intercept !== undefined) {
        for (let i = 0; i < tc.y_test.length; i++) {
          expect(approxEqual(yPred[i], tc.y_test[i], 1e-3)).toBe(true);
        }
      }
    });
  }
});

function computeR2(yTrue: number[], yPred: number[]): number {
  const mean = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;
  const ssTot = yTrue.reduce((s, v) => s + (v - mean) ** 2, 0);
  const ssRes = yTrue.reduce((s, v, i) => s + (v - yPred[i]) ** 2, 0);
  return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
}

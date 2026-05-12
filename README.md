# tsikit-learn

TypeScript port of scikit-learn estimators with a `tsessebe`-style dataframe adapter interface.

## Status

| Estimator | Status |
|---|---|
| `LinearRegression` | ✅ Ported (`src/sklearn/linear_model/`) |

## LinearRegression

```ts
import { LinearRegression } from './src/sklearn/linear_model';
import { fromArrays } from './src/sklearn/dataframe/tsessebeAdapter';

const X = fromArrays(['x'], [[1, 2, 3, 4, 5]]);
const lr = new LinearRegression();
lr.fit(X, [2, 4, 6, 8, 10]);
console.log(lr.coef_);       // [2]
console.log(lr.intercept_);  // 0
```

🤖 *This section is maintained by the [Autoloop](https://github.com/githubnext/tsikit-learn/issues) agent.*

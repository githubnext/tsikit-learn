# Agent Instructions for tsikit-learn

## Overview

`tsikit-learn` is a TypeScript port of [scikit-learn](https://scikit-learn.org/). The project is being built one feature at a time by the Autoloop agent.

## Stack

- **Runtime & bundler**: Bun
- **Language**: TypeScript (strictest settings — `strict: true`, `noUncheckedIndexedAccess: true`, `exactOptionalPropertyTypes: true`)
- **Linting**: Biome
- **Testing**: Bun test runner with fast-check for property-based tests
- **Data layer**: `tsb` (TypeScript pandas port) as a peer dependency; typed arrays for numeric computation

## Directory Structure

```
src/
  index.ts                — public entry point, re-exports everything
  exceptions.ts           — NotFittedError, ConvergenceWarning, etc.
  base.ts                 — BaseEstimator, mixins, clone, check_is_fitted
  utils/
    extmath.ts            — math utilities (safeDot, gramMatrix, cholesky, etc.)
    validation.ts         — input validation
    multiclass.ts         — multiclass helpers
    class_weight.ts       — class weight utilities
    index.ts              — re-exports all utils
  preprocessing/
    standard_scaler.ts    — StandardScaler
    minmax_scaler.ts      — MinMaxScaler
    label_encoder.ts      — LabelEncoder
    normalizer.ts         — Normalizer
    index.ts
  metrics/
    regression.ts         — MSE, MAE, R², MAPE, explained_variance
    classification.ts     — accuracy, confusion_matrix, precision, recall, F1, log_loss
    index.ts
  model_selection/
    split.ts              — train_test_split, KFold, StratifiedKFold
    index.ts
  linear_model/
    linear_regression.ts  — LinearRegression (OLS via Cholesky)
    ridge.ts              — Ridge (L2 regularization)
    index.ts
tests/
  base.test.ts
  preprocessing.test.ts
  metrics_model_selection.test.ts
  linear_model.test.ts
playground/
  index.html              — interactive demos, deployed to GitHub Pages
```

## TypeScript Conventions

- No `any`, no `@ts-ignore`, no `as` casts (unless provably safe)
- Use `Float64Array` for continuous numeric data, `Int32Array` for integer labels
- Use `?? 0` or null checks for `noUncheckedIndexedAccess` compliance
- Export everything from module `index.ts` files

## Evaluation Metric

The CI evaluation script counts TypeScript source files in `src/` (excluding `index.ts`) that contain `export`. Currently: **15 files**.

## Adding a New Module

1. Create `src/{module}/{feature}.ts` — implement the class with `fit`, `predict`/`transform`, `score`
2. Create or update `src/{module}/index.ts` — re-export from the new file
3. Update `src/index.ts` — add `export * from "./{module}/index.js"`
4. Add tests in `tests/{module}.test.ts`
5. Add a card to `playground/index.html`

## Running Locally

```bash
bun install
bun test
bunx tsc --noEmit
```

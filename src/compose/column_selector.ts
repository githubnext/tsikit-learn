/**
 * make_column_selector and related column-selection helpers for ColumnTransformer.
 * Analogous to sklearn.compose._column.make_column_selector.
 */

/** Column selector predicate: returns true for columns to include. */
export type ColumnSelectorFn = (colIndex: number, colName: string) => boolean;

/** Options for makeColumnSelector. */
export interface MakeColumnSelectorOptions {
  /**
   * String pattern or regex that column names must match (substring match by default).
   * Set to undefined to match all columns.
   */
  pattern?: string | RegExp;
  /**
   * If provided, only include columns whose dtype matches one of these strings.
   * Uses the dtypes map passed to the returned selector.
   * Supported values: "number", "string", "boolean".
   */
  dtypeInclude?: string[];
  /** If provided, exclude columns whose dtype matches one of these. */
  dtypeExclude?: string[];
}

/**
 * Returns a column-selector callable, analogous to sklearn's `make_column_selector`.
 *
 * The returned function accepts `(colNames: string[], dtypes?: Record<string, string>)`
 * and returns an array of column indices that pass the filter criteria.
 */
export function makeColumnSelector(
  options: MakeColumnSelectorOptions = {},
): (colNames: string[], dtypes?: Record<string, string>) => number[] {
  const { pattern, dtypeInclude, dtypeExclude } = options;

  return (colNames: string[], dtypes?: Record<string, string>): number[] => {
    const result: number[] = [];
    for (let i = 0; i < colNames.length; i++) {
      const name = colNames[i]!;

      // Pattern filter
      if (pattern !== undefined) {
        if (pattern instanceof RegExp) {
          if (!pattern.test(name)) continue;
        } else {
          if (!name.includes(pattern)) continue;
        }
      }

      // Dtype filters
      const dtype = dtypes?.[name];
      if (dtypeInclude !== undefined && dtype !== undefined && !dtypeInclude.includes(dtype)) continue;
      if (dtypeExclude !== undefined && dtype !== undefined && dtypeExclude.includes(dtype)) continue;

      result.push(i);
    }
    return result;
  };
}

/**
 * Returns the indices of all numeric columns (dtype "number").
 * Convenience wrapper around makeColumnSelector.
 */
export function numericColumns(
  colNames: string[],
  dtypes: Record<string, string>,
): number[] {
  return makeColumnSelector({ dtypeInclude: ["number"] })(colNames, dtypes);
}

/**
 * Returns the indices of all categorical columns (dtype "string").
 * Convenience wrapper around makeColumnSelector.
 */
export function categoricalColumns(
  colNames: string[],
  dtypes: Record<string, string>,
): number[] {
  return makeColumnSelector({ dtypeInclude: ["string"] })(colNames, dtypes);
}

/**
 * Selects a subset of columns from a flat row-major matrix.
 *
 * @param X         Flat Float64Array of shape (nSamples × nColsIn).
 * @param nSamples  Number of rows.
 * @param nColsIn   Number of columns in X.
 * @param cols      Column indices to select.
 * @returns         New Float64Array of shape (nSamples × cols.length).
 */
export function selectColumns(
  X: Float64Array,
  nSamples: number,
  nColsIn: number,
  cols: number[],
): Float64Array {
  const nOut = cols.length;
  const out = new Float64Array(nSamples * nOut);
  for (let i = 0; i < nSamples; i++) {
    for (let k = 0; k < nOut; k++) {
      out[i * nOut + k] = X[i * nColsIn + cols[k]!]!;
    }
  }
  return out;
}

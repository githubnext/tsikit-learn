/**
 * Dataframe adapter shaped for tsessebe-style typed column/series/table operations.
 * Provides a pandas-style tabular interface for scikit-learn estimators.
 */

export type Series<T = number> = T[];

export interface DataFrame {
  columns: string[];
  data: Series[];
  shape: [number, number]; // [rows, cols]
}

/** Create a DataFrame from a 2D array and column names */
export function fromArrays(columns: string[], arrays: Series[]): DataFrame {
  if (columns.length !== arrays.length) {
    throw new Error(`columns length (${columns.length}) must match arrays length (${arrays.length})`);
  }
  const nRows = arrays[0]?.length ?? 0;
  for (const arr of arrays) {
    if (arr.length !== nRows) {
      throw new Error('All arrays must have the same length');
    }
  }
  return { columns, data: arrays, shape: [nRows, columns.length] };
}

/** Create a DataFrame from row-oriented objects */
export function fromRecords(records: Record<string, number>[]): DataFrame {
  if (records.length === 0) return { columns: [], data: [], shape: [0, 0] };
  const columns = Object.keys(records[0]);
  const arrays: Series[] = columns.map((col) => records.map((r) => r[col]));
  return { columns, data: arrays, shape: [records.length, columns.length] };
}

/** Select a single column as a Series */
export function getColumn(df: DataFrame, col: string): Series {
  const idx = df.columns.indexOf(col);
  if (idx === -1) throw new Error(`Column "${col}" not found`);
  return df.data[idx];
}

/** Convert DataFrame to row-oriented 2D array (nRows x nCols) */
export function toMatrix(df: DataFrame): number[][] {
  const [nRows, nCols] = df.shape;
  const result: number[][] = Array.from({ length: nRows }, () => new Array(nCols).fill(0));
  for (let c = 0; c < nCols; c++) {
    for (let r = 0; r < nRows; r++) {
      result[r][c] = df.data[c][r];
    }
  }
  return result;
}

/** Convert a Series to a column DataFrame */
export function seriesAsDataFrame(name: string, series: Series): DataFrame {
  return { columns: [name], data: [series], shape: [series.length, 1] };
}

/**
 * OpenML dataset utilities.
 * Mirrors sklearn.datasets.fetch_openml.
 */

export interface OpenMLDataset {
  data: Float64Array[];
  target: Float64Array | Int32Array;
  featureNames: string[];
  targetNames: string[];
  description: string;
  details: Record<string, unknown>;
}

export interface FetchOpenMLOptions {
  name?: string;
  version?: number | "active";
  dataId?: number;
  dataHome?: string;
  targetColumn?: string | string[] | null;
  cacheDir?: string;
  returnX_y?: boolean;
  asFrame?: boolean;
  nRetries?: number;
  delay?: number;
  parser?: "auto" | "pandas" | "liac-arff";
}

const OPENML_BASE_URL = "https://api.openml.org/api/v1/json";

/**
 * Fetch a dataset from OpenML by name or ID.
 * Returns structured data suitable for machine learning.
 */
export async function fetchOpenML(
  options: FetchOpenMLOptions
): Promise<OpenMLDataset> {
  const { name, version = "active", dataId } = options;

  let url: string;
  if (dataId != null) {
    url = `${OPENML_BASE_URL}/data/${dataId}`;
  } else if (name != null) {
    url = `${OPENML_BASE_URL}/data/list/data_name/${encodeURIComponent(name)}/status/active/limit/1`;
  } else {
    throw new Error("fetchOpenML: must specify name or dataId");
  }

  let response: Response;
  try {
    response = await fetch(url);
  } catch (e) {
    throw new Error(`fetchOpenML: network error — ${String(e)}`);
  }

  if (!response.ok) {
    throw new Error(`fetchOpenML: HTTP ${response.status} for ${url}`);
  }

  const json = (await response.json()) as Record<string, unknown>;

  // Parse the dataset list to find the actual dataset ID
  let actualDataId = dataId;
  if (actualDataId == null) {
    const datasets = json["data"] as { dataset?: { did?: number }[] } | undefined;
    const did = datasets?.dataset?.[0]?.did;
    if (did == null) throw new Error(`fetchOpenML: dataset "${name}" not found`);
    actualDataId = did;
    void version; // version is used for filtering in production; simplified here
  }

  // Fetch dataset description
  const descResponse = await fetch(
    `${OPENML_BASE_URL}/data/${actualDataId}`
  );
  if (!descResponse.ok) {
    throw new Error(`fetchOpenML: HTTP ${descResponse.status} fetching dataset ${actualDataId}`);
  }
  const descJson = (await descResponse.json()) as {
    data_set_description?: {
      name?: string;
      description?: string;
      url?: string;
      row_id_attribute?: string;
      ignore_attribute?: string | string[];
      default_target_attribute?: string;
      feature?: Array<{ name: string; data_type: string }>;
    };
  };

  const desc = descJson.data_set_description ?? {};
  const description = desc.description ?? "";
  const targetCol =
    options.targetColumn ?? desc.default_target_attribute ?? "class";

  // Fetch the actual data file
  const dataUrl = desc.url;
  if (!dataUrl) throw new Error("fetchOpenML: no data URL in dataset description");

  const dataResponse = await fetch(dataUrl);
  if (!dataResponse.ok) {
    throw new Error(`fetchOpenML: HTTP ${dataResponse.status} fetching data file`);
  }
  const text = await dataResponse.text();
  return parseArff(text, targetCol as string, description, desc as Record<string, unknown>);
}

/**
 * Parse ARFF format into OpenMLDataset.
 */
export function parseArff(
  arffText: string,
  targetColumn: string,
  description = "",
  details: Record<string, unknown> = {}
): OpenMLDataset {
  const lines = arffText.split(/\r?\n/);
  const attributes: Array<{ name: string; type: string }> = [];
  let inData = false;
  const rows: string[][] = [];

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (line.startsWith("%") || line === "") continue;
    if (line.toLowerCase().startsWith("@attribute")) {
      const match = line.match(/@attribute\s+['"]?([^'"]+?)['"]?\s+(.*)/i);
      if (match) {
        attributes.push({ name: match[1]!.trim(), type: match[2]!.trim() });
      }
    } else if (line.toLowerCase().startsWith("@data")) {
      inData = true;
    } else if (inData) {
      rows.push(line.split(",").map((s) => s.trim()));
    }
  }

  const targetIdx = attributes.findIndex(
    (a) => a.name.toLowerCase() === targetColumn.toLowerCase()
  );
  const featureIdxs = attributes
    .map((_, i) => i)
    .filter((i) => i !== targetIdx);

  const featureNames = featureIdxs.map((i) => attributes[i]?.name ?? `f${i}`);
  const data: Float64Array[] = rows.map((row) =>
    new Float64Array(featureIdxs.map((i) => Number.parseFloat(row[i] ?? "0") || 0))
  );

  const targetAttr = targetIdx >= 0 ? attributes[targetIdx] : null;
  const targetType = targetAttr?.type ?? "NUMERIC";
  let target: Float64Array | Int32Array;

  if (
    targetType.toUpperCase().startsWith("NUMERIC") ||
    targetType.toUpperCase().startsWith("REAL") ||
    targetType.toUpperCase().startsWith("INTEGER")
  ) {
    target = new Float64Array(
      rows.map((row) => Number.parseFloat(row[targetIdx] ?? "0") || 0)
    );
  } else {
    // Nominal — encode as integers
    const vals = new Set(rows.map((row) => row[targetIdx] ?? ""));
    const valMap = new Map([...vals].map((v, i) => [v, i]));
    target = new Int32Array(
      rows.map((row) => valMap.get(row[targetIdx] ?? "") ?? 0)
    );
  }

  return {
    data,
    target,
    featureNames,
    targetNames: targetAttr ? [targetAttr.name] : [],
    description,
    details,
  };
}

/**
 * List available OpenML datasets matching the given criteria.
 */
export async function listOpenMLDatasets(options: {
  tag?: string;
  limit?: number;
  offset?: number;
} = {}): Promise<Array<{ id: number; name: string; version: number; status: string }>> {
  let url = `${OPENML_BASE_URL}/data/list`;
  const params: string[] = [];
  if (options.tag) params.push(`tag/${encodeURIComponent(options.tag)}`);
  if (params.length > 0) url += "/" + params.join("/");

  const response = await fetch(url);
  if (!response.ok) throw new Error(`listOpenMLDatasets: HTTP ${response.status}`);

  const json = (await response.json()) as {
    data?: {
      dataset?: Array<{ did: number; name: string; version: number; status: string }>;
    };
  };

  return (json.data?.dataset ?? [])
    .slice(0, options.limit ?? 100)
    .map((d) => ({
      id: d.did,
      name: d.name,
      version: d.version,
      status: d.status,
    }));
}

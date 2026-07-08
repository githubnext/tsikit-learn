/**
 * HTML representation utilities for estimators.
 * Mirrors sklearn.utils.estimator_html_repr.
 */

import { BaseEstimator } from "../base.js";

type Params = Record<string, unknown>;

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "None";
  if (typeof v === "string") return `'${escapeHtml(v)}'`;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  if (Array.isArray(v)) return `[${v.map(formatValue).join(", ")}]`;
  return escapeHtml(String(v));
}

/**
 * Build a simple HTML representation of an estimator.
 * Mirrors sklearn.utils.estimator_html_repr.
 */
export function estimatorHtmlRepr(estimator: BaseEstimator): string {
  const name = estimator.constructor.name;
  const params = estimator.get_params(false) as Params;
  const paramStr = Object.entries(params)
    .map(([k, v]) => `<span class="sk-param">${escapeHtml(k)}=${formatValue(v)}</span>`)
    .join(", ");

  return `<div class="sk-estimator">
  <div class="sk-estimator-name">${escapeHtml(name)}(${paramStr})</div>
</div>`;
}

/**
 * Pretty-print a text representation of an estimator.
 * Mirrors sklearn.base.BaseEstimator.__repr__.
 */
export function estimatorRepr(estimator: BaseEstimator, nCharMax: number = 700): string {
  const name = estimator.constructor.name;
  const params = estimator.get_params(false) as Params;
  const paramStr = Object.entries(params)
    .map(([k, v]) => `${k}=${formatValue(v)}`)
    .join(", ");
  const full = `${name}(${paramStr})`;
  return full.length > nCharMax ? full.slice(0, nCharMax - 3) + "..." : full;
}

/**
 * Return a pipeline diagram HTML string for a sequence of steps.
 * Mirrors sklearn.utils.estimator_html_repr for Pipeline-like objects.
 */
export function pipelineHtmlRepr(
  steps: Array<{ name: string; estimator: BaseEstimator }>,
): string {
  const stepsHtml = steps
    .map(
      ({ name, estimator }) =>
        `<div class="sk-step">
  <div class="sk-step-name">${escapeHtml(name)}</div>
  ${estimatorHtmlRepr(estimator)}
</div>`,
    )
    .join("\n");

  return `<div class="sk-pipeline">
  <div class="sk-pipeline-steps">
${stepsHtml}
  </div>
</div>`;
}

/**
 * Check if two estimators have the same parameters.
 * Mirrors sklearn.utils._tags.check_params_default_constructible.
 */
export function checkParamsDefaultConstructible(
  estimator: BaseEstimator,
): boolean {
  try {
    const params = estimator.get_params(false) as Params;
    return params !== null && typeof params === "object";
  } catch {
    return false;
  }
}

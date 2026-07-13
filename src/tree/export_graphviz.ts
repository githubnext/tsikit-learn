/**
 * Export decision trees to Graphviz DOT format.
 * Mirrors scikit-learn's tree.export_graphviz and tree.export_text.
 */

export interface TreeNode {
  feature: number;
  threshold: number;
  left: TreeNode | null;
  right: TreeNode | null;
  value: Float64Array;
  impurity: number;
  nSamples: number;
}

export interface ExportGraphvizOptions {
  featureNames?: string[];
  classNames?: string[];
  filled?: boolean;
  rounded?: boolean;
  precision?: number;
  maxDepth?: number;
}

/**
 * Export a decision tree in DOT format for visualization with Graphviz.
 */
export function exportGraphviz(
  tree: TreeNode,
  options: ExportGraphvizOptions = {},
): string {
  const {
    featureNames,
    classNames,
    filled = false,
    rounded = false,
    precision = 3,
    maxDepth,
  } = options;

  const nodeAttrs = ["shape=box"];
  if (rounded) nodeAttrs.push("style=rounded");
  if (filled) nodeAttrs.push('style="filled"');

  const lines: string[] = [
    "digraph Tree {",
    `node [${nodeAttrs.join(", ")}] ;`,
  ];

  let nodeId = 0;

  const writeNode = (node: TreeNode, depth: number): number => {
    const id = nodeId++;
    if (maxDepth !== undefined && depth > maxDepth) {
      lines.push(`${id} [label="(...)" shape=box] ;`);
      return id;
    }
    const isLeaf = node.left === null && node.right === null;
    let label: string;
    if (isLeaf) {
      const val = Array.from(node.value)
        .map((v) => v.toFixed(precision))
        .join(", ");
      const cls =
        classNames !== undefined
          ? `\\nclass = ${classNames[node.value.indexOf(Math.max(...Array.from(node.value)))] ?? "?"}`
          : "";
      label = `samples = ${node.nSamples}\\nvalue = [${val}]${cls}`;
    } else {
      const feat =
        featureNames !== undefined
          ? (featureNames[node.feature] ?? `X[${node.feature}]`)
          : `X[${node.feature}]`;
      label = `${feat} <= ${node.threshold.toFixed(precision)}\\nsamples = ${node.nSamples}\\nimpurity = ${node.impurity.toFixed(precision)}`;
    }
    lines.push(`${id} [label="${label}"] ;`);
    if (!isLeaf) {
      if (node.left !== null) {
        const leftId = writeNode(node.left, depth + 1);
        lines.push(
          `${id} -> ${leftId} [labeldistance=2.5, labelangle=45, headlabel="True"] ;`,
        );
      }
      if (node.right !== null) {
        const rightId = writeNode(node.right, depth + 1);
        lines.push(
          `${id} -> ${rightId} [labeldistance=2.5, labelangle=-45, headlabel="False"] ;`,
        );
      }
    }
    return id;
  };

  writeNode(tree, 0);
  lines.push("}");
  return lines.join("\n");
}

/**
 * Export a decision tree in ASCII text format.
 */
export function exportText(
  tree: TreeNode,
  options: {
    featureNames?: string[];
    maxDepth?: number;
    decimals?: number;
  } = {},
): string {
  const { featureNames, maxDepth, decimals = 2 } = options;
  const lines: string[] = [];

  const recurse = (node: TreeNode, depth: number): void => {
    if (maxDepth !== undefined && depth > maxDepth) return;
    const indent = "|   ".repeat(depth);
    if (node.left === null && node.right === null) {
      const predClass = node.value.indexOf(Math.max(...Array.from(node.value)));
      lines.push(`${indent}|--- class: ${predClass}`);
    } else {
      const feat =
        featureNames !== undefined
          ? (featureNames[node.feature] ?? `feature_${node.feature}`)
          : `feature_${node.feature}`;
      lines.push(
        `${indent}|--- ${feat} <= ${node.threshold.toFixed(decimals)}`,
      );
      if (node.left !== null) recurse(node.left, depth + 1);
      lines.push(`${indent}|--- ${feat} > ${node.threshold.toFixed(decimals)}`);
      if (node.right !== null) recurse(node.right, depth + 1);
    }
  };

  recurse(tree, 0);
  return lines.join("\n");
}

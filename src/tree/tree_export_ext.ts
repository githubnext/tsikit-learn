/**
 * Extended tree export utilities: exportMermaid, exportDotExt, costComplexityPrune
 */

export interface TreeNode {
  nodeId: number;
  featureIndex: number;
  threshold: number;
  impurity: number;
  nSamples: number;
  value: number[];
  leftChild: number | null;
  rightChild: number | null;
  isLeaf: boolean;
  className?: string;
}

export function exportMermaid(nodes: TreeNode[], featureNames?: string[]): string {
  const lines: string[] = ["flowchart TD"];
  for (const node of nodes) {
    const label = node.isLeaf
      ? `"Leaf\\nClass: ${node.className ?? node.value[0] ?? 0}\\nn=${node.nSamples}"`
      : `"Feature: ${featureNames?.[node.featureIndex] ?? `x${node.featureIndex}`}\\n≤ ${node.threshold.toFixed(3)}\\nn=${node.nSamples}"`;
    lines.push(`  node${node.nodeId}[${label}]`);
    if (node.leftChild !== null) {
      lines.push(`  node${node.nodeId} -->|Yes| node${node.leftChild}`);
    }
    if (node.rightChild !== null) {
      lines.push(`  node${node.nodeId} -->|No| node${node.rightChild}`);
    }
  }
  return lines.join("\n");
}

export function exportDotExt(
  nodes: TreeNode[],
  featureNames?: string[],
  classNames?: string[]
): string {
  const lines: string[] = [
    "digraph Tree {",
    '  node [shape=box, style="filled", color="black"];',
  ];
  for (const node of nodes) {
    const purity = 1 - node.impurity;
    const r = Math.round((1 - purity) * 255);
    const b = Math.round(purity * 255);
    const color = `"#${r.toString(16).padStart(2, "0")}00${b.toString(16).padStart(2, "0")}"`;
    if (node.isLeaf) {
      const cls = classNames?.[node.value[0] ?? 0] ?? String(node.value[0] ?? "?");
      lines.push(`  ${node.nodeId} [label="class = ${cls}\\nsamples = ${node.nSamples}\\nimpurity = ${node.impurity.toFixed(3)}", fillcolor=${color}];`);
    } else {
      const feat = featureNames?.[node.featureIndex] ?? `X[${node.featureIndex}]`;
      lines.push(`  ${node.nodeId} [label="${feat} <= ${node.threshold.toFixed(3)}\\nsamples = ${node.nSamples}\\nimpurity = ${node.impurity.toFixed(3)}", fillcolor=${color}];`);
    }
    if (node.leftChild !== null) lines.push(`  ${node.nodeId} -> ${node.leftChild} [label="True"];`);
    if (node.rightChild !== null) lines.push(`  ${node.nodeId} -> ${node.rightChild} [label="False"];`);
  }
  lines.push("}");
  return lines.join("\n");
}

export interface PrunedTree {
  nodes: TreeNode[];
  alpha: number;
  nLeaves: number;
}

export function costComplexityPrune(nodes: TreeNode[], ccp_alpha: number): PrunedTree {
  const nodesCopy = nodes.map((n) => ({ ...n }));
  const computeEffectiveAlpha = (nodeId: number): number => {
    const node = nodesCopy.find((n) => n.nodeId === nodeId);
    if (!node || node.isLeaf) return Number.POSITIVE_INFINITY;
    const leftAlpha = computeEffectiveAlpha(node.leftChild ?? -1);
    const rightAlpha = computeEffectiveAlpha(node.rightChild ?? -1);
    const totalImpurity = node.impurity * node.nSamples;
    const leftNode = nodesCopy.find((n) => n.nodeId === node.leftChild);
    const rightNode = nodesCopy.find((n) => n.nodeId === node.rightChild);
    const childImpurity = ((leftNode?.impurity ?? 0) * (leftNode?.nSamples ?? 0)) +
      ((rightNode?.impurity ?? 0) * (rightNode?.nSamples ?? 0));
    const nLeaves = countLeaves(node.nodeId, nodesCopy);
    const improvement = (totalImpurity - childImpurity) / node.nSamples;
    const nodeAlpha = nLeaves > 1 ? improvement / (nLeaves - 1) : Number.POSITIVE_INFINITY;
    return Math.min(nodeAlpha, leftAlpha, rightAlpha);
  };

  while (true) {
    const nonLeaves = nodesCopy.filter((n) => !n.isLeaf);
    if (nonLeaves.length === 0) break;
    const alphas = nonLeaves.map((n) => ({ node: n, alpha: computeEffectiveAlpha(n.nodeId) }));
    alphas.sort((a, b) => a.alpha - b.alpha);
    if (alphas[0] === undefined || alphas[0].alpha > ccp_alpha) break;
    alphas[0].node.isLeaf = true;
    alphas[0].node.leftChild = null;
    alphas[0].node.rightChild = null;
  }

  const nLeaves = nodesCopy.filter((n) => n.isLeaf).length;
  return { nodes: nodesCopy, alpha: ccp_alpha, nLeaves };
}

function countLeaves(nodeId: number, nodes: TreeNode[]): number {
  const node = nodes.find((n) => n.nodeId === nodeId);
  if (!node || node.isLeaf) return 1;
  return countLeaves(node.leftChild ?? -1, nodes) + countLeaves(node.rightChild ?? -1, nodes);
}

export function computeAlphasPath(nodes: TreeNode[]): Float64Array {
  const alphas: number[] = [0];
  const nonLeaves = nodes.filter((n) => !n.isLeaf);
  for (const node of nonLeaves) {
    const nLeaves = countLeaves(node.nodeId, nodes);
    if (nLeaves > 1) {
      const improvement = node.impurity * 0.1;
      alphas.push(improvement / (nLeaves - 1));
    }
  }
  return new Float64Array([...new Set(alphas)].sort((a, b) => a - b));
}

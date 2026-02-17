/**
 * Layer 4: Knowledge Graph.
 *
 * Persistent structural understanding of the project. Module boundaries,
 * dependency relationships, API contracts, data flow, ownership. Updated
 * after every completed task. Queryable by the Conductor when constructing
 * sub-agent prompts.
 *
 * Stored as JSON, not in the context window — pulled in selectively based
 * on which modules are relevant to the current task.
 *
 * Typical contribution to prompt: 1,000–4,000 tokens.
 */

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type { KnowledgeGraph, KnowledgeGraphNode } from "../types.js";

export type Layer4Config = {
  /** Path to persist the knowledge graph. */
  graphPath: string;
  /** Maximum nodes to include in a single prompt slice. */
  maxPromptNodes: number;
  /** Depth of dependency traversal for context building. */
  maxTraversalDepth: number;
};

const DEFAULT_CONFIG: Layer4Config = {
  graphPath: "knowledge-graph.json",
  maxPromptNodes: 30,
  maxTraversalDepth: 3,
};

export function createKnowledgeGraph(config?: Partial<Layer4Config>): {
  config: Layer4Config;
  graph: KnowledgeGraph;
} {
  return {
    config: { ...DEFAULT_CONFIG, ...config },
    graph: {
      version: 1,
      nodes: new Map(),
      lastFullScan: 0,
    },
  };
}

/**
 * Add or update a node in the knowledge graph.
 */
export function upsertNode(
  graph: KnowledgeGraph,
  node: Omit<KnowledgeGraphNode, "id" | "lastUpdated">,
): KnowledgeGraph {
  const id =
    node.path
      ? crypto.createHash("sha256").update(node.path).digest("hex").slice(0, 12)
      : crypto.createHash("sha256").update(`${node.type}:${node.name}`).digest("hex").slice(0, 12);

  const existing = graph.nodes.get(id);
  const updated: KnowledgeGraphNode = {
    ...node,
    id,
    lastUpdated: Date.now(),
    dependencies: mergeArrays(existing?.dependencies ?? [], node.dependencies),
    dependents: mergeArrays(existing?.dependents ?? [], node.dependents),
  };

  const nodes = new Map(graph.nodes);
  nodes.set(id, updated);

  return { ...graph, nodes };
}

/**
 * Remove a node and clean up references.
 */
export function removeNode(graph: KnowledgeGraph, nodeId: string): KnowledgeGraph {
  const nodes = new Map(graph.nodes);
  nodes.delete(nodeId);

  // Clean up references in other nodes
  for (const [id, node] of nodes) {
    if (node.dependencies.includes(nodeId) || node.dependents.includes(nodeId)) {
      nodes.set(id, {
        ...node,
        dependencies: node.dependencies.filter((d) => d !== nodeId),
        dependents: node.dependents.filter((d) => d !== nodeId),
      });
    }
  }

  return { ...graph, nodes };
}

/**
 * Query the knowledge graph for nodes relevant to specific files.
 * Returns the nodes directly matching the files plus their dependency
 * neighborhood up to maxTraversalDepth.
 */
export function queryByFiles(
  graph: KnowledgeGraph,
  files: string[],
  config: Layer4Config,
): KnowledgeGraphNode[] {
  const result = new Map<string, KnowledgeGraphNode>();
  const visited = new Set<string>();

  // Find direct matches
  for (const [id, node] of graph.nodes) {
    if (node.path && files.some((f) => node.path === f || f.includes(node.name))) {
      result.set(id, node);
    }
  }

  // Traverse dependencies and dependents up to maxTraversalDepth
  let frontier = new Set(result.keys());
  for (let depth = 0; depth < config.maxTraversalDepth && frontier.size > 0; depth++) {
    const nextFrontier = new Set<string>();
    for (const nodeId of frontier) {
      if (visited.has(nodeId)) continue;
      visited.add(nodeId);

      const node = graph.nodes.get(nodeId);
      if (!node) continue;

      for (const depId of [...node.dependencies, ...node.dependents]) {
        if (!visited.has(depId) && !result.has(depId)) {
          const depNode = graph.nodes.get(depId);
          if (depNode) {
            result.set(depId, depNode);
            nextFrontier.add(depId);
          }
        }
      }
    }
    frontier = nextFrontier;
  }

  // Limit to maxPromptNodes, prioritizing direct matches
  const nodes = Array.from(result.values());
  if (nodes.length > config.maxPromptNodes) {
    return nodes.slice(0, config.maxPromptNodes);
  }
  return nodes;
}

/**
 * Query by module/type for architectural context.
 */
export function queryByType(
  graph: KnowledgeGraph,
  type: KnowledgeGraphNode["type"],
): KnowledgeGraphNode[] {
  const result: KnowledgeGraphNode[] = [];
  for (const node of graph.nodes.values()) {
    if (node.type === type) {
      result.push(node);
    }
  }
  return result;
}

/**
 * Build a prompt fragment from a set of knowledge graph nodes.
 */
export function buildLayer4Prompt(nodes: KnowledgeGraphNode[]): string {
  if (nodes.length === 0) {
    return "";
  }

  const lines = ["=== PROJECT KNOWLEDGE (Layer 4) ===", ""];

  // Group by type for readability
  const grouped = new Map<string, KnowledgeGraphNode[]>();
  for (const node of nodes) {
    const list = grouped.get(node.type) ?? [];
    list.push(node);
    grouped.set(node.type, list);
  }

  for (const [type, typeNodes] of grouped) {
    lines.push(`[${type.toUpperCase()}S]`);
    for (const node of typeNodes) {
      const deps = node.dependencies.length > 0 ? ` → depends on: ${node.dependencies.join(", ")}` : "";
      const desc = node.description ? ` — ${node.description}` : "";
      const pathInfo = node.path ? ` (${node.path})` : "";
      lines.push(`  ${node.name}${pathInfo}${desc}${deps}`);
    }
    lines.push("");
  }

  lines.push("=== END PROJECT KNOWLEDGE ===");
  return lines.join("\n");
}

/**
 * Bootstrap the knowledge graph from a directory scan.
 * Creates module-level nodes from the file system structure.
 */
export function bootstrapFromDirectory(
  rootDir: string,
  graph: KnowledgeGraph,
): KnowledgeGraph {
  let updated = graph;

  const scanDir = (dir: string, depth: number) => {
    if (depth > 4) return; // Don't go too deep

    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }

    for (const entry of entries) {
      if (entry.name.startsWith(".") || entry.name === "node_modules" || entry.name === "dist") {
        continue;
      }

      const fullPath = path.join(dir, entry.name);
      const relativePath = path.relative(rootDir, fullPath);

      if (entry.isDirectory()) {
        // Create module node for directories with source files
        const hasSourceFiles = fs.readdirSync(fullPath).some(
          (f) => f.endsWith(".ts") || f.endsWith(".js") || f.endsWith(".py"),
        );

        if (hasSourceFiles) {
          updated = upsertNode(updated, {
            type: "module",
            name: entry.name,
            path: relativePath,
            description: `Module directory: ${relativePath}`,
            dependencies: [],
            dependents: [],
          });
        }

        scanDir(fullPath, depth + 1);
      } else if (
        entry.name.endsWith(".ts") ||
        entry.name.endsWith(".js") ||
        entry.name.endsWith(".py")
      ) {
        // Skip test files and config files for initial scan
        if (entry.name.includes(".test.") || entry.name.includes(".spec.")) {
          continue;
        }

        updated = upsertNode(updated, {
          type: "file",
          name: entry.name,
          path: relativePath,
          dependencies: [],
          dependents: [],
        });
      }
    }
  };

  scanDir(rootDir, 0);
  updated = { ...updated, lastFullScan: Date.now() };
  return updated;
}

/**
 * Persist the knowledge graph to disk.
 */
export function saveGraph(graph: KnowledgeGraph, filePath: string): void {
  const serializable = {
    version: graph.version,
    lastFullScan: graph.lastFullScan,
    nodes: Object.fromEntries(graph.nodes),
  };
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  fs.writeFileSync(filePath, JSON.stringify(serializable, null, 2), "utf-8");
}

/**
 * Load the knowledge graph from disk.
 */
export function loadGraph(filePath: string): KnowledgeGraph | null {
  if (!fs.existsSync(filePath)) {
    return null;
  }

  try {
    const raw = fs.readFileSync(filePath, "utf-8");
    const parsed = JSON.parse(raw) as {
      version: number;
      lastFullScan: number;
      nodes: Record<string, KnowledgeGraphNode>;
    };

    return {
      version: parsed.version,
      lastFullScan: parsed.lastFullScan,
      nodes: new Map(Object.entries(parsed.nodes)),
    };
  } catch {
    return null;
  }
}

function mergeArrays(a: string[], b: string[]): string[] {
  const set = new Set([...a, ...b]);
  return Array.from(set);
}

/**
 * Obsidian Adapter — File Watcher.
 *
 * Watches a designated Obsidian vault folder for new/modified `.md` files.
 * When a file matching the work order pattern is detected, it parses the
 * frontmatter via the Interface Agent and dispatches to the Conductor.
 *
 * This is the asynchronous "drop a file, get code" input path described
 * in the architecture. Users create markdown files in Obsidian, and the
 * Conductor picks them up automatically.
 *
 * Watch patterns:
 *   - inbox/*.md — new work orders
 *   - feedback/*.md — human feedback on completed tasks
 *   - constraints/*.md — new constraints to pin to Layer 0
 *
 * Uses Node's native fs.watch (no chokidar dependency) with debouncing
 * to handle rapid file saves.
 */

import fs from "node:fs";
import path from "node:path";
import type { HandoffMessage } from "../types.js";
import { parseObsidianInput } from "../agents/interface-agent.js";

export type ObsidianWatcherConfig = {
  /** Root path of the Obsidian vault (or the conductor subfolder). */
  vaultPath: string;
  /** Subdirectory names to watch. */
  watchDirs: string[];
  /** Debounce interval in ms. */
  debounceMs: number;
  /** File extensions to process. */
  extensions: string[];
  /** Move processed files to this subdirectory (relative to vaultPath). */
  processedDir: string;
  /** Whether to actually move files after processing (false = leave in place). */
  moveAfterProcessing: boolean;
};

const DEFAULT_CONFIG: ObsidianWatcherConfig = {
  vaultPath: "",
  watchDirs: ["inbox", "feedback", "constraints"],
  debounceMs: 500,
  extensions: [".md"],
  processedDir: "_processed",
  moveAfterProcessing: true,
};

export type WatcherCallback = (message: HandoffMessage) => void | Promise<void>;

export type WatcherHandle = {
  /** Stop watching all directories. */
  close: () => void;
  /** Number of files processed so far. */
  processedCount: number;
  /** Errors encountered during watching. */
  errors: Array<{ file: string; error: string; ts: number }>;
};

/**
 * Start watching an Obsidian vault for work orders.
 *
 * Returns a handle that can be used to stop watching and inspect state.
 */
export function startWatcher(
  config: Partial<ObsidianWatcherConfig> & { vaultPath: string },
  callback: WatcherCallback,
): WatcherHandle {
  const cfg: ObsidianWatcherConfig = { ...DEFAULT_CONFIG, ...config };
  const watchers: fs.FSWatcher[] = [];
  const debounceTimers = new Map<string, ReturnType<typeof setTimeout>>();
  const handle: WatcherHandle = {
    close: () => {
      for (const w of watchers) {
        try {
          w.close();
        } catch {
          // Ignore close errors
        }
      }
      for (const timer of debounceTimers.values()) {
        clearTimeout(timer);
      }
      debounceTimers.clear();
    },
    processedCount: 0,
    errors: [],
  };

  // Ensure vault path exists
  if (!fs.existsSync(cfg.vaultPath)) {
    throw new Error(`Obsidian vault path does not exist: ${cfg.vaultPath}`);
  }

  // Create watch directories if they don't exist
  for (const dir of cfg.watchDirs) {
    const fullDir = path.join(cfg.vaultPath, dir);
    if (!fs.existsSync(fullDir)) {
      fs.mkdirSync(fullDir, { recursive: true });
    }
  }

  // Create processed directory
  if (cfg.moveAfterProcessing) {
    const processedFull = path.join(cfg.vaultPath, cfg.processedDir);
    if (!fs.existsSync(processedFull)) {
      fs.mkdirSync(processedFull, { recursive: true });
    }
  }

  // Start watching each directory
  for (const dir of cfg.watchDirs) {
    const fullDir = path.join(cfg.vaultPath, dir);

    try {
      const watcher = fs.watch(fullDir, (eventType, filename) => {
        if (!filename) return;
        if (!cfg.extensions.some((ext) => filename.endsWith(ext))) return;

        const filePath = path.join(fullDir, filename);
        const key = filePath;

        // Debounce rapid saves
        const existing = debounceTimers.get(key);
        if (existing) clearTimeout(existing);

        debounceTimers.set(
          key,
          setTimeout(() => {
            debounceTimers.delete(key);
            processFile(filePath, filename, dir, cfg, callback, handle);
          }, cfg.debounceMs),
        );
      });

      watchers.push(watcher);
    } catch (err) {
      handle.errors.push({
        file: fullDir,
        error: `Failed to watch directory: ${String(err)}`,
        ts: Date.now(),
      });
    }
  }

  // Process any existing files in watch directories (catch up on startup)
  for (const dir of cfg.watchDirs) {
    const fullDir = path.join(cfg.vaultPath, dir);
    try {
      const existing = fs.readdirSync(fullDir);
      for (const filename of existing) {
        if (!cfg.extensions.some((ext) => filename.endsWith(ext))) continue;
        const filePath = path.join(fullDir, filename);
        processFile(filePath, filename, dir, cfg, callback, handle);
      }
    } catch {
      // Directory may not exist yet
    }
  }

  return handle;
}

/**
 * Scan a vault directory once (no persistent watcher).
 * Useful for batch processing or testing.
 */
export function scanOnce(
  config: Partial<ObsidianWatcherConfig> & { vaultPath: string },
): HandoffMessage[] {
  const cfg: ObsidianWatcherConfig = { ...DEFAULT_CONFIG, ...config };
  const messages: HandoffMessage[] = [];

  for (const dir of cfg.watchDirs) {
    const fullDir = path.join(cfg.vaultPath, dir);
    if (!fs.existsSync(fullDir)) continue;

    const files = fs.readdirSync(fullDir);
    for (const filename of files) {
      if (!cfg.extensions.some((ext) => filename.endsWith(ext))) continue;

      const filePath = path.join(fullDir, filename);
      try {
        const content = fs.readFileSync(filePath, "utf-8");
        const message = parseObsidianInput(content, filename);
        messages.push(message);
      } catch {
        // Skip unreadable files
      }
    }
  }

  return messages;
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

function processFile(
  filePath: string,
  filename: string,
  dir: string,
  cfg: ObsidianWatcherConfig,
  callback: WatcherCallback,
  handle: WatcherHandle,
): void {
  try {
    // Check file still exists (may have been moved/deleted)
    if (!fs.existsSync(filePath)) return;

    const content = fs.readFileSync(filePath, "utf-8");

    // Skip empty files
    if (content.trim().length === 0) return;

    // Parse through the Interface Agent
    const message = parseObsidianInput(content, filename);

    // Override intent based on directory
    if (dir === "feedback") {
      message.intent = "feedback";
    } else if (dir === "constraints") {
      message.intent = "constraint-update";
    }

    // Dispatch
    callback(message);
    handle.processedCount++;

    // Move to processed directory if configured
    if (cfg.moveAfterProcessing) {
      const processedPath = path.join(
        cfg.vaultPath,
        cfg.processedDir,
        `${Date.now()}-${filename}`,
      );
      fs.renameSync(filePath, processedPath);
    }
  } catch (err) {
    handle.errors.push({
      file: filePath,
      error: String(err),
      ts: Date.now(),
    });
  }
}

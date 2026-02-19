import { captureEnv } from "../test-utils/env.js";

export function snapshotStateDirEnv() {
  return captureEnv(["MAISTRO_STATE_DIR", "MAISTRO_STATE_DIR"]);
}

export function restoreStateDirEnv(snapshot: ReturnType<typeof snapshotStateDirEnv>): void {
  snapshot.restore();
}

export function setStateDirEnv(stateDir: string): void {
  process.env.MAISTRO_STATE_DIR = stateDir;
  delete process.env.MAISTRO_STATE_DIR;
}

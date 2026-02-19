import type { MaistroConfig } from "../config/config.js";

export function applyOnboardingLocalWorkspaceConfig(
  baseConfig: MaistroConfig,
  workspaceDir: string,
): MaistroConfig {
  return {
    ...baseConfig,
    agents: {
      ...baseConfig.agents,
      defaults: {
        ...baseConfig.agents?.defaults,
        workspace: workspaceDir,
      },
    },
    gateway: {
      ...baseConfig.gateway,
      mode: "local",
    },
  };
}

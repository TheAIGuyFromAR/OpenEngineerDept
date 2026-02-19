import type { MaistroConfig } from "./config.js";

export function ensurePluginAllowlisted(cfg: MaistroConfig, pluginId: string): MaistroConfig {
  const allow = cfg.plugins?.allow;
  if (!Array.isArray(allow) || allow.includes(pluginId)) {
    return cfg;
  }
  return {
    ...cfg,
    plugins: {
      ...cfg.plugins,
      allow: [...allow, pluginId],
    },
  };
}

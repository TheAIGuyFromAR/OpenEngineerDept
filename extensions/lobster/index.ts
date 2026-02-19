import type {
  AnyAgentTool,
  MaistroPluginApi,
  MaistroPluginToolFactory,
} from "../../src/plugins/types.js";
import { createLobsterTool } from "./src/lobster-tool.js";

export default function register(api: MaistroPluginApi) {
  api.registerTool(
    ((ctx) => {
      if (ctx.sandboxed) {
        return null;
      }
      return createLobsterTool(api) as AnyAgentTool;
    }) as MaistroPluginToolFactory,
    { optional: true },
  );
}

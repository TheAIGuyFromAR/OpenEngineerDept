"""
Prompt Management — Langfuse-backed with local git sync.

Prompts live in two places:
  1. Langfuse (source of truth for versioning, A/B testing, rollback)
  2. Local files in prompts/ (git-tracked, fallback when Langfuse is down)

Flow:
  - On startup: pull latest production prompts from Langfuse → write to local files
  - On prompt fetch: try Langfuse first, fall back to local files
  - sync-prompts.sh: push local file edits → Langfuse (for git-first workflow)

Prompt names follow the convention: {agent}.{action}
  e.g. "planner.decompose", "coder.generate", "reviewer.score"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default prompts directory (git-tracked alongside code)
_DEFAULT_PROMPTS_DIR = Path(__file__).parent / "templates"


class PromptManager:
    """Manages prompt templates via Langfuse with local file fallback."""

    def __init__(
        self,
        prompts_dir: str | Path | None = None,
        langfuse_enabled: bool = True,
    ) -> None:
        self._prompts_dir = Path(prompts_dir) if prompts_dir else _DEFAULT_PROMPTS_DIR
        self._prompts_dir.mkdir(parents=True, exist_ok=True)
        self._langfuse = None
        self._langfuse_enabled = langfuse_enabled
        self._cache: dict[str, dict] = {}

    def _get_langfuse(self):
        """Lazy-init Langfuse client."""
        if not self._langfuse_enabled:
            return None
        if self._langfuse is not None:
            return self._langfuse

        try:
            from langfuse import Langfuse

            client = Langfuse()
            client.auth_check()
            self._langfuse = client
            logger.info("Langfuse prompt management enabled")
            return client
        except ImportError:
            logger.info("Langfuse SDK not installed — using local prompts only")
            self._langfuse_enabled = False
        except Exception as exc:
            logger.warning("Langfuse not reachable for prompts: %s", exc)

        return None

    def get_prompt(
        self,
        name: str,
        *,
        variables: dict[str, str] | None = None,
        label: str = "production",
    ) -> str:
        """
        Fetch a prompt by name, compile with variables.

        Tries Langfuse first, falls back to local template file.
        """
        # Try Langfuse
        lf = self._get_langfuse()
        if lf is not None:
            try:
                prompt_obj = lf.get_prompt(name, label=label)
                template = prompt_obj.prompt

                # Cache locally for fallback
                self._write_local(name, template, prompt_obj.config)

                if variables:
                    return prompt_obj.compile(**variables)
                return template

            except Exception as exc:
                logger.debug("Langfuse prompt fetch failed for %r: %s", name, exc)

        # Fall back to local file
        return self._read_local(name, variables)

    def get_chat_prompt(
        self,
        name: str,
        *,
        variables: dict[str, str] | None = None,
        label: str = "production",
    ) -> list[dict[str, str]]:
        """
        Fetch a chat-format prompt (list of messages) by name.

        Tries Langfuse first, falls back to local template file.
        """
        lf = self._get_langfuse()
        if lf is not None:
            try:
                prompt_obj = lf.get_prompt(name, label=label, type="chat")

                # Cache locally
                self._write_local(
                    name,
                    json.dumps(prompt_obj.prompt, indent=2),
                    prompt_obj.config,
                )

                if variables:
                    return prompt_obj.compile(**variables)
                return prompt_obj.prompt

            except Exception as exc:
                logger.debug("Langfuse chat prompt fetch failed for %r: %s", name, exc)

        # Fall back to local file
        text = self._read_local(name, variables)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            # Single string → wrap as user message
            return [{"role": "user", "content": text}]

    def sync_from_langfuse(self, label: str = "production") -> list[str]:
        """
        Pull all prompts with the given label from Langfuse → local files.

        Returns list of synced prompt names.
        """
        lf = self._get_langfuse()
        if lf is None:
            logger.warning("Cannot sync: Langfuse not available")
            return []

        synced = []
        try:
            # Langfuse SDK: list prompts (paginated)
            prompts = lf.client.prompts.list()

            for prompt_meta in prompts.data:
                name = prompt_meta.name
                try:
                    prompt_obj = lf.get_prompt(name, label=label)
                    content = prompt_obj.prompt
                    if isinstance(content, list):
                        content = json.dumps(content, indent=2)

                    self._write_local(name, content, prompt_obj.config)
                    synced.append(name)
                    logger.debug("Synced prompt: %s", name)
                except Exception as exc:
                    logger.warning("Failed to sync prompt %r: %s", name, exc)

        except Exception as exc:
            logger.warning("Failed to list Langfuse prompts: %s", exc)

        return synced

    def sync_to_langfuse(self) -> list[str]:
        """
        Push all local prompt files → Langfuse.

        This is the git-first workflow: edit prompts locally, commit,
        then push to Langfuse for versioning/A/B testing.

        Returns list of pushed prompt names.
        """
        lf = self._get_langfuse()
        if lf is None:
            logger.warning("Cannot push: Langfuse not available")
            return []

        pushed = []
        for path in sorted(self._prompts_dir.glob("*.txt")):
            name = path.stem.replace("__", ".")  # planner__decompose → planner.decompose
            content = path.read_text(encoding="utf-8").strip()

            # Check for config sidecar
            config_path = path.with_suffix(".json")
            config = {}
            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass

            try:
                # Try chat format (JSON array)
                messages = json.loads(content)
                if isinstance(messages, list):
                    lf.create_prompt(
                        name=name,
                        prompt=messages,
                        config=config,
                        type="chat",
                        labels=["production"],
                    )
                    pushed.append(name)
                    continue
            except (json.JSONDecodeError, TypeError):
                pass

            # Text format
            lf.create_prompt(
                name=name,
                prompt=content,
                config=config,
                type="text",
                labels=["production"],
            )
            pushed.append(name)

        return pushed

    def list_local_prompts(self) -> list[str]:
        """List all locally available prompt template names."""
        names = []
        for path in sorted(self._prompts_dir.glob("*.txt")):
            names.append(path.stem.replace("__", "."))
        return names

    def _write_local(self, name: str, content: str, config: dict | None = None) -> None:
        """Write a prompt template to local file."""
        # Convert dots to double underscores for filesystem safety
        filename = name.replace(".", "__")
        path = self._prompts_dir / f"{filename}.txt"
        path.write_text(content, encoding="utf-8")

        # Write config sidecar if present
        if config:
            config_path = path.with_suffix(".json")
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    def _read_local(self, name: str, variables: dict[str, str] | None = None) -> str:
        """Read a prompt from local file, apply variable substitution."""
        filename = name.replace(".", "__")
        path = self._prompts_dir / f"{filename}.txt"

        if not path.exists():
            logger.warning("Prompt not found locally: %s (looked at %s)", name, path)
            return ""

        content = path.read_text(encoding="utf-8").strip()

        if variables:
            for key, value in variables.items():
                content = content.replace("{{" + key + "}}", value)

        return content

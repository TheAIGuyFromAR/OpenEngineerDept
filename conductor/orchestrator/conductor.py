"""
Conductor Orchestrator — Core Loop.

The main event loop that ties everything together:

1. Watch Obsidian inbox for tasks
2. Load project context via gateway
3. Planner decomposes task
4. For each subtask:
   - Estimate tier (heuristic)
   - Ultra Think generate candidates
   - Reviewer scores and selects
   - Apply candidate via file_ops
   - Run tests
   - Retry/escalate if needed
5. Write changelog entry
6. Record training data
7. Move task to completed/failed in Obsidian
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

import httpx
from rich.console import Console
from rich.logging import RichHandler

from .config import OrchestratorConfig
from .planner import Planner
from .coder import Coder
from .reviewer import Reviewer
from .memory.layer0 import Layer0
from .memory.layer1 import Layer1
from .memory.layer2 import Layer2
from .memory.changelog import Changelog, ChangelogEntry
from .memory.knowledge_graph import KnowledgeGraph
from .tools.file_ops import FileOps
from .tools.shell import Shell
from .tools.git import Git
from .tools.test_runner import TestRunner
from .training.data_collector import DataCollector, TrainingRow, CandidateRecord
from .training.exemplar_library import ExemplarLibrary, Exemplar
from .interfaces.obsidian_watcher import ObsidianWatcher
from .interfaces.vault_sync import create_sync_adapter
from .agents.intent_router import IntentRouter, Intent
from .agents.abra import Abra, DeviceRegistry

logger = logging.getLogger(__name__)
console = Console()


class Conductor:
    def __init__(self, config: OrchestratorConfig) -> None:
        self._config = config

        # Memory stack
        self._layer0 = Layer0(config.layer0_path)
        self._layer1 = Layer1(max_tokens=config.max_working_memory_tokens)
        self._layer2 = Layer2()
        self._changelog = Changelog(
            log_path=str(Path(config.training_data_dir).parent / "changelog.jsonl")
        )
        self._knowledge = KnowledgeGraph()

        # Agents
        self._intent_router = IntentRouter(gateway_url=config.gateway_url)
        self._abra = Abra(
            registry=DeviceRegistry(
                alexa_map=config.ha_alexa_device_map or {},
            ),
            ha_url=config.ha_url,
            ha_token=config.ha_token,
        )
        self._planner = Planner(config.gateway_url)
        self._coder = Coder(config.gateway_url)
        self._reviewer = Reviewer(config.gateway_url, config.accept_threshold)

        # Tools
        self._file_ops = FileOps(config.project_dir)
        self._shell = Shell(config.project_dir)
        self._git = Git(config.project_dir)
        self._test_runner = TestRunner(config.project_dir)

        # Training
        self._data_collector = DataCollector(config.training_data_dir)
        self._exemplar_library = ExemplarLibrary(config.exemplar_library_dir)

        # Vault sync adapter (local, git, syncthing, or couchdb)
        sync_adapter = create_sync_adapter(
            mode=config.vault_sync_mode,
            vault_path=config.obsidian_vault,
            git_remote=config.vault_sync_git_remote,
            git_branch=config.vault_sync_git_branch,
            syncthing_api=config.vault_sync_syncthing_api,
            syncthing_api_key=config.vault_sync_syncthing_api_key,
            syncthing_folder_id=config.vault_sync_syncthing_folder_id,
            couchdb_url=config.vault_sync_couchdb_url,
            couchdb_database=config.vault_sync_couchdb_database,
            couchdb_username=config.vault_sync_couchdb_username,
            couchdb_password=config.vault_sync_couchdb_password,
            couchdb_conductor_prefix=config.vault_sync_couchdb_conductor_prefix,
        )

        # Obsidian interface
        self._watcher = ObsidianWatcher(
            vault_path=config.obsidian_vault,
            layer0_path=config.layer0_path,
            on_new_task=self._handle_new_task,
            on_constraints_changed=self._handle_constraints_changed,
            sync_adapter=sync_adapter,
        )

        # Task queue
        self._task_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._running = False

    async def start(self) -> None:
        """Start the conductor main loop."""
        self._running = True
        loop = asyncio.get_running_loop()

        # Start Obsidian watcher
        self._watcher.start(loop)

        # Load project context into gateway
        await self._load_project_context()

        # Sync HA entities for Abra (if configured)
        if self._config.ha_url and self._config.ha_sync_entities:
            try:
                count = await self._abra._registry.sync_from_ha(
                    ha_url=self._config.ha_url,
                    ha_token=self._config.ha_token,
                    alexa_map=self._config.ha_alexa_device_map,
                )
                console.print(f"  HA entities synced: {count}")
            except Exception as exc:
                logger.warning("HA entity sync failed (using defaults): %s", exc)

        # Process any pending tasks in inbox (sync + scan)
        pending = await self._watcher.list_pending()
        for filename, content in pending:
            await self._task_queue.put((filename, content))

        console.print(f"[bold green]Conductor started[/] — project: {self._config.project_id}")
        console.print(f"  Gateway: {self._config.gateway_url}")
        console.print(f"  Inbox: {self._config.obsidian_vault}/conductor/inbox/")
        console.print(f"  Acceptance rate: {self._changelog.acceptance_rate():.0%}")

        # Main loop
        try:
            while self._running:
                try:
                    filename, content = await asyncio.wait_for(
                        self._task_queue.get(), timeout=5.0
                    )
                    await self._process_task(filename, content)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Shutdown gracefully."""
        self._running = False
        self._watcher.stop()
        await self._intent_router.close()
        await self._abra.close()
        await self._planner.close()
        await self._coder.close()
        await self._reviewer.close()
        console.print("[bold yellow]Conductor stopped[/]")

    # ------------------------------------------------------------------
    # Core task processing
    # ------------------------------------------------------------------

    async def _process_task(self, filename: str, task_text: str) -> None:
        """Process a single task from inbox."""
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        console.print(f"\n[bold blue]Processing:[/] {filename} ({task_id})")

        try:
            # Step 0: Intent routing (bouncer)
            console.print("  [dim]Routing intent...[/]")
            routing = await self._intent_router.route(task_text)

            if routing.intent == Intent.DENIED:
                console.print(f"  [bold red]Denied:[/] {routing.denial_reason}")
                await self._watcher.write_failed(
                    filename, f"Denied: {routing.denial_reason}"
                )
                return

            if routing.intent == Intent.UNCLEAR:
                console.print(f"  [bold yellow]Unclear intent — asking for clarification[/]")
                await self._watcher.write_failed(
                    filename, f"Needs clarification:\n{routing.clarification_prompt}"
                )
                return

            if routing.intent == Intent.HOME_AUTOMATION:
                console.print(f"  [bold cyan]Home automation → Abra[/]")
                await self._handle_home_automation(filename, routing)
                return

            if routing.intent == Intent.ARTIFACT:
                console.print(f"  [bold magenta]Artifact creation[/]")
                # TODO: route to artifact agent when implemented
                await self._watcher.write_failed(
                    filename, f"Artifact creation not yet wired (agent: {routing.agent_name}). Task: {routing.rewritten_task}"
                )
                return

            if routing.intent == Intent.CONVERSATION:
                console.print(f"  [bold white]Conversation[/]")
                # TODO: route to conversation handler
                await self._watcher.write_failed(
                    filename, f"Conversation handling not yet wired. Input: {task_text[:200]}"
                )
                return

            # CODE or ANALYSIS → existing pipeline
            console.print(f"  [dim]Intent: {routing.intent.value} → {routing.agent_name} (confidence: {routing.confidence:.0%})[/]")
            effective_task = routing.rewritten_task or task_text

            # Build full context
            context = self._build_context()

            # Step 1: Plan
            console.print("  [dim]Planning...[/]")
            plan = await self._planner.decompose(task_id, effective_task, context)
            self._layer1.start_task(effective_task, plan.summary)

            for st in plan.subtasks:
                self._layer1.add_subtask(st.subtask_id, st.description)

            console.print(f"  Plan: {len(plan.subtasks)} subtasks")

            # Step 2: Execute each subtask
            all_passed = True
            for subtask in plan.subtasks:
                success = await self._execute_subtask(
                    task_id=task_id,
                    subtask_id=subtask.subtask_id,
                    description=subtask.description,
                    tier=subtask.tier,
                )
                if not success:
                    all_passed = False
                    break

            # Step 3: Finalize
            if all_passed:
                await self._watcher.write_completed(filename, f"Task {task_id} completed successfully.")
                console.print(f"  [bold green]Completed:[/] {task_id}")
            else:
                await self._watcher.write_failed(filename, f"Task {task_id} failed after retries.")
                console.print(f"  [bold red]Failed:[/] {task_id}")

        except Exception as exc:
            logger.exception("Task %s failed with exception", task_id)
            await self._watcher.write_failed(filename, f"Exception: {exc}")
            console.print(f"  [bold red]Error:[/] {exc}")

    async def _handle_home_automation(self, filename: str, routing) -> None:
        """Route a home automation intent through Abra.

        Abra resolves the room from the source device, discovers devices,
        interprets the comfort intent, checks environment, and builds
        the HA service calls.
        """
        task_text = routing.rewritten_task or routing.raw_input

        # Extract source device from metadata if available
        # (Alexa skill would attach this; for now check raw_input for hints)
        source_device_id = getattr(routing, "source_device_id", "")
        area_id = getattr(routing, "area_id", "")

        result = await self._abra.handle(
            utterance=task_text,
            source_device_id=source_device_id,
            area_id=area_id,
        )

        if not result.success:
            console.print(f"  [yellow]Abra failed:[/] {result.error}")
            await self._watcher.write_failed(
                filename, f"Abra: {result.error}"
            )
            return

        console.print(f"  [cyan]Room:[/] {result.room_resolved}")
        console.print(f"  [cyan]Reasoning:[/] {result.reasoning}")
        for call in result.service_calls:
            console.print(
                f"  [cyan]→[/] {call.domain}.{call.service}({call.entity_id}"
                + (f", {call.data}" if call.data else "")
                + ")"
            )

        # Execute the calls against HA
        responses = await self._abra.execute(result)
        console.print(f"  [bold green]Executed {len(responses)} service call(s)[/]")

        # Write completion
        summary_lines = [
            f"Room: {result.room_resolved}",
            f"Reasoning: {result.reasoning}",
        ]
        for call in result.service_calls:
            summary_lines.append(f"  {call.domain}.{call.service}({call.entity_id})")
        await self._watcher.write_completed(filename, "\n".join(summary_lines))

    async def _execute_subtask(
        self,
        *,
        task_id: str,
        subtask_id: str,
        description: str,
        tier: int,
    ) -> bool:
        """Execute a single subtask with retry loop."""
        self._layer1.start_subtask(subtask_id)
        context = self._build_context()

        for attempt in range(self._config.max_retries + 1):
            console.print(
                f"    [{subtask_id}] Tier {tier}, attempt {attempt + 1}"
            )

            # Generate candidates
            coder_result = await self._coder.generate(
                subtask_id=subtask_id,
                subtask_description=description,
                context=context,
                tier=tier,
                project_id=self._config.project_id,
            )

            if not coder_result.candidates:
                console.print(f"    [red]No candidates generated[/]")
                if coder_result.errors:
                    console.print(f"    Errors: {coder_result.errors}")
                continue

            # Review candidates
            candidate_texts = [c.content for c in coder_result.candidates]
            review = await self._reviewer.review(
                subtask_id=subtask_id,
                subtask_description=description,
                candidates=candidate_texts,
                context=context,
            )

            console.print(
                f"    Selected candidate {review.selected_idx} "
                f"(score={review.selected_score:.1f})"
            )

            # Check threshold
            if review.selected_score < self._reviewer.accept_threshold:
                console.print(f"    [yellow]Below threshold ({self._reviewer.accept_threshold})[/]")
                self._layer1.add_feedback(review.feedback_summary)

                # Escalate tier on retry
                if tier < 3:
                    tier += 1
                    console.print(f"    Escalating to Tier {tier}")

                # Record training data even for failures
                self._record_training(
                    task_id, subtask_id, tier, coder_result, review, False, ""
                )
                continue

            # Apply the selected candidate
            selected_content = candidate_texts[review.selected_idx]
            self._apply_candidate(selected_content)

            # Run tests
            test_result = await self._test_runner.run()
            console.print(
                f"    Tests: {'PASS' if test_result.success else 'FAIL'} "
                f"({test_result.tests_passed}/{test_result.tests_total})"
            )

            # Record training data
            self._record_training(
                task_id, subtask_id, tier, coder_result, review,
                test_result.success, test_result.output[:200],
            )

            if test_result.success:
                # Record changelog
                self._changelog.append(
                    ChangelogEntry(
                        task_id=task_id,
                        project_id=self._config.project_id,
                        description=description,
                        tier_used=tier,
                        candidates_generated=len(coder_result.candidates),
                        accepted_candidate_idx=review.selected_idx,
                        reviewer_score=review.selected_score,
                        test_passed=True,
                        retries=attempt,
                    )
                )

                # Store as exemplar if score is high enough
                if review.selected_score >= 8.0:
                    self._exemplar_library.add(
                        Exemplar(
                            task_type=self._classify_task(description),
                            description=description[:200],
                            prompt=description,
                            solution=selected_content[:2000],
                            reviewer_score=review.selected_score,
                            project_id=self._config.project_id,
                            tags=[],
                        )
                    )

                self._layer1.complete_subtask(
                    subtask_id, selected_content[:100]
                )
                return True

            # Test failed — add feedback and retry
            self._layer1.add_feedback(
                f"Tests failed: {test_result.output[:200]}"
            )
            if tier < 3:
                tier += 1

        # All retries exhausted
        self._layer1.fail_subtask(subtask_id, "Max retries exhausted")
        return False

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def _build_context(self) -> str:
        """Assemble the full prompt context from memory layers."""
        parts = [
            self._layer0.build_prompt_section(),
            self._layer1.build_prompt_section(),
            self._layer2.build_prompt_section(),
            self._knowledge.build_prompt_section(),
        ]
        return "\n\n".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # Candidate application
    # ------------------------------------------------------------------

    def _apply_candidate(self, content: str) -> bool:
        """Apply a code candidate to the project files."""
        # Parse the candidate output for file operations
        # Convention: === NEW FILE: path === or unified diff
        lines = content.splitlines()
        current_file: str | None = None
        current_content: list[str] = []

        for line in lines:
            if line.startswith("=== NEW FILE:") and line.endswith("==="):
                # Flush previous
                if current_file:
                    self._file_ops.write(current_file, "\n".join(current_content))
                current_file = line.split(":", 1)[1].rsplit("===", 1)[0].strip()
                current_content = []
            elif current_file is not None:
                current_content.append(line)

        # Flush last file
        if current_file:
            self._file_ops.write(current_file, "\n".join(current_content))

        return True

    # ------------------------------------------------------------------
    # Training data
    # ------------------------------------------------------------------

    def _record_training(
        self,
        task_id: str,
        subtask_id: str,
        tier: int,
        coder_result,
        review,
        test_passed: bool,
        test_output: str,
    ) -> None:
        candidates = [
            CandidateRecord(
                content=c.content[:2000],
                sampling_params=c.sampling_params,
                reviewer_score=(
                    review.scores[i].overall
                    if i < len(review.scores)
                    else 0.0
                ),
                tokens_generated=c.tokens_generated,
            )
            for i, c in enumerate(coder_result.candidates)
        ]

        row = TrainingRow(
            task_id=task_id,
            subtask_id=subtask_id,
            project_id=self._config.project_id,
            tier=tier,
            candidates=candidates,
            test_passed=test_passed,
            test_output_summary=test_output,
            accepted_candidate_idx=review.selected_idx,
            prompt_hash=DataCollector.hash_content(self._layer1.build_prompt_section()),
            context_hash=self._layer0.content_hash,
        )
        self._data_collector.record(row)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _handle_new_task(self, filename: str, content: str) -> None:
        """Called by ObsidianWatcher when a new task file appears."""
        console.print(f"[cyan]New task:[/] {filename}")
        await self._task_queue.put((filename, content))

    async def _handle_constraints_changed(self) -> None:
        """Called when Layer 0 constraints file is modified."""
        console.print("[yellow]Constraints changed — reloading and invalidating cache[/]")
        self._layer0.reload()
        await self._load_project_context()

    async def _load_project_context(self) -> None:
        """Load/reload project context into the gateway."""
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{self._config.gateway_url}/v1/project/load",
                    json={
                        "project_id": self._config.project_id,
                        "layer0_text": self._layer0.content,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                console.print(f"  Context loaded: {data.get('action', 'unknown')}")
        except Exception as exc:
            logger.error("Failed to load project context: %s", exc)

    @staticmethod
    def _classify_task(description: str) -> str:
        """Simple heuristic task type classification."""
        desc_lower = description.lower()
        if any(w in desc_lower for w in ["fix", "bug", "error", "crash"]):
            return "bugfix"
        if any(w in desc_lower for w in ["test", "spec", "assert"]):
            return "test"
        if any(w in desc_lower for w in ["refactor", "rename", "extract", "clean"]):
            return "refactor"
        return "feature"


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Conductor Orchestrator")
    parser.add_argument(
        "--config", required=True, help="Path to conductor.yaml"
    )
    parser.add_argument(
        "--project", required=True, help="Project ID"
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    config = OrchestratorConfig.from_yaml(args.config)
    if args.project:
        config.project_id = args.project

    conductor = Conductor(config)

    try:
        asyncio.run(conductor.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/]")


if __name__ == "__main__":
    main()

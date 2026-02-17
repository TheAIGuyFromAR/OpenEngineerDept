"""Orchestrator configuration — loaded from conductor.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class OrchestratorConfig(BaseModel):
    project_id: str
    project_dir: str
    obsidian_vault: str
    gateway_url: str = "http://localhost:9090"
    inference_url: str = "http://localhost:8080"
    max_retries: int = 3
    accept_threshold: float = 7.0
    max_working_memory_tokens: int = 8000
    layer0_path: str = "./constraints.md"
    training_data_dir: str = "./data/training"
    exemplar_library_dir: str = "./data/exemplars"

    @classmethod
    def from_yaml(cls, path: str) -> OrchestratorConfig:
        """Load config from a YAML file."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

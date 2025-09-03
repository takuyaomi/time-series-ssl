from dataclasses import dataclass
from typing import Any, Dict
import yaml

@dataclass
class Config:
    data_root: str
    model: Dict[str, Any]
    ssl: Dict[str, Any] | None = None
    train: Dict[str, Any] | None = None
    seed: int = 42

    @staticmethod
    def load(path: str, overrides: Dict[str, Any] | None = None) -> "Config":
        with open(path, 'r') as f:
            raw = yaml.safe_load(f) or {}

        # ensure containers exist
        if not isinstance(raw.get('model'), dict):
            raw['model'] = {}
        if not isinstance(raw.get('train'), dict):
            raw['train'] = {}
        if raw.get('ssl') is None:
            raw['ssl'] = None

        # apply overrides (if any) with shallow merge for dicts
        if overrides:
            for k, v in overrides.items():
                if isinstance(v, dict) and isinstance(raw.get(k), dict):
                    raw[k].update(v)
                else:
                    raw[k] = v

        # required fields with safe defaults; may be overwritten later by CLI
        if 'data_root' not in raw:
            raw['data_root'] = ''
        if 'seed' not in raw:
            raw['seed'] = 42

        return Config(**raw)
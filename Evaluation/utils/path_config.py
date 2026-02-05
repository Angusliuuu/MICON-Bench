from pathlib import Path
from typing import Dict, Any
import yaml


DEFAULT_PATHS = {
    "data_root": "data",
    "templates_root": "templates",
    "prompts_root": "prompts",
    "outputs_root": "outputs",
    "reports_root": "outputs/reports",
    "schema_file": "prompts/judge/schema.json",
    "generations_root": "outputs/generations",
    "eval_results_root": "outputs/eval_res",
}


def load_global_config(project_root: Path, config_path: str = "configs/global.yaml") -> Dict[str, Any]:
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = project_root / config_path
    if not config_file.exists():
        return {}
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def build_path_settings(config: Dict[str, Any]) -> Dict[str, str]:
    cfg_paths = (config.get("project", {}) or {}).get("paths", {}) or {}
    outputs_root = cfg_paths.get("outputs_root", DEFAULT_PATHS["outputs_root"])
    return {
        "data_root": cfg_paths.get("data_root", DEFAULT_PATHS["data_root"]),
        "templates_root": cfg_paths.get("templates_root", DEFAULT_PATHS["templates_root"]),
        "prompts_root": cfg_paths.get("prompts_root", DEFAULT_PATHS["prompts_root"]),
        "outputs_root": outputs_root,
        "reports_root": cfg_paths.get("reports_root", str(Path(outputs_root) / "reports")),
        "schema_file": cfg_paths.get("schema_file", DEFAULT_PATHS["schema_file"]),
        "generations_root": cfg_paths.get("generations_root", str(Path(outputs_root) / "generations")),
        "eval_results_root": cfg_paths.get("eval_results_root", str(Path(outputs_root) / "eval_res")),
    }


def resolve_path(project_root: Path, path_value: str) -> Path:
    path_obj = Path(path_value)
    return path_obj if path_obj.is_absolute() else project_root / path_obj


def resolve_paths(project_root: Path, path_settings: Dict[str, str]) -> Dict[str, Path]:
    return {key: resolve_path(project_root, value) for key, value in path_settings.items()}


def resolve_data_path(ref_path: str, data_root: Path) -> Path:
    path_obj = Path(ref_path)
    if path_obj.is_absolute():
        return path_obj
    if path_obj.parts:
        if path_obj.parts[0] in {"data", data_root.name}:
            return data_root / Path(*path_obj.parts[1:])
    return data_root / path_obj


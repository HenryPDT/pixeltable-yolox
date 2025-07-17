import importlib
import os
from typing import Optional

from yolox.config import YoloxConfig


def resolve_config(config_str: str) -> YoloxConfig:
    config = YoloxConfig.get_named_config(config_str)
    if config is not None:
        return config

    config_class: Optional[type[YoloxConfig]] = None
    classpath = config_str.split(":")
    if len(classpath) == 2:
        try:
            module = importlib.import_module(classpath[0])
            config_class = getattr(module, classpath[1], None)
        except ImportError:
            pass
    if config_class is None:
        raise ValueError(f"Unknown config class: {config_str}")
    if not issubclass(config_class, YoloxConfig):
        raise ValueError(f"Invalid config class (does not extend `YoloxConfig`): {config_str}")

    try:
        return config_class()
    except Exception as e:
        raise ValueError(f"Error loading model config: {config_str}") from e


def parse_model_config_opts(kv_opts: Optional[list[str]]) -> dict[str, str]:
    """
    Parse key-value options from a list of strings.
    """
    if kv_opts is None:
        kv_opts = []

    kv_dict = {}
    for kv in kv_opts:
        if "=" not in kv:
            raise ValueError(f"Invalid model configuration option (must be of the form OPT=VALUE): {kv}")
        key, value = kv.split("=", 1)
        kv_dict[key] = value
    return kv_dict


def get_unique_output_name(base_dir, name):
    """
    Generate a unique output directory name to avoid overwriting existing experiments.
    Args:
        base_dir (str): Base output directory (e.g., "out")
        name (str): Desired experiment name
    Returns:
        tuple: (full_output_dir, experiment_name)
            - full_output_dir: Complete path like "out/custom_train"
            - experiment_name: Final experiment name like "custom_train" or "yolox_s_2"
    """
    full_path = os.path.join(base_dir, name)
    if not os.path.exists(full_path):
        return base_dir, name
    counter = 2
    while True:
        new_name = f"{name}_{counter}"
        new_path = os.path.join(base_dir, new_name)
        if not os.path.exists(new_path):
            return base_dir, new_name
        counter += 1

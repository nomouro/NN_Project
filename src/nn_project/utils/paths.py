from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("PROJECT_ROOT",PROJECT_ROOT)

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

print("CONFIG_PATH",CONFIG_PATH)
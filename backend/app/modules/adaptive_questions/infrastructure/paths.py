import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_ROOT / "config"
CATALOG_PATH = CONFIG_DIR / "activity_catalog.csv"
MIGRATION_PATH = PACKAGE_ROOT / "migrations" / "001_initial.sql"
MODEL_DIR = PACKAGE_ROOT / "models" / "transferable"
DEFAULT_DB_PATH = Path(
    os.environ.get("C2_DATABASE_PATH", Path.cwd() / ".c2_runtime" / "component2.sqlite3")
)

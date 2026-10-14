from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
CATALOG_PATH = CONFIG_DIR / "activity_catalog.csv"
MIGRATION_PATH = PROJECT_ROOT / "migrations" / "001_initial.sql"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "component2.sqlite3"

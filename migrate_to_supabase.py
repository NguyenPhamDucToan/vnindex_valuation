"""One-time migration: copy all data from local SQLite to Supabase Postgres.

Usage:
    Set DATABASE_URL env var (or .env entry) to the Supabase connection string,
    then run:
        python migrate_to_supabase.py

Creates all tables on the target (via Base.metadata.create_all), then copies
each table's rows from data/vnindex.db in batches.
"""
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, Boolean

from config import DB_URL, DB_PATH
from models.schema import Base

BATCH_SIZE = 5000


def main():
    if DB_URL.startswith("sqlite"):
        print("DATABASE_URL is not set (or points to SQLite) — nothing to migrate to.")
        print("Set DATABASE_URL to your Supabase Postgres connection string and re-run.")
        sys.exit(1)

    if not os.path.exists(DB_PATH):
        print(f"Source SQLite DB not found at {DB_PATH}")
        sys.exit(1)

    src_engine = create_engine(f"sqlite:///{DB_PATH}")
    dst_engine = create_engine(DB_URL, pool_pre_ping=True)

    print(f"Source: sqlite:///{DB_PATH}")
    print(f"Target: {DB_URL.split('@')[-1]}")  # hide credentials in output

    print("Creating tables on target (if not exist)...")
    Base.metadata.create_all(bind=dst_engine)

    for table in Base.metadata.sorted_tables:
        name = table.name
        total = pd.read_sql(f"SELECT COUNT(*) AS n FROM {name}", src_engine)["n"].iloc[0]
        print(f"\n{name}: {total} rows")
        if total == 0:
            continue

        bool_cols = [c.name for c in table.columns if isinstance(c.type, Boolean)]

        with dst_engine.begin() as conn:
            conn.exec_driver_sql(f"TRUNCATE TABLE {name} RESTART IDENTITY CASCADE")

        copied = 0
        for chunk in pd.read_sql(f"SELECT * FROM {name}", src_engine, chunksize=BATCH_SIZE):
            for col in bool_cols:
                chunk[col] = chunk[col].astype(bool)
            chunk.to_sql(name, dst_engine, if_exists="append", index=False, method="multi")
            copied += len(chunk)
            print(f"  copied {copied}/{total}", end="\r")
        print(f"  copied {copied}/{total} - done")

    print("\nMigration complete.")


if __name__ == "__main__":
    main()

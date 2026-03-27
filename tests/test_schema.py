"""Tests for data.schema — connection pooling and DB init."""

import pytest

from data.schema import close_connection, get_connection, init_db


class TestConnectionPool:
    def test_returns_same_connection(self, tmp_path):
        db = tmp_path / "test.duckdb"
        conn1 = get_connection(db)
        conn2 = get_connection(db)
        assert conn1 is conn2

    def test_different_paths_different_connections(self, tmp_path):
        db1 = tmp_path / "a.duckdb"
        db2 = tmp_path / "b.duckdb"
        conn1 = get_connection(db1)
        conn2 = get_connection(db2)
        assert conn1 is not conn2
        close_connection(db1)
        close_connection(db2)

    def test_close_and_reopen(self, tmp_path):
        db = tmp_path / "test.duckdb"
        conn1 = get_connection(db)
        close_connection(db)
        conn2 = get_connection(db)
        assert conn1 is not conn2
        close_connection(db)

    def test_init_db_creates_tables(self, tmp_path):
        db = tmp_path / "test.duckdb"
        conn = init_db(db)
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = {t[0] for t in tables}
        assert "candles" in table_names
        assert "funding_rates" in table_names
        assert "universe" in table_names
        assert "orders" in table_names
        assert "positions" in table_names
        close_connection(db)

    def test_init_db_idempotent(self, tmp_path):
        db = tmp_path / "test.duckdb"
        init_db(db)
        init_db(db)
        conn = get_connection(db)
        tables = conn.execute("SHOW TABLES").fetchall()
        assert len(tables) == 12
        close_connection(db)

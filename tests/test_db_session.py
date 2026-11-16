from app.db.session import load_topic_state, record_duplicates, save_topic_state


class FakeCursor:
    def __init__(self, rows=None, fail=False):
        self.rows = rows or []
        self.fail = fail
        self.statements: list[tuple[str, tuple | None]] = []

    def execute(self, statement, params=None):
        if self.fail:
            raise RuntimeError("boom")
        self.statements.append((statement, params))

    def fetchall(self):
        return self.rows


class FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0
        self.closed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def close(self):
        self.closed = True


def test_load_topic_state_selects_and_maps_rows():
    cursor = FakeCursor(rows=[("reranking", 80, "cursor-1"), ("retrieval", 50, None)])
    state = load_topic_state(FakeConn(cursor))
    assert state == {"reranking": {"min_citations": 80, "cursor": "cursor-1"}, "retrieval": {"min_citations": 50, "cursor": None}}
    assert "FROM topic_state" in cursor.statements[0][0]
    assert "min_citations" in cursor.statements[0][0]


def test_save_topic_state_upserts_with_params():
    cursor = FakeCursor()
    conn = FakeConn(cursor)
    save_topic_state(conn, "reranking", 80, "cursor-1")
    assert len(cursor.statements) == 1
    statement, params = cursor.statements[0]
    assert "INSERT INTO topic_state" in statement
    assert "ON CONFLICT (topic)" in statement
    assert params == ("reranking", 80, "cursor-1")
    assert conn.commits == 1


def test_record_duplicates_inserts_each_pair():
    cursor = FakeCursor()
    conn = FakeConn(cursor)
    pairs = [
        {"kept_id": "a", "duplicate_id": "b", "score": 0.9, "reason": "title_similarity"},
        {"kept_id": "a", "duplicate_id": "c", "score": 0.75, "reason": "possibleDuplicateOf"},
    ]
    record_duplicates(conn, pairs)
    assert len(cursor.statements) == 2
    assert all("INSERT INTO paper_duplicates" in statement for statement, _ in cursor.statements)
    assert cursor.statements[1][1] == ("a", "c", 0.75, "possibleDuplicateOf")
    assert conn.commits == 1


def test_record_duplicates_empty_is_noop():
    cursor = FakeCursor()
    conn = FakeConn(cursor)
    record_duplicates(conn, [])
    assert cursor.statements == []
    assert conn.commits == 0


def test_record_duplicates_never_raises():
    conn = FakeConn(FakeCursor(fail=True))
    record_duplicates(conn, [{"kept_id": "a", "duplicate_id": "b", "score": 0.9, "reason": "title_similarity"}])

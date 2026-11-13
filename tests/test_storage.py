import time

from chatclient.storage import MIGRATIONS, Store


def test_migrations_are_idempotent(tmp_path):
    path = tmp_path / "x.db"
    Store(path).close()
    s = Store(path)
    assert s.conn.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)
    s.close()


def test_upgrade_from_v1_keeps_data(tmp_path, monkeypatch):
    import chatclient.storage as storage

    path = tmp_path / "old.db"
    monkeypatch.setattr(storage, "MIGRATIONS", MIGRATIONS[:1])
    old = Store(path)
    # write data as a v1 install would have (raw SQL: today's readers expect v2 columns)
    old.conn.execute(
        "INSERT INTO sessions(id, title, model, created_at, updated_at) "
        "VALUES (1, 'eski sohbet', 'echo:echo', 't', 't')"
    )
    old.conn.execute(
        "INSERT INTO messages(id, session_id, role, content, created_at, local_day) "
        "VALUES (1, 1, 'user', 'eski mesaj', 't', '2026-11-12')"
    )
    old.conn.execute("UPDATE sessions SET head_message_id = 1 WHERE id = 1")
    old.close()
    sess = type("S", (), {"id": 1})
    monkeypatch.setattr(storage, "MIGRATIONS", MIGRATIONS)
    new = Store(path)
    s = new.get_session(sess.id)
    assert s.title == "eski sohbet" and s.character_id is None
    assert [m.content for m in new.branch(s.head_message_id)] == ["eski mesaj"]
    new.close()


def test_branch_follows_head(store):
    s = store.create_session("echo:echo")
    a = store.add_message(s.id, None, "user", "first")
    b = store.add_message(s.id, a.id, "assistant", "reply one")
    # a sibling reply (regenerate) becomes the new head
    c = store.add_message(s.id, a.id, "assistant", "reply two")
    s = store.get_session(s.id)
    assert s.head_message_id == c.id
    assert [m.content for m in store.branch(s.head_message_id)] == ["first", "reply two"]
    assert [m.content for m in store.branch(b.id)] == ["first", "reply one"]
    assert [m.content for m in store.branch(c.id, limit=1)] == ["reply two"]


def test_search_is_suffix_and_diacritic_tolerant(store):
    s = store.create_session("echo:echo")
    store.add_message(s.id, None, "user", "Dün kütüphanedeki mühürlü haritayı bulduk.")
    store.add_message(s.id, None, "user", "İstanbul'da yağmur yağıyordu.")
    assert len(store.search("kutuphane")) == 1
    assert len(store.search("Kütüphane")) == 1
    assert len(store.search("istanbul")) == 1
    assert len(store.search("ISTANBUL")) == 1
    assert store.search("xyzzy") == []
    assert store.search("ab") == []  # too short for trigram search
    hit = store.search("harita")[0]
    assert "haritayı" in hit.snippet


def test_search_scoped_to_session(store):
    s1 = store.create_session("echo:echo")
    s2 = store.create_session("echo:echo")
    store.add_message(s1.id, None, "user", "deniz feneri")
    store.add_message(s2.id, None, "user", "deniz feneri")
    assert len(store.search("feneri")) == 2
    assert len(store.search("feneri", session_id=s1.id)) == 1


def test_delete_session_removes_messages_and_index(store):
    s = store.create_session("echo:echo")
    m = store.add_message(s.id, None, "user", "silinecek mesaj")
    store.delete_session(s.id)
    assert store.get_session(s.id) is None
    assert store.get_message(m.id) is None
    assert store.search("silinecek") == []


def test_session_params_roundtrip(store):
    s = store.create_session(None)
    store.update_session(s.id, params={"temperature": 0.5}, title="başlık")
    s = store.get_session(s.id)
    assert s.params == {"temperature": 0.5}
    assert s.title == "başlık"


def test_latest_session_and_kv(store):
    assert store.latest_session() is None
    a = store.create_session(None)
    b = store.create_session(None)
    time.sleep(0.01)
    store.add_message(a.id, None, "user", "touch a")  # bumps updated_at
    assert store.latest_session().id == a.id
    store.kv_set("k", "v1")
    store.kv_set("k", "v2")
    assert store.kv_get("k") == "v2"
    assert b.id != a.id

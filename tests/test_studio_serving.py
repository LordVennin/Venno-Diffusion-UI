"""The Studio's page links its files under a version, so an update never runs old cached scripts with a new
page (that left the model list empty); the files are served with `Cache-Control: no-cache`."""

import re

from fastapi import FastAPI
from fastapi.testclient import TestClient

from venno.server import app as server


def client():
    app = FastAPI()
    app.mount("/studio/assets", server.RevalidatedStaticFiles(directory=server.STUDIO_DIR))
    return TestClient(app)


def test_versioned_and_plain_paths():
    c = client()
    stamp = server._studio_stamp()
    for url in (f"/studio/assets/v/{stamp}/js/main.js", "/studio/assets/js/main.js"):
        r = c.get(url)
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache" and "import" in r.text
        assert c.get(url, headers={"If-None-Match": r.headers["etag"]}).status_code == 304
    assert c.get(f"/studio/assets/v/{stamp}/../../../venno/server/app.py").status_code == 404


def test_stamp_follows_the_files(tmp_path, monkeypatch):
    (tmp_path / "a.js").write_text("1")
    monkeypatch.setattr(server, "STUDIO_DIR", str(tmp_path))
    before = server._studio_stamp()
    (tmp_path / "a.js").write_text("22")
    assert re.fullmatch(r"[0-9a-f]{12}", before) and server._studio_stamp() != before

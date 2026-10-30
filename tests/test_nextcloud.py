"""Tests de l'outil nextcloud (serveur WebDAV / CalDAV / CardDAV simulé, sans réseau)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import httpx
import pytest

from seldon.core.types import ActionContext
from seldon.tools.nextcloud import Nextcloud, complete_todo, ical_escape, norm_path, parse_ical

HOST = "https://nc.test"
DAV = "/remote.php/dav"
FILES = f"{DAV}/files/bob/"


def multistatus(*responses: str) -> str:
    return f'<?xml version="1.0"?><d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav" ' \
           f'xmlns:card="urn:ietf:params:xml:ns:carddav">{"".join(responses)}</d:multistatus>'


def response(href: str, props: str) -> str:
    return f"<d:response><d:href>{href}</d:href><d:propstat><d:prop>{props}</d:prop>" \
           "<d:status>HTTP/1.1 200 OK</d:status></d:propstat></d:response>"


def file_entry(href: str, size: int | None = None, etag: str = '"e1"') -> str:
    if size is None:
        return response(href, "<d:resourcetype><d:collection/></d:resourcetype>")
    return response(href, f"<d:resourcetype/><d:getcontentlength>{size}</d:getcontentlength><d:getetag>{etag}</d:getetag>")


def ics(kind: str, *props: str) -> str:
    return "\r\n".join(["BEGIN:VCALENDAR", "VERSION:2.0", f"BEGIN:{kind}", *props, f"END:{kind}", "END:VCALENDAR"])


class FakeDAV:
    """Nextcloud simulé : fichiers en mémoire, un agenda, un carnet d'adresses."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {"Documents/liste.md": "lait\noeufs", "Notes/Courses.md": "pain",
                                      "Notes/Maison/Chauffage.md": "purger les radiateurs"}
        self.events: list[str] = []
        self.todos: dict[str, str] = {}
        self.cards: list[str] = []
        self.requests: list[httpx.Request] = []

    def dirs(self) -> set[str]:
        out = {""}
        for f in self.files:
            parts = f.split("/")
            out |= {"/".join(parts[:i]) for i in range(1, len(parts))}
        return out

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = httpx.URL(str(request.url)).path
        m = request.method
        if path.startswith(FILES):
            rel = path[len(FILES):].strip("/")
            if m == "PROPFIND":
                if rel in self.files:
                    return httpx.Response(207, text=multistatus(file_entry(FILES + rel, len(self.files[rel]))))
                if rel not in self.dirs():
                    return httpx.Response(404)
                body = [file_entry(FILES + rel + "/")]
                prefix = rel + "/" if rel else ""
                for d in self.dirs():
                    if d.startswith(prefix) and d != rel and "/" not in d[len(prefix):]:
                        body.append(file_entry(FILES + d + "/"))
                for f, text in self.files.items():
                    if f.startswith(prefix) and "/" not in f[len(prefix):]:
                        body.append(file_entry(FILES + f, len(text)))
                return httpx.Response(207, text=multistatus(*body))
            if m == "GET":
                return httpx.Response(200, text=self.files[rel], headers={"etag": '"e1"'}) if rel in self.files \
                    else httpx.Response(404)
            if m == "PUT":
                if request.headers.get("if-none-match") == "*" and rel in self.files:
                    return httpx.Response(412)
                self.files[rel] = request.content.decode()
                return httpx.Response(201)
            if m == "DELETE":
                self.files.pop(rel, None)
                return httpx.Response(204)
            if m == "MOVE":
                dest = httpx.URL(request.headers["destination"]).path[len(FILES):]
                self.files[dest] = self.files.pop(rel)
                return httpx.Response(201)
        if path == f"{DAV}/calendars/bob/" and m == "PROPFIND":
            cal = ("<d:resourcetype><d:collection/><c:calendar/></d:resourcetype><d:displayname>Perso</d:displayname>"
                   '<c:supported-calendar-component-set><c:comp name="VEVENT"/><c:comp name="VTODO"/>'
                   "</c:supported-calendar-component-set>")
            return httpx.Response(207, text=multistatus(
                response(f"{DAV}/calendars/bob/", "<d:resourcetype><d:collection/></d:resourcetype>"),
                response(f"{DAV}/calendars/bob/perso/", cal),
                response(f"{DAV}/calendars/bob/inbox/", "<d:resourcetype><d:collection/></d:resourcetype>")))
        if path == f"{DAV}/calendars/bob/perso/" and m == "REPORT":
            if b'name="VTODO"' in request.content:
                items = [response(href, f"<d:getetag>\"t\"</d:getetag><c:calendar-data>{data}</c:calendar-data>")
                         for href, data in self.todos.items()]
            else:
                items = [response(f"{DAV}/calendars/bob/perso/e{i}.ics",
                                  f"<c:calendar-data>{data}</c:calendar-data>") for i, data in enumerate(self.events)]
            return httpx.Response(207, text=multistatus(*items))
        if path.startswith(f"{DAV}/calendars/bob/perso/") and m == "PUT":
            self.todos[path] = request.content.decode()
            return httpx.Response(201)
        if path == f"{DAV}/addressbooks/users/bob/" and m == "PROPFIND":
            return httpx.Response(207, text=multistatus(response(
                f"{DAV}/addressbooks/users/bob/contacts/",
                "<d:resourcetype><d:collection/><card:addressbook/></d:resourcetype>")))
        if path == f"{DAV}/addressbooks/users/bob/contacts/" and m == "REPORT":
            return httpx.Response(207, text=multistatus(*(
                response(f"{DAV}/addressbooks/users/bob/contacts/c{i}.vcf", f"<card:address-data>{c}</card:address-data>")
                for i, c in enumerate(self.cards))))
        return httpx.Response(405)


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("NEXTCLOUD_HOST", HOST)
    monkeypatch.setenv("NEXTCLOUD_USERNAME", "bob")
    monkeypatch.setenv("NEXTCLOUD_PASSWORD", "app-pass")
    return FakeDAV()


def make(server: FakeDAV, **options) -> Nextcloud:
    nc = Nextcloud(**options)
    nc.client = httpx.AsyncClient(transport=httpx.MockTransport(server), auth=("bob", "app-pass"))
    return nc


def ctx(answer: bool = True, asked: list | None = None) -> ActionContext:
    async def confirm(question: str) -> bool:
        if asked is not None:
            asked.append(question)
        return answer

    return ActionContext(confirm=confirm)


async def test_missing_credentials(monkeypatch):
    for var in ("NEXTCLOUD_HOST", "NEXTCLOUD_USERNAME", "NEXTCLOUD_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    result = await Nextcloud().run({"command": "files_list"}, ctx())
    assert result.is_error and "NEXTCLOUD_PASSWORD" in result.text


def test_apps_limit_commands(server):
    nc = Nextcloud(apps=["calendar"])
    assert nc.input_schema["properties"]["command"]["enum"] == ["events", "event_create"]
    assert "files_list" not in nc.description
    with pytest.raises(ValueError):
        Nextcloud(apps=["deck"])


@pytest.mark.parametrize("path", ["../secret", "a/../../b", "x\x00y", "a\nb"])
def test_paths_cannot_escape(path):
    with pytest.raises(ValueError):
        norm_path(path)


def test_norm_path():
    assert norm_path("/Documents//./liste.md") == "Documents/liste.md" and norm_path(None) == ""


async def test_traversal_sends_nothing(server):
    nc = make(server)
    result = await nc.run({"command": "file_read", "path": "../../calendars/bob"}, ctx())
    assert result.is_error and not server.requests


async def test_list_and_read(server):
    nc = make(server)
    listing = (await nc.run({"command": "files_list", "path": "Documents"}, ctx())).text
    assert "liste.md" in listing
    root = (await nc.run({"command": "files_list"}, ctx())).text
    assert "- Documents/" in root and "- Notes/" in root
    text = (await nc.run({"command": "file_read", "path": "Documents/liste.md"}, ctx())).text
    assert "lait" in text and "oeufs" in text


async def test_write_new_append_and_overwrite(server):
    nc = make(server)
    asked: list[str] = []
    r = await nc.run({"command": "file_write", "path": "Documents/neuf.txt", "content": "salut"}, ctx(asked=asked))
    assert not r.is_error and server.files["Documents/neuf.txt"] == "salut" and not asked
    await nc.run({"command": "file_write", "path": "Documents/liste.md", "content": "beurre", "append": True}, ctx())
    assert server.files["Documents/liste.md"] == "lait\noeufs\nbeurre"
    r = await nc.run({"command": "file_write", "path": "Documents/liste.md", "content": "rien"}, ctx(False))
    assert r.is_error and server.files["Documents/liste.md"].startswith("lait")   # écrasement refusé


async def test_delete_and_move_need_confirmation(server):
    nc = make(server)
    r = await nc.run({"command": "file_delete", "path": "Documents/liste.md"}, ctx(False))
    assert r.is_error and "Documents/liste.md" in server.files
    assert not any(q.method == "DELETE" for q in server.requests)
    await nc.run({"command": "file_move", "path": "Documents/liste.md", "destination": "Archives/liste.md"}, ctx())
    assert "Archives/liste.md" in server.files
    await nc.run({"command": "file_delete", "path": "Archives/liste.md"}, ctx())
    assert "Archives/liste.md" not in server.files


async def test_confirm_writes_asks_before_creating(server):
    nc = make(server, confirm_writes=True)
    r = await nc.run({"command": "note_write", "title": "Idées", "content": "x"}, ctx(False))
    assert r.is_error and "Notes/Idées.md" not in server.files


async def test_notes(server):
    nc = make(server)
    listing = (await nc.run({"command": "notes_list"}, ctx())).text
    assert "Courses" in listing and "Chauffage (Maison)" in listing
    assert "pain" in (await nc.run({"command": "note_read", "title": "courses"}, ctx())).text
    found = (await nc.run({"command": "notes_search", "query": "radiateurs"}, ctx())).text
    assert "Chauffage" in found and "purger" in found
    await nc.run({"command": "note_write", "title": "Courses", "content": "lait", "append": True}, ctx())
    assert server.files["Notes/Courses.md"] == "pain\nlait"
    await nc.run({"command": "note_write", "title": "Idées/../x", "content": "a"}, ctx())
    assert "Notes/Idées-..-x.md" in server.files   # le titre ne crée pas de chemin


async def test_events_sorted_and_local(server):
    today = date.today().strftime("%Y%m%d")
    server.events = [
        ics("VEVENT", "SUMMARY:Dentiste", f"DTSTART:{today}T150000Z", f"DTEND:{today}T160000Z", "LOCATION:Lausanne"),
        ics("VEVENT", "SUMMARY:Anniversaire", f"DTSTART;VALUE=DATE:{today}"),
        ics("VEVENT", "SUMMARY:Annulé", f"DTSTART:{today}T090000Z", "STATUS:CANCELLED"),
    ]
    text = (await make(server).run({"command": "events"}, ctx())).text
    local = datetime.strptime(f"{today}T150000", "%Y%m%dT%H%M%S").replace(tzinfo=__import__("datetime").timezone.utc).astimezone()
    assert "Anniversaire" in text and "(journée)" in text and "Annulé" not in text
    assert f"{local:%H:%M}" in text and "à Lausanne" in text
    assert text.index("Anniversaire") < text.index("Dentiste")
    report = next(q for q in server.requests if q.method == "REPORT")
    assert b"<c:expand" in report.content and b"time-range" in report.content


async def test_event_create_cannot_inject_properties(server):
    nc = make(server)
    title = "Réunion\r\nATTENDEE:mailto:pirate@example.com\nEND:VEVENT"
    r = await nc.run({"command": "event_create", "title": title, "start": "2030-05-02T14:00",
                      "location": "Salle; 2, étage"}, ctx())
    assert not r.is_error
    body = next(iter(server.todos.values()))
    lines = body.replace("\r\n ", "").split("\r\n")
    assert not any(line.startswith("ATTENDEE") for line in lines)
    assert sum(line == "END:VEVENT" for line in lines) == 1
    assert "LOCATION:Salle\\; 2\\, étage" in lines
    event = parse_ical(body)[0]
    assert event["SUMMARY"][0][1].startswith("Réunion\\n")


async def test_event_create_rejects_end_before_start(server):
    r = await make(server).run({"command": "event_create", "title": "x", "start": "2030-05-02T14:00",
                                "end": "2030-05-02T13:00"}, ctx())
    assert r.is_error


async def test_tasks_and_complete(server):
    overdue = (date.today() - timedelta(days=2)).strftime("%Y%m%d")
    later = (date.today() + timedelta(days=30)).strftime("%Y%m%d")
    href = f"{DAV}/calendars/bob/perso/t1.ics"
    server.todos = {
        href: ics("VTODO", "UID:1", "SUMMARY:Payer la facture", f"DUE;VALUE=DATE:{overdue}", "STATUS:NEEDS-ACTION"),
        f"{DAV}/calendars/bob/perso/t2.ics": ics("VTODO", "UID:2", "SUMMARY:Vacances", f"DUE;VALUE=DATE:{later}"),
        f"{DAV}/calendars/bob/perso/t3.ics": ics("VTODO", "UID:3", "SUMMARY:Fait", "STATUS:COMPLETED"),
    }
    nc = make(server)
    soon = (await nc.run({"command": "tasks", "days": 1}, ctx())).text
    assert "Payer la facture" in soon and "en retard" in soon and "Vacances" not in soon and "Fait" not in soon
    r = await nc.run({"command": "task_complete", "title": "facture"}, ctx())
    assert not r.is_error
    put = next(q for q in server.requests if q.method == "PUT")
    assert put.headers["if-match"] == '"t"' and "STATUS:COMPLETED" in server.todos[href]
    assert "NEEDS-ACTION" not in server.todos[href]


def test_complete_todo_keeps_alarm():
    data = ics("VTODO", "SUMMARY:x", "STATUS:IN-PROCESS", "PERCENT-COMPLETE:40",
               "BEGIN:VALARM", "TRIGGER:-PT5M", "END:VALARM")
    done = complete_todo(data)
    assert "PERCENT-COMPLETE:100" in done and "PERCENT-COMPLETE:40" not in done and "TRIGGER:-PT5M" in done


async def test_contacts(server):
    server.cards = ["BEGIN:VCARD\r\nVERSION:3.0\r\nFN:Alice Martin\r\nTEL;TYPE=cell:+41 79 000 00 00\r\n"
                    "EMAIL:alice@example.com\r\nADR:;;Rue du Lac 1;Vevey;;1800;Suisse\r\nEND:VCARD"]
    text = (await make(server).run({"command": "contacts_search", "query": "alice"}, ctx())).text
    assert "Alice Martin" in text and "+41 79 000 00 00" in text and "Vevey" in text
    report = next(q for q in server.requests if q.method == "REPORT")
    assert b"alice" in report.content


def test_ical_escape():
    assert ical_escape("a,b;c\\d\ne") == "a\\,b\\;c\\\\d\\ne"

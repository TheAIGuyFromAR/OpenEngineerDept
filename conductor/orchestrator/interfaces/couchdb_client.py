"""
CouchDB Sync Adapter — Direct CouchDB access for Obsidian LiveSync vaults.

Obsidian LiveSync stores vault files as CouchDB documents with chunked content:

Document types (relevant to us):
  "plain"   — text files (.md, .txt, etc). Content in children[] chunks.
  "newnote" — binary files. Content in children[] chunks, base64-encoded.
  "leaf"    — a content chunk. Has _id "h:<hash>" and data field.

Document schema (PlainEntry):
  _id:      file path (e.g. "conductor/inbox/task.md")
  _rev:     CouchDB revision
  type:     "plain"
  path:     same as _id for non-obfuscated vaults
  ctime:    creation time (ms epoch)
  mtime:    modification time (ms epoch)
  size:     file size in bytes
  children: ["h:abc123", "h:def456"] — chunk document IDs
  eden:     {chunk_id: {data: "...", epoch: ...}} — inline temp chunks
  deleted:  bool

Chunk schema (EntryLeaf):
  _id:   "h:<hash>"
  type:  "leaf"
  data:  chunk content string

Content reassembly:
  1. Read document → get children[] list
  2. For each child: check eden first, then fetch from DB
  3. Concatenate chunk data strings → full file content

_changes feed:
  GET /{db}/_changes?feed=continuous&include_docs=true&since=<seq>
  Filter: type != "leaf" (skip chunks, only watch document-level changes)

Conductor paths:
  conductor/inbox/*.md    — tasks to process
  conductor/completed/*.md — results
  conductor/failed/*.md    — errors
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# LiveSync constants
PREFIX_CHUNK = "h:"
PLAIN_TEXT_EXTENSIONS = {".md", ".txt", ".svg", ".html", ".csv", ".css", ".js", ".xml", ".canvas"}


@dataclass
class LiveSyncDoc:
    """A reconstructed LiveSync document with its content."""
    doc_id: str
    path: str
    content: str
    doc_type: str
    ctime: int
    mtime: int
    rev: str = ""


class CouchDBClient:
    """
    Low-level CouchDB client for Obsidian LiveSync databases.

    Handles the LiveSync document format: reads chunks, reassembles content,
    writes new documents with proper chunking, and watches the _changes feed.
    """

    def __init__(
        self,
        url: str,
        database: str,
        username: str = "",
        password: str = "",
    ) -> None:
        auth = None
        if username and password:
            auth = httpx.BasicAuth(username, password)
        self._client = httpx.AsyncClient(
            base_url=f"{url.rstrip('/')}/{database}",
            auth=auth,
            timeout=30,
        )
        self._db_url = f"{url.rstrip('/')}/{database}"
        self._since: str = "now"  # Start from current seq on first connect

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_doc(self, doc_id: str) -> dict | None:
        """Fetch a raw CouchDB document."""
        try:
            resp = await self._client.get(f"/{doc_id}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError:
            return None

    async def get_chunk(self, chunk_id: str) -> str:
        """Fetch a single chunk's data."""
        doc = await self.get_doc(chunk_id)
        if doc is None:
            logger.warning("Chunk not found: %s", chunk_id)
            return ""
        return doc.get("data", "")

    async def read_file(self, path: str) -> LiveSyncDoc | None:
        """
        Read a file from the vault by path.
        Fetches the document and reassembles content from chunks.
        """
        doc = await self.get_doc(path)
        if doc is None:
            return None

        doc_type = doc.get("type", "")
        if doc_type not in ("plain", "newnote"):
            return None

        if doc.get("deleted", False) or doc.get("_deleted", False):
            return None

        # Reassemble content from chunks
        children = doc.get("children", [])
        eden = doc.get("eden", {})
        content = await self._reassemble_chunks(children, eden)

        return LiveSyncDoc(
            doc_id=doc["_id"],
            path=doc.get("path", doc["_id"]),
            content=content,
            doc_type=doc_type,
            ctime=doc.get("ctime", 0),
            mtime=doc.get("mtime", 0),
            rev=doc.get("_rev", ""),
        )

    async def _reassemble_chunks(
        self, children: list[str], eden: dict
    ) -> str:
        """Reassemble file content from chunk references."""
        parts: list[str] = []
        for chunk_id in children:
            # Check eden (inline temp chunks) first
            if chunk_id in eden:
                eden_chunk = eden[chunk_id]
                if isinstance(eden_chunk, dict):
                    parts.append(eden_chunk.get("data", ""))
                else:
                    parts.append(str(eden_chunk))
            else:
                # Fetch from DB
                data = await self.get_chunk(chunk_id)
                parts.append(data)
        return "".join(parts)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def write_file(self, path: str, content: str) -> str:
        """
        Write a file to the vault.
        Creates chunks and a document entry in LiveSync format.
        Returns the new document revision.
        """
        now_ms = int(time.time() * 1000)

        # Split content into chunks
        chunks = self._split_content(content)
        chunk_ids: list[str] = []

        # Write each chunk
        for chunk_data in chunks:
            chunk_id = self._compute_chunk_id(chunk_data)
            chunk_ids.append(chunk_id)

            # Check if chunk already exists (content-addressed dedup)
            existing = await self.get_doc(chunk_id)
            if existing is None:
                await self._client.put(
                    f"/{chunk_id}",
                    json={
                        "_id": chunk_id,
                        "type": "leaf",
                        "data": chunk_data,
                    },
                )

        # Check if document already exists (need _rev for update)
        existing_doc = await self.get_doc(path)
        doc: dict = {
            "_id": path,
            "type": "plain",
            "path": path,
            "ctime": now_ms,
            "mtime": now_ms,
            "size": len(content.encode("utf-8")),
            "children": chunk_ids,
            "eden": {},
            "deleted": False,
        }

        if existing_doc:
            doc["_rev"] = existing_doc["_rev"]
            doc["ctime"] = existing_doc.get("ctime", now_ms)

        resp = await self._client.put(f"/{path}", json=doc)
        resp.raise_for_status()
        return resp.json().get("rev", "")

    async def delete_file(self, path: str) -> bool:
        """Mark a document as deleted in LiveSync."""
        doc = await self.get_doc(path)
        if doc is None:
            return False

        doc["deleted"] = True
        doc["mtime"] = int(time.time() * 1000)
        resp = await self._client.put(f"/{path}", json=doc)
        return resp.status_code == 201

    # ------------------------------------------------------------------
    # List operations
    # ------------------------------------------------------------------

    async def list_files(self, prefix: str) -> list[str]:
        """
        List all file paths under a prefix.
        Uses CouchDB _all_docs with startkey/endkey.
        """
        resp = await self._client.get(
            "/_all_docs",
            params={
                "startkey": f'"{prefix}"',
                "endkey": f'"{prefix}\\ufff0"',
                "include_docs": "false",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        paths: list[str] = []
        for row in data.get("rows", []):
            doc_id = row["id"]
            # Skip chunks and system docs
            if doc_id.startswith(PREFIX_CHUNK) or doc_id.startswith("_"):
                continue
            paths.append(doc_id)
        return paths

    # ------------------------------------------------------------------
    # Changes feed
    # ------------------------------------------------------------------

    async def poll_changes(
        self,
        prefix: str | None = None,
        timeout_ms: int = 5000,
    ) -> list[dict]:
        """
        Long-poll the _changes feed for new/modified documents.
        Filters out chunk documents (type=leaf).
        Returns list of changed documents.
        """
        params: dict = {
            "feed": "longpoll",
            "include_docs": "true",
            "since": self._since,
            "timeout": str(timeout_ms),
            "filter": "_selector",
        }

        body = {
            "selector": {
                "type": {"$ne": "leaf"},
            }
        }

        try:
            resp = await self._client.post(
                "/_changes",
                params=params,
                json=body,
                timeout=timeout_ms / 1000 + 5,
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            logger.debug("Changes poll timeout/error: %s", exc)
            return []

        self._since = data.get("last_seq", self._since)

        changes: list[dict] = []
        for result in data.get("results", []):
            doc = result.get("doc", {})
            path = doc.get("path", doc.get("_id", ""))

            # Filter by prefix if specified
            if prefix and not path.startswith(prefix):
                continue

            # Skip internal docs
            if ":" in path or path.startswith("_"):
                continue

            changes.append(doc)

        return changes

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def check_connection(self) -> dict:
        """Check CouchDB connectivity and database info."""
        try:
            resp = await self._client.get("/")
            resp.raise_for_status()
            data = resp.json()
            return {
                "status": "ok",
                "db_name": data.get("db_name", ""),
                "doc_count": data.get("doc_count", 0),
                "update_seq": str(data.get("update_seq", ""))[:20],
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Content chunking (Phase 0: simple fixed-size splitting)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_content(content: str, max_chunk_size: int = 1000) -> list[str]:
        """
        Split content into chunks for storage.
        Phase 0: Simple fixed-size splitting. LiveSync uses Rabin-Karp
        content-defined chunking for dedup, but that's overkill for
        our conductor output files.
        """
        if len(content) <= max_chunk_size:
            return [content]
        chunks: list[str] = []
        for i in range(0, len(content), max_chunk_size):
            chunks.append(content[i : i + max_chunk_size])
        return chunks

    @staticmethod
    def _compute_chunk_id(data: str) -> str:
        """Compute a content-addressed chunk ID."""
        h = hashlib.sha256(data.encode()).hexdigest()[:16]
        return f"{PREFIX_CHUNK}{h}"

    async def close(self) -> None:
        await self._client.aclose()

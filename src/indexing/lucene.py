from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
import logging
import math
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Iterable, Sequence

from src.data.phase2_contracts import IndexManifest, StageManifest
from src.data.schemas import SearchHit
from src.indexing.lexical_tokens import tokenize_and_encode

logger = logging.getLogger(__name__)


@dataclass
class LexicalRecord:
    external_id: str
    official_doc_id: int
    language: str
    item_type: str
    encoded_contents: str


class LuceneSparseBackend:
    """Disk-backed sparse retrieval backend implementing the Lucene/BM25 protocol.

    Provides:
    - Partitioned disk builds with atomic commit and StageManifest.
    - Global multi-partition assembly with unified IDF and avgdl across all segments.
    - Crash and replay safety: duplicate document IDs are idempotently resolved.
    - Sub-second disk-backed BM25 scoring with k1=1.2, b=0.75.
    - Source text retrieval decoupled from index (no storeRaw body duplication).
    """

    DEFAULT_K1: float = 1.2
    DEFAULT_B: float = 0.75

    def __init__(
        self,
        index_dir: Path | str | None = None,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        self.index_dir = Path(index_dir) if index_dir is not None else None
        self.k1 = k1
        self.b = b
        self._open_conns: dict[str, sqlite3.Connection] = {}

    def _resolve_index_path(self, language: str, item_type: str, explicit_path: Path | None = None) -> Path | None:
        if explicit_path is not None:
            return explicit_path
        if self.index_dir is None:
            return None

        # Check candidate directory names
        candidates = [
            self.index_dir / f"{language}_{item_type}",
            self.index_dir / f"{language}_{item_type}s",
            self.index_dir / language / item_type,
            self.index_dir / language / f"{item_type}s",
            self.index_dir / item_type / language,
            self.index_dir,
        ]
        for cand in candidates:
            if (cand / "index.db").exists():
                return cand
            if (cand / "manifest.json").exists() and (cand / "index.db").exists():
                return cand

        return None

    @classmethod
    def build_partition(
        cls,
        rows: Iterable[LexicalRecord],
        output: Path | str,
        snapshot_id: str = "phase2_full",
    ) -> StageManifest:
        """Builds an isolated disk-backed partition for a shard/language/item_type segment."""
        out_path = Path(output)
        out_path.mkdir(parents=True, exist_ok=True)
        db_path = out_path / "partition.db"

        # Build in temporary file first for atomic commit
        tmp_dir = Path(tempfile.mkdtemp(dir=out_path.parent))
        tmp_db = tmp_dir / "partition.db"

        conn = sqlite3.connect(str(tmp_db))
        conn.execute("PRAGMA synchronous = OFF;")
        conn.execute("PRAGMA journal_mode = MEMORY;")

        with conn:
            conn.execute(
                """
                CREATE TABLE documents (
                    doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    external_id TEXT UNIQUE NOT NULL,
                    official_doc_id INTEGER NOT NULL,
                    length INTEGER NOT NULL,
                    language TEXT NOT NULL,
                    item_type TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE postings (
                    term TEXT NOT NULL,
                    doc_id INTEGER NOT NULL,
                    tf INTEGER NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE term_stats (
                    term TEXT PRIMARY KEY,
                    df INTEGER NOT NULL
                );
                """
            )

        doc_rows: list[tuple[str, int, int, str, str]] = []
        postings_rows: list[tuple[str, int, int]] = []
        term_df: Counter[str] = Counter()
        ext_ids_hash = hashlib.sha256()

        total_units = 0
        current_internal_id = 0

        for r in rows:
            tokens = [t for t in r.encoded_contents.split() if t]
            doc_len = len(tokens)
            current_internal_id += 1
            doc_rows.append((r.external_id, r.official_doc_id, doc_len, r.language, r.item_type))
            ext_ids_hash.update(r.external_id.encode("utf-8"))

            t_counts = Counter(tokens)
            for t, count in t_counts.items():
                postings_rows.append((t, current_internal_id, count))
                term_df[t] += 1
            total_units += 1

        with conn:
            conn.executemany(
                "INSERT INTO documents (external_id, official_doc_id, length, language, item_type) VALUES (?, ?, ?, ?, ?)",
                doc_rows,
            )
            conn.executemany(
                "INSERT INTO postings (term, doc_id, tf) VALUES (?, ?, ?)",
                postings_rows,
            )
            conn.executemany(
                "INSERT INTO term_stats (term, df) VALUES (?, ?)",
                term_df.items(),
            )
            conn.execute("CREATE INDEX idx_part_postings ON postings(term);")

        conn.close()

        # Atomic replace of partition.db
        if db_path.exists():
            db_path.unlink()
        tmp_db.replace(db_path)
        try:
            tmp_dir.rmdir()
        except Exception:
            pass

        # Compute hash of db
        with open(db_path, "rb") as f:
            db_sha = hashlib.sha256(f.read()).hexdigest()

        stage_manifest = StageManifest(
            stage="sparse_partition",
            fingerprint=ext_ids_hash.hexdigest(),
            snapshot_id=snapshot_id,
            completed_units=total_units,
            expected_units=total_units,
            artifact_paths_and_hashes={"partition.db": db_sha},
        )

        manifest_file = out_path / "stage_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(asdict(stage_manifest), f, indent=2)

        return stage_manifest

    @classmethod
    def assemble(
        cls,
        partitions: Sequence[Path | str],
        output: Path | str,
        snapshot_id: str = "phase2_full",
        language: str = "und",
        item_type: str = "document",
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> IndexManifest:
        """Assembles partition segments into a single logical index with global IDF and avgdl."""
        out_path = Path(output)
        out_path.mkdir(parents=True, exist_ok=True)
        final_db = out_path / "index.db"

        tmp_dir = Path(tempfile.mkdtemp(dir=out_path.parent))
        tmp_db = tmp_dir / "index.db"

        conn = sqlite3.connect(str(tmp_db))
        conn.execute("PRAGMA synchronous = OFF;")
        conn.execute("PRAGMA journal_mode = MEMORY;")

        with conn:
            conn.execute(
                """
                CREATE TABLE documents (
                    doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    external_id TEXT UNIQUE NOT NULL,
                    official_doc_id INTEGER NOT NULL,
                    length INTEGER NOT NULL,
                    language TEXT NOT NULL,
                    item_type TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE postings (
                    term TEXT NOT NULL,
                    doc_id INTEGER NOT NULL,
                    tf INTEGER NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE corpus_stats (
                    total_docs INTEGER NOT NULL,
                    avgdl REAL NOT NULL,
                    k1 REAL NOT NULL,
                    b REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE term_stats (
                    term TEXT PRIMARY KEY,
                    df INTEGER NOT NULL,
                    idf REAL NOT NULL
                );
                """
            )

        seen_external_ids: set[str] = set()
        inferred_lang = language
        inferred_type = item_type

        # Ingestion across all partitions
        for part_p in partitions:
            p_dir = Path(part_p)
            p_db = p_dir / "partition.db" if p_dir.is_dir() else p_dir
            if not p_db.exists():
                logger.warning("Partition database not found at %s", p_db)
                continue

            p_conn = sqlite3.connect(str(p_db))
            p_docs = p_conn.execute(
                "SELECT doc_id, external_id, official_doc_id, length, language, item_type FROM documents"
            ).fetchall()

            doc_id_mapping: dict[int, int] = {}
            new_doc_rows: list[tuple[str, int, int, str, str]] = []

            for old_id, ext_id, off_id, length, lang, itype in p_docs:
                if ext_id in seen_external_ids:
                    # Idempotent replay safety: skip duplicate external_id
                    continue
                seen_external_ids.add(ext_id)
                inferred_lang = lang
                inferred_type = itype

                with conn:
                    cur = conn.execute(
                        "INSERT INTO documents (external_id, official_doc_id, length, language, item_type) VALUES (?, ?, ?, ?, ?)",
                        (ext_id, off_id, length, lang, itype),
                    )
                    doc_id_mapping[old_id] = cur.lastrowid

            # Postings from this partition
            if doc_id_mapping:
                p_postings = p_conn.execute("SELECT term, doc_id, tf FROM postings").fetchall()
                remapped_postings = [
                    (term, doc_id_mapping[old_id], tf)
                    for term, old_id, tf in p_postings
                    if old_id in doc_id_mapping
                ]
                with conn:
                    conn.executemany(
                        "INSERT INTO postings (term, doc_id, tf) VALUES (?, ?, ?)",
                        remapped_postings,
                    )

            p_conn.close()

        # Compute GLOBAL statistics across all assembled partitions
        total_docs = len(seen_external_ids)
        total_length = conn.execute("SELECT COALESCE(SUM(length), 0) FROM documents").fetchone()[0]
        avgdl = (total_length / total_docs) if total_docs > 0 else 0.0

        with conn:
            conn.execute(
                "INSERT INTO corpus_stats (total_docs, avgdl, k1, b) VALUES (?, ?, ?, ?)",
                (total_docs, avgdl, k1, b),
            )

            # Compute global DF and global IDF for all terms
            # Standard Lucene BM25 IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
            term_dfs = conn.execute(
                "SELECT term, COUNT(DISTINCT doc_id) FROM postings GROUP BY term"
            ).fetchall()

            term_stats_rows = []
            for term, df in term_dfs:
                idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                term_stats_rows.append((term, df, max(0.0, idf)))

            conn.executemany(
                "INSERT INTO term_stats (term, df, idf) VALUES (?, ?, ?)",
                term_stats_rows,
            )

            # Build fast indexes for search
            conn.execute("CREATE INDEX idx_postings_term ON postings(term, doc_id);")
            conn.execute("CREATE INDEX idx_docs_ext_id ON documents(external_id);")

        conn.close()

        if final_db.exists():
            final_db.unlink()
        tmp_db.replace(final_db)
        try:
            tmp_dir.rmdir()
        except Exception:
            pass

        # Membership hash: sorted external IDs
        sorted_ids = sorted(seen_external_ids)
        ids_hasher = hashlib.sha256()
        for sid in sorted_ids:
            ids_hasher.update(sid.encode("utf-8"))
        membership_sha256 = ids_hasher.hexdigest()

        with open(final_db, "rb") as f:
            final_db_sha = hashlib.sha256(f.read()).hexdigest()

        manifest = IndexManifest(
            snapshot_id=snapshot_id,
            item_type=inferred_type,
            language=inferred_lang,
            member_count=total_docs,
            membership_sha256=membership_sha256,
            backend="lucene",
            model_fingerprint=None,
            tokenizer_fingerprint="sha256_hex_v1",
            parameters={"k1": k1, "b": b, "analyzer": "whitespace"},
            artifact_paths_and_hashes={"index.db": final_db_sha},
        )

        manifest_file = out_path / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2)

        return manifest

    def search(
        self,
        query: str,
        language: str,
        item_type: str = "document",
        top_k: int = 10,
        index_path: Path | str | None = None,
    ) -> list[SearchHit]:
        """Encodes query terms and scores against assembled logical index using global BM25."""
        if not query or not query.strip():
            return []

        eff_path = self._resolve_index_path(
            language=language,
            item_type=item_type,
            explicit_path=Path(index_path) if index_path is not None else None,
        )
        if eff_path is None:
            logger.debug("No index found for language=%s, item_type=%s", language, item_type)
            return []

        db_file = eff_path / "index.db" if eff_path.is_dir() else eff_path
        if not db_file.exists():
            return []

        db_key = str(db_file.resolve())
        if db_key not in self._open_conns:
            # Read-only SQLite connection for speed and concurrency
            uri = f"file:{db_file.resolve().as_posix()}?mode=ro"
            self._open_conns[db_key] = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn = self._open_conns[db_key]

        # 1. Encode query terms using safe lexical tokenizer
        encoded_query_str = tokenize_and_encode(query, language=language)
        query_terms = [t for t in encoded_query_str.split() if t]
        if not query_terms:
            return []

        q_term_counts = Counter(query_terms)
        unique_terms = list(q_term_counts.keys())

        # 2. Retrieve global corpus statistics
        stats_row = conn.execute("SELECT total_docs, avgdl, k1, b FROM corpus_stats").fetchone()
        if not stats_row or stats_row[0] == 0:
            return []
        total_docs, avgdl, k1, b = stats_row

        # 3. Retrieve IDFs for query terms
        placeholders = ",".join(["?"] * len(unique_terms))
        idf_rows = conn.execute(
            f"SELECT term, idf FROM term_stats WHERE term IN ({placeholders})",
            unique_terms,
        ).fetchall()
        idfs = {term: idf for term, idf in idf_rows if idf > 0.0}
        if not idfs:
            return []

        # 4. Retrieve postings for matching terms
        matching_terms = list(idfs.keys())
        p_placeholders = ",".join(["?"] * len(matching_terms))
        postings = conn.execute(
            f"""
            SELECT p.doc_id, p.term, p.tf, d.external_id, d.official_doc_id, d.length
            FROM postings p
            JOIN documents d ON p.doc_id = d.doc_id
            WHERE p.term IN ({p_placeholders})
            """,
            matching_terms,
        ).fetchall()

        if not postings:
            return []

        # 5. Score candidates using Lucene BM25
        doc_scores: dict[int, float] = Counter()
        doc_meta: dict[int, tuple[str, int]] = {}

        for doc_id, term, tf, ext_id, off_id, doc_len in postings:
            doc_meta[doc_id] = (ext_id, off_id)
            term_idf = idfs[term]
            q_weight = q_term_counts[term]

            # BM25 tf component
            denom = tf + k1 * (1.0 - b + b * (doc_len / avgdl if avgdl > 0 else 1.0))
            score_contrib = term_idf * ((tf * (k1 + 1.0)) / denom) * q_weight
            doc_scores[doc_id] += score_contrib

        # 6. Rank and format SearchHit items
        ranked_docs = sorted(
            [(doc_id, score) for doc_id, score in doc_scores.items() if score > 0.0],
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        hits: list[SearchHit] = []
        for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
            ext_id, off_id = doc_meta[doc_id]
            hits.append(
                SearchHit(
                    item_id=ext_id,
                    item_type=item_type,  # type: ignore
                    doc_id=str(off_id),
                    language=language,
                    source=f"{language}_sparse",
                    raw_score=float(score),
                    rank=rank,
                )
            )

        return hits

    def search_documents(
        self,
        query: str,
        language: str,
        top_k: int = 10,
    ) -> list[SearchHit]:
        """Convenience retriever protocol method for document search."""
        return self.search(query=query, language=language, item_type="document", top_k=top_k)

    def search_chunks(
        self,
        query: str,
        language: str,
        top_k: int = 10,
    ) -> list[SearchHit]:
        """Convenience retriever protocol method for chunk search."""
        return self.search(query=query, language=language, item_type="chunk", top_k=top_k)

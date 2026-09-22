from src.data.schemas import ChunkRecord, DocumentRecord

VALID_LANGUAGES = {"vi", "en", "zh"}


def validate_corpus_integrity(
    documents: list[DocumentRecord],
    chunks: list[ChunkRecord],
    strict: bool = True,
) -> bool:
    doc_ids = {d.doc_id for d in documents}
    if len(doc_ids) != len(documents):
        if strict:
            raise ValueError("Duplicate doc_id found in document list.")
        return False

    chunk_ids = {c.chunk_id for c in chunks}
    if len(chunk_ids) != len(chunks):
        if strict:
            raise ValueError("Duplicate chunk_id found in chunk list.")
        return False

    for d in documents:
        if d.language not in VALID_LANGUAGES:
            if strict:
                raise ValueError(f"Invalid document language: {d.language}")
            return False

    for c in chunks:
        if c.language not in VALID_LANGUAGES:
            if strict:
                raise ValueError(f"Invalid chunk language: {c.language}")
            return False
        if c.doc_id not in doc_ids:
            if strict:
                raise ValueError(
                    f"Chunk {c.chunk_id} references doc_id '{c.doc_id}' which does not exist in document table."
                )
            return False

    return True


# Alias for pipeline entrypoints
validate_corpus = validate_corpus_integrity

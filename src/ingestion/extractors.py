import hashlib
import logging
from typing import Sequence
from lxml import etree, html
import trafilatura

from src.data.vibiomir_contracts import (
    ExtractedBlock,
    ExtractedDocument,
    ExtractionConfig,
)

logger = logging.getLogger(__name__)


def decode_html_bytes(raw: bytes) -> str:
    """Safely decode HTML bytes to string handling utf-8 and fallbacks."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass

    try:
        import charset_normalizer
        best = charset_normalizer.from_bytes(raw).best()
        if best and best.encoding:
            return str(best)
    except Exception:
        pass

    try:
        return raw.decode("windows-1258")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def _extract_html_title(tree) -> str | None:
    """Extract page title without altering body text."""
    try:
        title_elems = tree.xpath("//title")
        if title_elems:
            t = "".join(title_elems[0].itertext()).strip()
            if t:
                return t
    except Exception:
        pass
    return None


def _clean_non_content_tags(tree) -> None:
    """Remove boilerplate non-content tags: nav, header, footer, scripts, etc."""
    tags_to_remove = (
        "script",
        "style",
        "noscript",
        "nav",
        "header",
        "footer",
        "aside",
        "svg",
        "iframe",
    )
    for tag in tags_to_remove:
        for el in tree.xpath(f"//{tag}"):
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)


def _parse_trafilatura_xml(xml_str: str) -> list[tuple[str, str | None, str]]:
    """Parse Trafilatura XML into structured (kind, section, text) blocks."""
    root = etree.fromstring(xml_str.encode("utf-8"))
    raw_blocks: list[tuple[str, str | None, str]] = []
    current_section: str | None = None

    for container in root.xpath("//main | //comments"):
        for elem in container.iterchildren():
            tag = elem.tag
            if tag == "head":
                t = "".join(elem.itertext()).strip()
                if t:
                    current_section = t
                    raw_blocks.append(("heading", current_section, t))
            elif tag == "p":
                t = "".join(elem.itertext()).strip()
                if t:
                    raw_blocks.append(("paragraph", current_section, t))
            elif tag == "list":
                items = ["".join(it.itertext()).strip() for it in elem.findall(".//item")]
                items = [it for it in items if it]
                if items:
                    t = "\n".join(f"- {it}" for it in items)
                    raw_blocks.append(("list", current_section, t))
            elif tag == "table":
                rows = []
                for row in elem.findall(".//row"):
                    cells = ["".join(c.itertext()).strip() for c in row.findall(".//cell")]
                    if any(cells):
                        rows.append("| " + " | ".join(cells) + " |")
                if rows:
                    t = "\n".join(rows)
                    raw_blocks.append(("table", current_section, t))
            else:
                t = "".join(elem.itertext()).strip()
                if t:
                    raw_blocks.append(("paragraph", current_section, t))

    return raw_blocks


def _extract_lxml_fallback(tree) -> list[tuple[str, str | None, str]]:
    """Direct lxml block extraction fallback if Trafilatura yields no blocks."""
    raw_blocks: list[tuple[str, str | None, str]] = []
    current_section: str | None = None

    for elem in tree.xpath("//h1 | //h2 | //h3 | //h4 | //h5 | //h6 | //p | //ul | //ol | //table | //blockquote"):
        tag = elem.tag.lower()
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            t = "".join(elem.itertext()).strip()
            if t:
                current_section = t
                raw_blocks.append(("heading", current_section, t))
        elif tag == "p":
            t = "".join(elem.itertext()).strip()
            if t:
                raw_blocks.append(("paragraph", current_section, t))
        elif tag in ("ul", "ol"):
            items = ["".join(li.itertext()).strip() for li in elem.xpath(".//li")]
            items = [it for it in items if it]
            if items:
                t = "\n".join(f"- {it}" for it in items)
                raw_blocks.append(("list", current_section, t))
        elif tag == "table":
            rows = []
            for tr in elem.xpath(".//tr"):
                cells = ["".join(c.itertext()).strip() for c in tr.xpath(".//th | .//td")]
                if any(cells):
                    rows.append("| " + " | ".join(cells) + " |")
            if rows:
                t = "\n".join(rows)
                raw_blocks.append(("table", current_section, t))
        elif tag == "blockquote":
            t = "".join(elem.itertext()).strip()
            if t:
                raw_blocks.append(("paragraph", current_section, t))

    return raw_blocks


def extract_html_document(
    doc_id: int,
    raw: bytes,
    raw_sha256: str,
    config: ExtractionConfig,
) -> ExtractedDocument:
    """Extract structured document from HTML using Trafilatura + lxml fallback."""
    html_str = decode_html_bytes(raw)

    try:
        tree = html.fromstring(html_str)
    except Exception as e:
        logger.warning(f"Failed to parse HTML for doc_id {doc_id}: {e}")
        return ExtractedDocument(
            doc_id=doc_id,
            source_version="trafilatura_v1",
            source_text="",
            source_sha256=hashlib.sha256(b"").hexdigest(),
            raw_sha256=raw_sha256,
            title=None,
            language="und",
            blocks=(),
            quality_flags=("extraction_failed",),
        )

    title = _extract_html_title(tree)
    _clean_non_content_tags(tree)

    raw_blocks: list[tuple[str, str | None, str]] = []

    # 1. Primary extractor: Trafilatura with XML output
    try:
        cleaned_html_bytes = html.tostring(tree, encoding="utf-8")
        xml_output = trafilatura.extract(
            cleaned_html_bytes,
            output_format="xml",
            include_tables=True,
            include_formatting=True,
            include_comments=True,
            favor_recall=True,
        )
        if xml_output:
            raw_blocks = _parse_trafilatura_xml(xml_output)
    except Exception as e:
        logger.debug(f"Trafilatura extraction failed for doc_id {doc_id}: {e}")
        raw_blocks = []

    # 2. Fallback extractor: Direct lxml traversal if Trafilatura produced no blocks
    if not raw_blocks:
        raw_blocks = _extract_lxml_fallback(tree)

    quality_flags: list[str] = []
    if not raw_blocks:
        quality_flags.append("empty_text")
        return ExtractedDocument(
            doc_id=doc_id,
            source_version="trafilatura_v1",
            source_text="",
            source_sha256=hashlib.sha256(b"").hexdigest(),
            raw_sha256=raw_sha256,
            title=title,
            language="und",
            blocks=(),
            quality_flags=tuple(quality_flags),
        )

    # 3. Deterministic source_text and exact Python character offset reconstruction
    blocks: list[ExtractedBlock] = []
    current_offset = 0
    for i, (kind, section, text) in enumerate(raw_blocks):
        if i > 0:
            current_offset += 2  # account for deterministic "\n\n" separator
        start_char = current_offset
        end_char = start_char + len(text)
        blocks.append(
            ExtractedBlock(
                block_index=i,
                kind=kind,
                start_char=start_char,
                end_char=end_char,
                section=section,
            )
        )
        current_offset = end_char

    source_text = "\n\n".join(t for _, _, t in raw_blocks)
    source_sha256 = hashlib.sha256(source_text.encode("utf-8")).hexdigest()

    return ExtractedDocument(
        doc_id=doc_id,
        source_version="trafilatura_v1",
        source_text=source_text,
        source_sha256=source_sha256,
        raw_sha256=raw_sha256,
        title=title,
        language="und",
        blocks=tuple(blocks),
        quality_flags=tuple(quality_flags),
    )


def extract_pdf_document(
    doc_id: int,
    raw: bytes,
    raw_sha256: str,
    config: ExtractionConfig,
) -> ExtractedDocument:
    """Extract structured document from PDF using PyMuPDF preserving block order."""
    try:
        import pymupdf
        pdf_doc = pymupdf.open(stream=raw, filetype="pdf")
    except Exception as e:
        logger.warning(f"Failed to open PDF stream for doc_id {doc_id}: {e}")
        return ExtractedDocument(
            doc_id=doc_id,
            source_version="pymupdf_v1",
            source_text="",
            source_sha256=hashlib.sha256(b"").hexdigest(),
            raw_sha256=raw_sha256,
            title=None,
            language="und",
            blocks=(),
            quality_flags=("extraction_failed",),
        )

    raw_blocks: list[tuple[str, str | None, str]] = []
    current_section: str | None = None

    try:
        for page_idx, page in enumerate(pdf_doc):
            page_blocks = page.get_text("blocks")
            for b in page_blocks:
                # b: (x0, y0, x1, y1, text, block_no, block_type)
                if len(b) >= 7 and b[6] == 0:  # block_type 0 = text
                    text = b[4].strip()
                    if text:
                        raw_blocks.append(("paragraph", current_section, text))
    finally:
        pdf_doc.close()

    if not raw_blocks:
        return ExtractedDocument(
            doc_id=doc_id,
            source_version="pymupdf_v1",
            source_text="",
            source_sha256=hashlib.sha256(b"").hexdigest(),
            raw_sha256=raw_sha256,
            title=None,
            language="und",
            blocks=(),
            quality_flags=("needs_ocr",),
        )

    blocks: list[ExtractedBlock] = []
    current_offset = 0
    for i, (kind, section, text) in enumerate(raw_blocks):
        if i > 0:
            current_offset += 2  # account for deterministic "\n\n" separator
        start_char = current_offset
        end_char = start_char + len(text)
        blocks.append(
            ExtractedBlock(
                block_index=i,
                kind=kind,
                start_char=start_char,
                end_char=end_char,
                section=section,
            )
        )
        current_offset = end_char

    source_text = "\n\n".join(t for _, _, t in raw_blocks)
    source_sha256 = hashlib.sha256(source_text.encode("utf-8")).hexdigest()

    return ExtractedDocument(
        doc_id=doc_id,
        source_version="pymupdf_v1",
        source_text=source_text,
        source_sha256=source_sha256,
        raw_sha256=raw_sha256,
        title=None,
        language="und",
        blocks=tuple(blocks),
        quality_flags=(),
    )


def extract_document(
    doc_id: int,
    raw: bytes,
    mime_type: str,
    raw_sha256: str,
    config: ExtractionConfig | None = None,
) -> ExtractedDocument:
    """Primary document extraction dispatcher for HTML and PDF official sources."""
    if config is None:
        config = ExtractionConfig()

    clean_mime = (mime_type or "").lower().split(";")[0].strip()

    if clean_mime in ("text/html", "application/xhtml+xml", "text/xml"):
        return extract_html_document(doc_id, raw, raw_sha256, config)
    elif clean_mime == "application/pdf":
        return extract_pdf_document(doc_id, raw, raw_sha256, config)
    else:
        return ExtractedDocument(
            doc_id=doc_id,
            source_version="v1",
            source_text="",
            source_sha256=hashlib.sha256(b"").hexdigest(),
            raw_sha256=raw_sha256,
            title=None,
            language="und",
            blocks=(),
            quality_flags=("unsupported_mime_type",),
        )

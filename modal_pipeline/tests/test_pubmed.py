"""Parser tests for PubMed records fetched from Europe PMC (no network)."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetch_pubmed import epmc_query, parse_epmc_result, pmid_from_url  # noqa: E402


def test_parse_labeled_sections_and_markup():
    rec = parse_epmc_result({
        "pmid": "27871086",
        "title": "Effect of <i>Helicobacter pylori</i> eradication.",
        "abstractText": "<h4>Background</h4>Peptic   ulcer is common.<h4>Results</h4>"
                        "Healing rose to 90% (p&lt;0.05) with H<sub>2</sub> blockers.",
        "journalInfo": {"journal": {"title": "Journal of Tests"}},
    })
    assert rec == {
        "pmid": 27871086,
        "title": "Effect of Helicobacter pylori eradication.",
        "journal": "Journal of Tests",
        "sections": [["Background", "Peptic ulcer is common."],
                     ["Results", "Healing rose to 90% (p<0.05) with H2 blockers."]],
    }


def test_parse_unlabeled_and_missing_abstract():
    plain = parse_epmc_result({"pmid": "11197", "title": "[Prolactin].", "abstractText": "Prolactin (PRL) is a hormone."})
    assert plain["sections"] == [[None, "Prolactin (PRL) is a hormone."]] and plain["journal"] is None
    bare = parse_epmc_result({"pmid": "5", "title": "Title only."})
    assert bare["sections"] == [] and bare["title"] == "Title only."


def test_query_and_pmid_from_url():
    assert epmc_query([1, 22]) == "EXT_ID:(1 OR 22) AND SRC:MED"
    assert pmid_from_url("https://pubmed.ncbi.nlm.nih.gov/27871086/") == 27871086
    assert pmid_from_url("https://pubmed.ncbi.nlm.nih.gov/27871086") == 27871086
    assert pmid_from_url("https://example.com/x") is None

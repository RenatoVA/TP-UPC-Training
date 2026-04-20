from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag

CATEGORY_NAMES = {
    "Corte Suprema",
    "Corte Superior",
    "Legislación",
    "Juzgados",
    "Derecho comparado",
    "Corte IDH",
    "Corte Interamericana de Derechos Humanos",
}

RESOLUTION_PATTERNS = [
    r"(Casación\s+[^\]]+)",
    r"(Apelación\s+[^\]]+)",
    r"(RN\s+[^\]]+)",
    r"(Exp\.\s+[^\]]+)",
    r"(AP\s+[^\]]+)",
    r"(Queja\s+[^\]]+)",
    r"(Consulta\s+[^\]]+)",
    r"(Revisión de Sentencia\s+[^\]]+)",
    r"(Rev\.\s*Sent\.?[^\]]+)",
    r"(RA\s+[^\]]+)",
    r"(Pleno\s+[^\]]+)",
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_concordances(block_text: str) -> list[dict[str, Any]]:
    items = []
    for part in block_text.split(";"):
        part = normalize_text(part)
        if not part or ":" not in part:
            continue
        code, refs = part.split(":", 1)
        code = normalize_text(code).replace(":", "")
        refs = normalize_text(refs)
        article_tokens = re.findall(r"\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?", refs)
        items.append({"code": code, "articles": article_tokens})
    return items


def extract_resolution_reference(text: str) -> str | None:
    for pattern in RESOLUTION_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return normalize_text(m.group(1))
    return None


def parse_resolution_metadata(reference: str | None) -> dict[str, Any]:
    if not reference:
        return {
            "resolution_type": None,
            "resolution_number": None,
            "resolution_year": None,
            "resolution_jurisdiction": None,
        }
    ref = normalize_text(reference)
    typ = ref.split(" ", 1)[0]
    year_match = re.search(r"(19|20)\d{2}", ref)
    year = int(year_match.group(0)) if year_match else None
    number_match = re.search(r"([0-9][0-9A-Za-z./-]*)", ref)
    number = number_match.group(1) if number_match else None
    jurisdiction = None
    if "," in ref:
        jurisdiction = normalize_text(ref.split(",", 1)[1])
    return {
        "resolution_type": typ,
        "resolution_number": number,
        "resolution_year": year,
        "resolution_jurisdiction": jurisdiction,
    }


def detect_entry_type(category: str, text: str, reference: str | None) -> str:
    lower = text.lower()
    if category == "Legislación":
        return "legislacion"
    if "doctrina legal" in lower:
        return "doctrina_legal"
    if category in {"Corte Suprema", "Corte Superior", "Juzgados", "Corte IDH", "Corte Interamericana de Derechos Humanos"}:
        return "jurisprudencia"
    return "otro"


def parse_index_entries(parsed_pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for page in parsed_pages:
        raw_path = Path(page["raw_html_path"])
        html = raw_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")
        content = soup.select_one("div.td-post-content")
        if not content:
            continue

        article_heading = None
        concordances = []
        entry_counter = 0

        for node in content.find_all(["h2", "p"]):
            text = normalize_text(node.get_text(" ", strip=True))
            if not text:
                continue
            if text.startswith("Artículo ") and article_heading is None:
                article_heading = text
            if ("art." in text.lower() or "arts." in text.lower()) and ":" in text and any(code in text for code in ["CP", "NCPP", "C:", "CC", "CADH", "DADDH"]):
                parsed = parse_concordances(text)
                if parsed:
                    concordances = parsed

        category_headers = []
        for tag in content.find_all(["h3", "li"]):
            text = normalize_text(tag.get_text(" ", strip=True))
            if text in CATEGORY_NAMES:
                category_headers.append(tag)

        seen_links = set()
        for cat in category_headers:
            category = normalize_text(cat.get_text(" ", strip=True))
            for sib in cat.find_next_siblings():
                if isinstance(sib, Tag):
                    sib_text = normalize_text(sib.get_text(" ", strip=True))
                    if sib.name in {"h3", "li"} and sib_text in CATEGORY_NAMES:
                        break
                    if sib.name != "li":
                        continue
                    if not sib_text or sib_text in CATEGORY_NAMES:
                        continue
                    links = sib.find_all("a", href=True)
                    lp_links = [a["href"] for a in links if "lpderecho.pe" in a["href"]]
                    if not lp_links:
                        continue
                    entry_link = lp_links[0]
                    if entry_link in seen_links:
                        continue
                    seen_links.add(entry_link)
                    short_link = lp_links[-1] if lp_links else None
                    entry_counter += 1
                    flags = []
                    if sib_text.startswith("NUEVO") or sib_text.startswith("NUEVO:"):
                        flags.append("NUEVO")
                    reference = extract_resolution_reference(sib_text)
                    metadata = parse_resolution_metadata(reference)
                    entry_type = detect_entry_type(category, sib_text, reference)
                    entry_id = f"{page['id']}_e{entry_counter:04d}"
                    related_norms = []
                    if page.get("linked_articles"):
                        anchor_article = page["linked_articles"][0]
                        related_norms.append(
                            {
                                "code": anchor_article.get("source_dataset", ""),
                                "article_id": anchor_article.get("article_id"),
                                "relation_type": "page_anchor",
                                "confidence": "high",
                            }
                        )
                    for c in concordances:
                        for article in c.get("articles", []):
                            related_norms.append(
                                {
                                    "code": c.get("code"),
                                    "article_id": article,
                                    "relation_type": "page_concordance",
                                    "confidence": "high",
                                }
                            )
                    results.append(
                        {
                            "entry_id": entry_id,
                            "page_id": page["id"],
                            "page_source_url": page["jurisprudence_url"],
                            "entry_category": category,
                            "entry_type": entry_type,
                            "entry_prefix_flags": flags,
                            "entry_text": sib_text,
                            "entry_title_candidate": sib_text.split("[")[0].replace("Link:", "").strip(),
                            "resolution_reference_raw": reference,
                            "lp_entry_url": entry_link,
                            "short_link_url": short_link,
                            "page_anchor_code": None,
                            "page_anchor_article_id": page.get("linked_articles", [{}])[0].get("article_id"),
                            "page_anchor_article_heading": page.get("linked_articles", [{}])[0].get("heading"),
                            "page_concordances": concordances,
                            "related_norms": related_norms,
                            "domain_relevance_candidate": None,
                            **metadata,
                        }
                    )
    return results

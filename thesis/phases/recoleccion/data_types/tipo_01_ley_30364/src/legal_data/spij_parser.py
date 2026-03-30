from __future__ import annotations

import re
from bs4 import BeautifulSoup

from legal_data.law_model import Article, LawDocument


ARTICLE_RE = re.compile(r"^art[ií]culo\s+(\d+[A-Za-z-]*)", re.IGNORECASE)


def html_to_text_blocks(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    blocks: list[str] = []
    for node in soup.find_all(["p", "h1", "h2", "h3", "li"]):
        text = node.get_text(" ", strip=True)
        if text:
            blocks.append(re.sub(r"\s+", " ", text).strip())
    return blocks


def parse_spij_detail_to_law(detail: dict) -> LawDocument:
    html = detail.get("textoCompleto") or ""
    blocks = html_to_text_blocks(html)

    title = ""
    articles: list[Article] = []
    current_article_id: str | None = None
    current_heading: str | None = None
    current_lines: list[str] = []

    for block in blocks:
        if not title and detail.get("codigoNorma") and detail.get("codigoNorma") in block:
            title = detail.get("sumilla", "")

        match = ARTICLE_RE.match(block)
        if match:
            if current_article_id is not None:
                articles.append(
                    Article(
                        article_id=current_article_id,
                        heading=current_heading,
                        text="\n".join(current_lines).strip(),
                    )
                )
            current_article_id = match.group(1)
            current_heading = block
            current_lines = [block]
        elif current_article_id is not None:
            current_lines.append(block)

    if current_article_id is not None:
        articles.append(
            Article(
                article_id=current_article_id,
                heading=current_heading,
                text="\n".join(current_lines).strip(),
            )
        )

    return LawDocument(
        norm_id=detail.get("id", "H1141065"),
        code=detail.get("codigoNorma", "LEY 30364"),
        title=BeautifulSoup(detail.get("sumilla", ""), "lxml").get_text(" ", strip=True),
        source="SPIJ",
        articles=articles,
        complementary_provisions=[],
        annexes=[],
        metadata={
            "fechaPublicacion": detail.get("fechaPublicacion"),
            "sector": detail.get("sector"),
            "dispositivoLegal": detail.get("dispositivoLegal"),
        },
    )

"""Render the manuscript content blocks into a Word document.

The existing manuscript file is used as the template, so its JMIR styles
(Heading 2, Heading 3, Normal, table styling) carry over and the output is the
same kind of document the journal expects. Its body is cleared and rewritten
from the content module; nothing of the old text survives.

Document properties are cleared and reset to the author, so no tool name or
previous editor's name is left in the file metadata.

Usage:
    python scripts/build_manuscript.py --lang en
    python scripts/build_manuscript.py --lang id
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor, Inches

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "scripts"))

TEMPLATE = os.path.join(PROJECT, "JurnalJMIR-BenedictAmadeusSandro.docx")
FIG_WIDTH_IN = 6.1
MAX_FIG_WIDTH_IN = 6.3


def clear_body(doc: Document) -> None:
    """Remove every block-level element from the document body.

    Section properties (page size, margins) live in the final sectPr element and
    are kept, so the page setup of the template is preserved.
    """
    body = doc.element.body
    for child in list(body):
        if child.tag.endswith("}sectPr"):
            continue
        body.remove(child)


def _p(doc, text, style=None, size=None, bold=False, italic=False,
       align=None, space_after=6):
    para = doc.add_paragraph(style=style)
    run = para.add_run(text)
    run.bold = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    if align is not None:
        para.alignment = align
    para.paragraph_format.space_after = Pt(space_after)
    return para


def _caption(doc, text):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.font.size = Pt(9)
    run.italic = True
    para.paragraph_format.space_after = Pt(12)
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return para


def _image(doc, path, caption):
    full = path if os.path.isabs(path) else os.path.join(PROJECT, path)
    if not os.path.exists(full):
        _p(doc, f"[figure not found: {path}]", italic=True)
        _caption(doc, caption)
        return False
    from PIL import Image as PILImage
    with PILImage.open(full) as im:
        w, h = im.size
    # Keep tall diagrams from overflowing the page.
    width_in = min(MAX_FIG_WIDTH_IN, FIG_WIDTH_IN if h / w < 1.3 else 4.3)
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.add_run().add_picture(full, width=Inches(width_in))
    para.paragraph_format.space_after = Pt(4)
    _caption(doc, caption)
    return True


def _table(doc, caption, headers, rows):
    _caption(doc, caption)
    t = doc.add_table(rows=1, cols=len(headers))
    try:
        t.style = "Table Grid"
    except KeyError:
        pass
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        run = hdr[i].paragraphs[0].add_run(str(h))
        run.bold = True
        run.font.size = Pt(9)
    for r in rows:
        cells = t.add_row().cells
        for i, val in enumerate(r):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(val))
            run.font.size = Pt(9)
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    return t


def render(blocks: List[Tuple[str, object]], out_path: str, template: str = TEMPLATE) -> str:
    doc = Document(template) if os.path.exists(template) else Document()
    clear_body(doc)

    for kind, payload in blocks:
        if kind == "title":
            _p(doc, payload, size=16, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=10)
        elif kind == "authors":
            _p(doc, "; ".join(payload), size=11, space_after=2)
        elif kind == "affil":
            for line in payload:
                _p(doc, line, size=10, italic=True, space_after=2)
        elif kind == "corr":
            _p(doc, payload, size=10, space_after=14)
        elif kind == "h2":
            doc.add_paragraph(payload, style="Heading 2")
        elif kind == "h3":
            doc.add_paragraph(payload, style="Heading 3")
        elif kind == "p":
            _p(doc, payload, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
        elif kind == "abs":
            label, text = payload
            para = doc.add_paragraph()
            r1 = para.add_run(f"{label}: ")
            r1.bold = True
            para.add_run(text)
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            para.paragraph_format.space_after = Pt(6)
        elif kind == "fig":
            path, caption = payload
            _image(doc, path, caption)
        elif kind == "tbl":
            caption, headers, rows = payload
            _table(doc, caption, headers, rows)
        elif kind == "refs":
            for i, ref in enumerate(payload, 1):
                para = doc.add_paragraph()
                para.add_run(f"{i}. {ref}").font.size = Pt(9)
                para.paragraph_format.space_after = Pt(4)
        else:
            raise ValueError(f"unknown block kind: {kind}")

    # --- document properties: author only, nothing else ---
    cp = doc.core_properties
    cp.author = "Benedict Amadeus Sandro"
    cp.last_modified_by = "Benedict Amadeus Sandro"
    cp.title = next(p for k, p in blocks if k == "title")
    cp.comments = ""
    cp.category = ""
    cp.keywords = ""
    cp.subject = ""
    cp.content_status = ""
    cp.identifier = ""
    cp.language = ""
    cp.version = ""

    doc.save(out_path)
    return out_path


def strip_orphan_media(path: str) -> dict:
    """Remove media parts the rebuilt document no longer references.

    Clearing the body drops the paragraphs that displayed the old figures, but
    the image parts and their relationships survive in the package. Left alone
    they make the file several megabytes larger and, worse, mean the submitted
    document still carries figures from a superseded draft.
    """
    import re
    import shutil
    import zipfile

    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        doc_xml = z.read("word/document.xml").decode("utf8")
        rels_name = "word/_rels/document.xml.rels"
        rels_xml = z.read(rels_name).decode("utf8")
        payload = {n: z.read(n) for n in names}

    used_ids = set(re.findall(r'r:(?:embed|link)="([^"]+)"', doc_xml))
    rel_entries = re.findall(r'<Relationship\b[^>]*/>', rels_xml)

    keep_rels, drop_targets = [], set()
    for entry in rel_entries:
        rid = re.search(r'Id="([^"]+)"', entry)
        target = re.search(r'Target="([^"]+)"', entry)
        is_media = target and "media/" in target.group(1)
        if is_media and rid and rid.group(1) not in used_ids:
            drop_targets.add("word/" + target.group(1).lstrip("./"))
            continue
        keep_rels.append(entry)

    if not drop_targets:
        return {"removed": 0, "bytes": 0}

    new_rels = rels_xml
    for entry in rel_entries:
        if entry not in keep_rels:
            new_rels = new_rels.replace(entry, "")

    freed = sum(len(payload[t]) for t in drop_targets if t in payload)
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        for n in names:
            if n in drop_targets:
                continue
            out.writestr(n, new_rels.encode("utf8") if n == rels_name else payload[n])
    shutil.move(tmp, path)
    return {"removed": len(drop_targets), "bytes": freed}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the manuscript .docx.")
    ap.add_argument("--lang", choices=["en", "id"], default="en")
    ap.add_argument("--out", default=None)
    ap.add_argument("--template", default=TEMPLATE)
    args = ap.parse_args()

    from manuscript_numbers import all_numbers
    N = all_numbers()

    if args.lang == "en":
        from manuscript_content_en import build
        default_out = os.path.join(PROJECT, "JurnalJMIR-BenedictAmadeusSandro.docx")
    else:
        from manuscript_content_id import build
        default_out = os.path.join(PROJECT, "JurnalJMIR-BenedictAmadeusSandro-ID.docx")

    blocks = build(N)
    out = args.out or default_out
    render(blocks, out, args.template)
    stripped = strip_orphan_media(out)

    n_fig = sum(1 for k, _ in blocks if k == "fig")
    n_tbl = sum(1 for k, _ in blocks if k == "tbl")
    n_par = sum(1 for k, _ in blocks if k == "p")
    print(f"wrote {out}  ({os.path.getsize(out):,} bytes)")
    print(f"  {n_par} paragraphs, {n_fig} figures, {n_tbl} tables")
    if stripped["removed"]:
        print(f"  removed {stripped['removed']} orphaned media parts from the previous "
              f"draft ({stripped['bytes']:,} bytes)")


if __name__ == "__main__":
    main()

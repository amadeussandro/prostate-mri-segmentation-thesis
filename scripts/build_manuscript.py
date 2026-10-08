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
# The journal template gives a 6.00 inch text column; a figure wider than that
# overflows into the margin.
TEXT_WIDTH_IN = 6.0
FIG_WIDTH_IN = 6.0
TALL_FIG_WIDTH_IN = 4.1


_CITE = __import__("re").compile(r"\[@([A-Za-z0-9_,]+)\]")


def resolve_citations(blocks: List[Tuple[str, object]], library: dict) -> Tuple[list, list]:
    """Turn ``[@key]`` and ``[@key1,key2]`` markers into numbered citations.

    Numbers are assigned in order of first appearance, which is what the
    journal's instructions ask for, and the reference list is emitted in that
    same order containing only the entries actually cited. Keeping the numbering
    derived rather than written by hand is what makes adding a reference safe:
    the alternative is renumbering every marker in the prose by hand, which is
    how a citation ends up pointing at the wrong paper.
    """
    order: List[str] = []

    def sub(text: str) -> str:
        def one(m):
            nums = []
            for key in m.group(1).split(","):
                if key not in library:
                    raise KeyError(f"citation [@{key}] has no entry in the reference library")
                if key not in order:
                    order.append(key)
                nums.append(order.index(key) + 1)
            return "[" + ",".join(str(n) for n in sorted(nums)) + "]"
        return _CITE.sub(one, text)

    resolved = []
    for kind, payload in blocks:
        if kind in ("p", "title", "corr"):
            resolved.append((kind, sub(payload)))
        elif kind in ("authors", "affil"):
            resolved.append((kind, [sub(x) for x in payload]))
        elif kind == "abs":
            resolved.append((kind, (payload[0], sub(payload[1]))))
        elif kind == "fig":
            resolved.append((kind, (payload[0], sub(payload[1]))))
        elif kind == "tbl":
            cap, hdr, rows = payload
            resolved.append((kind, (sub(cap), hdr,
                                    [[sub(str(c)) for c in r] for r in rows])))
        elif kind == "refs":
            resolved.append((kind, "__PENDING__"))
        else:
            resolved.append((kind, payload))

    numbered = [library[k] for k in order]
    out = [(k, numbered if (k == "refs" and v == "__PENDING__") else v) for k, v in resolved]

    # A document that carries bracketed citations but resolved no keys has prose
    # still using hard-coded numbers; it would ship with a populated body and an
    # empty reference list, which is worse than failing here.
    if not order:
        import re as _re
        body = " ".join(p for k, p in blocks if k == "p" and isinstance(p, str))
        if _re.search(r"\[\d+(,\d+)*\]", body):
            raise ValueError(
                "No [@key] citations resolved, but the text contains bracketed numbers. "
                "The content module is still using hard-coded citation numbers - convert "
                "them to [@key] markers so the numbering stays derived from the library.")
    return out, order


def apply_jmir_page_setup(doc: Document) -> dict:
    """Force US Letter portrait with the journal template's margins.

    The file this builder uses as its template carries an 11 x 8.5 inch page -
    landscape dimensions, even though its orientation flag still reads portrait,
    which is the inconsistent state Word produces when the page size is edited
    directly. The journal's own template is 8.5 x 11 portrait with 1.25 inch
    side margins and 1 inch top and bottom, giving a 6 inch text column, so that
    is what is set here rather than whatever the previous draft happened to have.
    """
    from docx.enum.section import WD_ORIENT
    for s in doc.sections:
        s.orientation = WD_ORIENT.PORTRAIT
        s.page_width = Inches(8.5)
        s.page_height = Inches(11)
        s.left_margin = s.right_margin = Inches(1.25)
        s.top_margin = s.bottom_margin = Inches(1.0)
    sec = doc.sections[0]
    # Subtracting Length objects yields a plain int (EMU), so convert first.
    text_width = sec.page_width.inches - sec.left_margin.inches - sec.right_margin.inches
    return {"page": (sec.page_width.inches, sec.page_height.inches),
            "text_width": text_width}


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
    width_in = FIG_WIDTH_IN if h / w < 1.3 else TALL_FIG_WIDTH_IN
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
        # A "<<MERGE>>" cell joins the one before it, for a value that belongs
        # to the row as a whole rather than to one column - a macro average
        # across both zones, say. An empty cell there would read as missing data.
        merge_from = None
        for i, val in enumerate(r):
            if str(val) == "<<MERGE>>":
                if merge_from is None:
                    merge_from = i - 1
                continue
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(val))
            run.font.size = Pt(9)
        if merge_from is not None:
            cells[merge_from].merge(cells[len(r) - 1])
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    return t


def render(blocks: List[Tuple[str, object]], out_path: str, template: str = TEMPLATE) -> str:
    doc = Document(template) if os.path.exists(template) else Document()
    apply_jmir_page_setup(doc)
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


_DECIMAL = __import__("re").compile(r"(?<![\w/.])(\d+)\.(\d+)(?![\w/.])")


def to_indonesian_decimals(text: str) -> str:
    """Render decimal points as commas, the Indonesian convention.

    The numbers arrive formatted by Python, which always emits a point. Mixing
    the two inside one document is worse than either convention, and parts of
    the prose already use commas (1,96 SD; 0,15 ng/mL/cc).

    The pattern deliberately refuses to fire when the number is touching a word
    character, a slash or another dot, which leaves version strings, file
    extensions, URLs and already-grouped thousands alone.
    """
    return _DECIMAL.sub(lambda m: f"{m.group(1)},{m.group(2)}", text)


def localize(blocks: List[Tuple[str, object]], fn) -> List[Tuple[str, object]]:
    """Apply a text transform to every human-readable string in the blocks."""
    out = []
    for kind, payload in blocks:
        if kind in ("p", "title", "corr"):
            out.append((kind, fn(payload)))
        elif kind in ("authors", "affil", "refs"):
            out.append((kind, [fn(x) for x in payload]))
        elif kind == "abs":
            out.append((kind, (payload[0], fn(payload[1]))))
        elif kind == "fig":
            out.append((kind, (payload[0], fn(payload[1]))))
        elif kind == "tbl":
            caption, headers, rows = payload
            out.append((kind, (fn(caption), [fn(str(h)) for h in headers],
                               [[fn(str(c)) for c in r] for r in rows])))
        else:
            out.append((kind, payload))
    return out


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
    from manuscript_content_en import REFERENCES
    blocks, cited = resolve_citations(blocks, REFERENCES)
    if args.lang == "id":
        blocks = localize(blocks, to_indonesian_decimals)
    out = args.out or default_out
    render(blocks, out, args.template)
    stripped = strip_orphan_media(out)

    n_fig = sum(1 for k, _ in blocks if k == "fig")
    n_tbl = sum(1 for k, _ in blocks if k == "tbl")
    n_par = sum(1 for k, _ in blocks if k == "p")
    print(f"wrote {out}  ({os.path.getsize(out):,} bytes)")
    print(f"  {n_par} paragraphs, {n_fig} figures, {n_tbl} tables")
    print(f"  {len(cited)} references, numbered in order of first citation")
    if stripped["removed"]:
        print(f"  removed {stripped['removed']} orphaned media parts from the previous "
              f"draft ({stripped['bytes']:,} bytes)")


if __name__ == "__main__":
    main()

"""Parse eCFR XML into citable passages.

Rule text: each `DIV8 TYPE="SECTION"` holds `P` elements whose paragraph labels are
inline ("(a) ... (1) ..."). A small stack machine rebuilds the full paragraph path,
so every passage carries its official citation, e.g. "12 CFR 1005.11(c)(1)".

Official interpretations: `Supplement I to Part N` uses HD1 for the section, HD2 for
the paragraph ("11(c) Time Limits ..."), and numbered `P` comments ("1. Title. ...").
Each comment becomes one passage cited as "Supplement I to Part N, comment 11(c)-1".
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field

LABEL = re.compile(r"\(([a-z]{1,3}|[0-9]{1,2}|[ivxl]{1,6}|[A-Z]{1,3})\)")
ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv",
         "xvi", "xvii", "xviii", "xix", "xx", "xxi", "xxii", "xxiii", "xxiv", "xxv"}
COMMENT = re.compile(r"^(\d{1,2})\.\s+(.*)", re.DOTALL)
SUPP_PARA = re.compile(r"^(?:Paragraph\s+)?(\d+[a-z]?(?:\([a-zA-Z0-9]{1,4}\))*)\s*(.*)$")


@dataclass
class Passage:
    part: str
    section: str              # "1005.11"; for interpretations the section they interpret
    kind: str                 # "rule" | "interpretation"
    citation: str             # human citation
    label: str                # "(c)(1)(i)" for rules, "11(c)-1" for comments
    heading: str              # section heading
    text: str
    children: list[str] = field(default_factory=list)


def _text(el: ET.Element) -> str:
    return re.sub(r"\s+", " ", ET.tostring(el, encoding="unicode", method="text")).strip()


FIRST = {1: "a", 2: "1", 3: "i", 4: "A", 5: "1", 6: "i"}
_ROMAN_ORDER = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv",
                "xvi", "xvii", "xviii", "xix", "xx", "xxi", "xxii", "xxiii", "xxiv", "xxv"]


def _successor(tok: str, level: int) -> str | None:
    if level in (2, 5):
        return str(int(tok) + 1) if tok.isdigit() else None
    if level in (3, 6):
        return _ROMAN_ORDER[_ROMAN_ORDER.index(tok) + 1] if tok in _ROMAN_ORDER[:-1] else None
    # letters run a..z, then aa, bb, ..., zz, then aaa, bbb, ...
    if tok[0] in ("z", "Z"):
        return ("a" if tok.islower() else "A") * (len(tok) + 1)
    return chr(ord(tok[0]) + 1) * len(tok)


def _candidates(tok: str) -> list[int]:
    """CFR levels: 1 (a), 2 (1), 3 (i), 4 (A), 5 italic (1), 6 italic (i)."""
    if tok.isdigit():
        return [2, 5]
    if tok.isupper():
        return [4]
    return [3, 6, 1] if tok in ROMAN else [1]


def _level(tok: str, stack: list[tuple[int, str]]) -> int:
    """Pick the level where `tok` is the next sibling of an open paragraph, or the first child of the
    current deepest one. Siblings are preferred deepest-first; this resolves (i) vs (i)-the-letter and
    italic (1) under (A) vs a new top-level (1)."""
    top = stack[-1][0] if stack else 0
    best: tuple[int, int] | None = None  # (score, level)
    for lv in _candidates(tok):
        open_at = [t for l, t in stack if l == lv]
        if open_at and _successor(open_at[-1], lv) == tok:
            score = 100 + lv  # sibling of an open paragraph, deeper wins
        elif tok == FIRST[lv] and lv == top + 1:
            score = 50  # first child of the current paragraph
        else:
            continue
        if best is None or score > best[0]:
            best = (score, lv)
    if best:
        return best[1]
    # skip-level child: e.g. "(1)" directly under "(ii)" (CFR sometimes omits the (A) level)
    deeper = [lv for lv in _candidates(tok) if tok == FIRST[lv] and lv > top]
    if deeper:
        return min(deeper)
    # malformed numbering: fall back to the shallowest plausible level
    return _candidates(tok)[-1] if tok in ROMAN and not stack else _candidates(tok)[0]


def _leading_labels(text: str) -> tuple[list[str], str]:
    """Labels at the start of a paragraph: "(D)(1) ...", or "(a) Heading—(1) ..." across an em dash."""
    labels, pos = [], 0
    while True:
        m = LABEL.match(text, pos)
        if not m:
            break
        labels.append(m.group(1))
        pos = m.end()
        if LABEL.match(text, pos):  # adjacent labels, e.g. "(D)(1)"
            continue
        rest = text[pos:]
        # a short paragraph heading ending in an em dash or a period may precede the next label:
        # "(c) Time limits—(1) ..." and "(c) Time for resolution; general procedures. (1) ..."
        dash = re.match(r"\s*[^()—.]{0,120}?(?:—|\.)\s*(?=\()", rest)
        if dash and LABEL.match(rest, dash.end()):
            pos += dash.end()
        else:
            break
    return labels, text


def parse_rules(section_el: ET.Element, part: str) -> list[Passage]:
    n = section_el.get("N")
    heading = _text(section_el.find("HEAD")) if section_el.find("HEAD") is not None else n
    stack: list[tuple[int, str]] = []
    out: list[Passage] = []
    for p in section_el:
        if p.tag not in ("P", "FP", "FP-1", "FP-2"):
            continue
        text = _text(p)
        if not text:
            continue
        labels, _ = _leading_labels(text)
        for tok in labels:
            lv = _level(tok, stack)
            while stack and stack[-1][0] >= lv:
                stack.pop()
            stack.append((lv, tok))
        label = "".join(f"({t})" for _, t in stack)
        if out and not labels and out[-1].label == label:
            out[-1].text += " " + text  # continuation paragraph
            continue
        out.append(Passage(part=part, section=n, kind="rule", citation=f"12 CFR {n}{label}",
                           label=label, heading=heading, text=text))
    return out


def parse_interpretations(supp_el: ET.Element, part: str, warnings: list[str] | None = None) -> list[Passage]:
    """Comments are grouped into blocks per paragraph heading. A numbering restart without a new
    heading means either (a) the source repeats a block (first sentence matches an existing comment):
    the repeat is dropped and logged, or (b) the source lost a heading: the paragraph is inferred from
    the block's own references to this section, else the block is marked unlabeled and logged."""
    warnings = warnings if warnings is not None else []
    out: list[Passage] = []
    section, sec_head, para = None, "", None
    block: list[Passage] = []
    last_num = 0

    def flush(blk: list[Passage]) -> None:
        if not blk:
            return
        restart = blk[0].label.endswith("-RESTART")
        if not restart:
            out.extend(blk)
            return
        base_para = blk[0].label.rsplit("-", 2)[0]
        firsts = {p.text[:160] for p in out if p.label.startswith(base_para + "-")}
        if blk[0].text[:160] in firsts:
            warnings.append(f"dropped repeated comment block under {part} {base_para} ({len(blk)} comments)")
            return
        sec_short = blk[0].section.split(".")[1]
        refs = Counter(m.group(1) for p in blk for m in re.finditer(
            rf"{re.escape(blk[0].section)}((?:\([a-z0-9]+\))+)", p.text))
        new_para = f"{sec_short}{refs.most_common(1)[0][0]}" if refs else f"{base_para}[unlabeled]"
        warnings.append(f"comment block under {part} {base_para} has no heading; assigned {new_para}")
        for p in blk:
            num = p.label.split("-")[1]
            p.label = f"{new_para}-{num}"
            p.citation = f"Supplement I to Part {part}, comment {p.label}"
        out.extend(blk)

    for el in supp_el:
        text = _text(el)
        sec = re.match(r"Section\s+(\d+)\.(\d+[a-z]?)", text) if el.tag in ("HD1", "HD2", "HD3") else None
        new_para = None
        if sec:
            # Reg E puts sections in HD1, Reg Z in HD2; comments directly under a section cite as "12-1"
            section, sec_head, new_para = f"{sec.group(1)}.{sec.group(2)}", text, sec.group(2)
        elif el.tag == "HD1":
            flush(block); block, last_num = [], 0
            section, para = None, None  # subpart headings
            continue
        elif el.tag in ("HD2", "HD3"):
            m = SUPP_PARA.match(text)
            if m and section:
                new_para = m.group(1)
        elif el.tag == "P" and section and para and re.match(
                r"^(?:Paragraph\s+)?\d+[a-z]?\([a-zA-Z0-9]{1,4}\)", text) and len(text) < 160:
            new_para = SUPP_PARA.match(text).group(1)  # paragraph heading stored as a P element
        if new_para:
            flush(block); block, last_num, para = [], 0, new_para
            continue
        if el.tag == "P" and section and para:
            m = COMMENT.match(text)
            if m:
                num = int(m.group(1))
                if num <= last_num:  # numbering restarted with no heading
                    flush(block); block = []
                    lbl = f"{para}-{num}-RESTART"
                else:
                    lbl = f"{para}-{num}" if not (block and block[0].label.endswith("-RESTART")) else f"{para}-{num}-RESTART"
                last_num = num
                block.append(Passage(part=part, section=section, kind="interpretation",
                                     citation=f"Supplement I to Part {part}, comment {para}-{num}",
                                     label=lbl, heading=sec_head, text=text))
            elif block:
                block[-1].text += " " + text  # sub-items "i. ..." continue the comment
    flush(block)
    for p in out:
        p.label = p.label.replace("-RESTART", "")
    return out


def parse_part(xml_bytes: bytes, part: str, sections: set[str] | None = None,
               warnings: list[str] | None = None) -> list[Passage]:
    root = ET.fromstring(xml_bytes)
    passages: list[Passage] = []
    for el in root.iter():
        if el.tag == "DIV8" and el.get("TYPE") == "SECTION":
            if sections is None or el.get("N") in sections:
                passages.extend(parse_rules(el, part))
        elif el.tag == "DIV9" and (el.get("N") or "").startswith("Supplement I"):
            local: list[str] = []
            interps = parse_interpretations(el, part, local)
            if sections is not None:
                interps = [p for p in interps if p.section in sections]
                shorts = {s.split(".")[1] for s in sections}
                local = [w for w in local if re.search(rf"under {part} ({'|'.join(shorts)})[(\s]", w)]
            if warnings is not None:
                warnings.extend(local)
            passages.extend(interps)
    # the source repeats some comments verbatim; keep the first copy
    seen, unique = set(), []
    for p in passages:
        key = (p.citation, p.text)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique

#!/usr/bin/env python3
"""Render the Chinese post-training roadmap Markdown into a polished PDF."""

from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import urlparse

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)
from reportlab.platypus.tableofcontents import TableOfContents


REPORT_DIR = Path(__file__).resolve().parent
MARKDOWN_PATH = REPORT_DIR / "post-training-roadmap.md"
OUTPUT_PATH = REPORT_DIR / "post-training-roadmap.pdf"

PAGE_W, PAGE_H = A4
LEFT = 17 * mm
RIGHT = 17 * mm
TOP = 19 * mm
BOTTOM = 17 * mm
CONTENT_W = PAGE_W - LEFT - RIGHT

INK = HexColor("#172033")
MUTED = HexColor("#667085")
NAVY = HexColor("#0B1736")
BLUE = HexColor("#246BFD")
CYAN = HexColor("#34C9B5")
PALE_BLUE = HexColor("#EEF4FF")
PALE_CYAN = HexColor("#EAFBF7")
PALE_GOLD = HexColor("#FFF8E7")
GOLD = HexColor("#E6A83E")
LINE = HexColor("#D9E0EB")
WHITE = colors.white


def register_fonts() -> tuple[str, str]:
    candidates = [
        (
            Path("/System/Library/Fonts/STHeiti Light.ttc"),
            Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        ),
        (
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        ),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("ReportSans", str(regular), subfontIndex=0))
            pdfmetrics.registerFont(TTFont("ReportSansBold", str(bold), subfontIndex=0))
            pdfmetrics.registerFontFamily(
                "ReportSans",
                normal="ReportSans",
                bold="ReportSansBold",
                italic="ReportSans",
                boldItalic="ReportSansBold",
            )
            return "ReportSans", "ReportSansBold"
    raise FileNotFoundError("No suitable CJK font found")


FONT, FONT_BOLD = register_fonts()


def styles():
    base = getSampleStyleSheet()
    result = {
        "CoverKicker": ParagraphStyle(
            "CoverKicker",
            parent=base["Normal"],
            fontName=FONT_BOLD,
            fontSize=9.5,
            leading=12,
            textColor=CYAN,
            spaceAfter=18,
            tracking=1.4,
        ),
        "CoverTitle": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName=FONT_BOLD,
            fontSize=29,
            leading=38,
            textColor=WHITE,
            alignment=TA_LEFT,
            spaceAfter=16,
            wordWrap="CJK",
        ),
        "CoverSubtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=13.5,
            leading=21,
            textColor=HexColor("#D8E4FF"),
            spaceAfter=22,
            wordWrap="CJK",
        ),
        "CoverMeta": ParagraphStyle(
            "CoverMeta",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=9.5,
            leading=15,
            textColor=HexColor("#AFC4EE"),
            wordWrap="CJK",
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName=FONT_BOLD,
            fontSize=20,
            leading=27,
            textColor=NAVY,
            spaceBefore=2,
            spaceAfter=12,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName=FONT_BOLD,
            fontSize=14.5,
            leading=21,
            textColor=HexColor("#184DAF"),
            spaceBefore=14,
            spaceAfter=7,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "H3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName=FONT_BOLD,
            fontSize=11.8,
            leading=17,
            textColor=INK,
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.7,
            leading=15.1,
            textColor=INK,
            spaceAfter=7,
            wordWrap="CJK",
            allowWidows=0,
            allowOrphans=0,
        ),
        "BodyTight": ParagraphStyle(
            "BodyTight",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=8.5,
            leading=12.1,
            textColor=INK,
            wordWrap="CJK",
        ),
        "List": ParagraphStyle(
            "List",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.4,
            leading=14.5,
            textColor=INK,
            leftIndent=1,
            wordWrap="CJK",
        ),
        "Quote": ParagraphStyle(
            "Quote",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.4,
            leading=15,
            textColor=HexColor("#24435F"),
            wordWrap="CJK",
        ),
        "Code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName=FONT,
            fontSize=7.5,
            leading=11.4,
            textColor=HexColor("#20324D"),
            leftIndent=0,
            rightIndent=0,
            wordWrap="CJK",
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName=FONT_BOLD,
            fontSize=7.6,
            leading=10.5,
            textColor=WHITE,
            wordWrap="CJK",
        ),
        "TableBody": ParagraphStyle(
            "TableBody",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.3,
            leading=10.4,
            textColor=INK,
            wordWrap="CJK",
        ),
        "TOCTitle": ParagraphStyle(
            "TOCTitle",
            parent=base["Heading1"],
            fontName=FONT_BOLD,
            fontSize=22,
            leading=28,
            textColor=NAVY,
            spaceAfter=16,
        ),
        "Small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.6,
            leading=11,
            textColor=MUTED,
            wordWrap="CJK",
        ),
    }
    return result


ST = styles()


def strip_markup(text: str) -> str:
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"[*`_]", "", text)
    return html.unescape(text)


def short_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if len(path) > 34:
        path = path[:31] + "…"
    return f"{parsed.netloc}{path}"


def inline_md(text: str) -> str:
    text = html.escape(text, quote=False)

    link_tokens: list[str] = []

    def hold_link(match: re.Match[str]) -> str:
        label, url = match.group(1), match.group(2)
        label = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", label)
        label = re.sub(
            r"`([^`]+)`",
            r'<font name="ReportSansBold">\1</font>',
            label,
        )
        token = f"@@LINK{len(link_tokens)}@@"
        link_tokens.append(
            f'<link href="{url}" color="#246BFD"><u>{label}</u></link>'
        )
        return token

    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", hold_link, text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(
        r"`([^`]+)`",
        r'<font name="ReportSansBold" color="#174EA6">\1</font>',
        text,
    )
    if re.fullmatch(r"https?://[^\s]+", html.unescape(text)):
        raw = html.unescape(text)
        text = f'<link href="{raw}" color="#246BFD"><u>{html.escape(short_url(raw))}</u></link>'
    for index, link in enumerate(link_tokens):
        text = text.replace(f"@@LINK{index}@@", link)
    return text


def weighted_len(value: str) -> float:
    plain = strip_markup(value)
    cjk = sum(1 for char in plain if "\u2e80" <= char <= "\u9fff")
    return min(54.0, max(8.0, len(plain) + cjk * 0.7))


def table_widths(rows: list[list[str]]) -> list[float]:
    count = max(len(row) for row in rows)
    scores = []
    for col in range(count):
        values = [row[col] if col < len(row) else "" for row in rows]
        longest = max(weighted_len(value) for value in values)
        scores.append(max(10.0, longest ** 0.72))
    min_width = 42 if count >= 5 else 54
    raw = [CONTENT_W * score / sum(scores) for score in scores]
    for _ in range(4):
        deficits = [max(0, min_width - width) for width in raw]
        if not any(deficits):
            break
        fixed = sum(deficits)
        flexible = [max(0, width - min_width) for width in raw]
        pool = sum(flexible) or 1
        raw = [
            max(min_width, width) - fixed * flex / pool
            for width, flex in zip(raw, flexible)
        ]
    scale = CONTENT_W / sum(raw)
    return [width * scale for width in raw]


def make_table(rows: list[list[str]]) -> Table:
    width_count = max(len(row) for row in rows)
    normalized = [row + [""] * (width_count - len(row)) for row in rows]
    data = []
    for row_index, row in enumerate(normalized):
        style = ST["TableHeader"] if row_index == 0 else ST["TableBody"]
        data.append([Paragraph(inline_md(cell), style) for cell in row])
    table = Table(
        data,
        colWidths=table_widths(normalized),
        repeatRows=1,
        hAlign="LEFT",
        splitByRow=1,
    )
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5.5),
        ("TOPPADDING", (0, 0), (-1, -1), 5.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5.2),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
    ]
    for row_index in range(1, len(data)):
        background = colors.white if row_index % 2 else HexColor("#F6F8FC")
        commands.append(("BACKGROUND", (0, row_index), (-1, row_index), background))
    table.setStyle(TableStyle(commands))
    return table


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=LEFT,
            rightMargin=RIGHT,
            topMargin=TOP,
            bottomMargin=BOTTOM,
            title="GLM-5.2、Kimi K3、DeepSeek V4 后训练路线图",
            author="OpenAI Codex",
            subject="Post-training technical report comparison and engineering roadmap",
        )
        frame = Frame(
            LEFT,
            BOTTOM,
            CONTENT_W,
            PAGE_H - TOP - BOTTOM,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id="normal",
        )
        self.addPageTemplates(
            [
                PageTemplate(id="report", frames=[frame], onPage=self.draw_page),
            ]
        )
        self.heading_counter = 0

    def beforeDocument(self):
        # multiBuild performs several passes for the TOC; bookmark keys must be stable.
        self.heading_counter = 0

    def draw_page(self, canvas, doc):
        page = canvas.getPageNumber()
        canvas.saveState()
        if page == 1:
            canvas.setFillColor(NAVY)
            canvas.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
            canvas.setFillColor(BLUE)
            canvas.circle(PAGE_W - 32 * mm, PAGE_H - 26 * mm, 31 * mm, stroke=0, fill=1)
            canvas.setFillColor(CYAN)
            canvas.circle(PAGE_W - 7 * mm, PAGE_H - 2 * mm, 18 * mm, stroke=0, fill=1)
            canvas.setFillColor(HexColor("#102650"))
            canvas.rect(0, 0, PAGE_W, 48 * mm, stroke=0, fill=1)
            canvas.setStrokeColor(HexColor("#284778"))
            canvas.setLineWidth(0.5)
            for offset in range(0, 8):
                y = 12 * mm + offset * 5 * mm
                canvas.line(LEFT, y, PAGE_W - RIGHT, y)
        else:
            canvas.setStrokeColor(LINE)
            canvas.setLineWidth(0.55)
            canvas.line(LEFT, PAGE_H - 12 * mm, PAGE_W - RIGHT, PAGE_H - 12 * mm)
            canvas.setFont(FONT_BOLD, 6.8)
            canvas.setFillColor(HexColor("#567098"))
            canvas.drawString(LEFT, PAGE_H - 9.4 * mm, "POST-TRAINING ROADMAP · 2026")
            canvas.setFont(FONT, 6.8)
            canvas.drawRightString(
                PAGE_W - RIGHT,
                PAGE_H - 9.4 * mm,
                "GLM-5.2  /  KIMI K3  /  DEEPSEEK V4",
            )
            canvas.setStrokeColor(LINE)
            canvas.line(LEFT, 10.5 * mm, PAGE_W - RIGHT, 10.5 * mm)
            canvas.setFont(FONT, 7.2)
            canvas.setFillColor(MUTED)
            canvas.drawString(LEFT, 7.2 * mm, "官方证据优先 · [D] / [C] / [R] 分层")
            canvas.drawRightString(PAGE_W - RIGHT, 7.2 * mm, f"{page:02d}")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        style_name = flowable.style.name
        if style_name not in {"H1", "H2", "H3"}:
            return
        level = {"H1": 0, "H2": 1, "H3": 2}[style_name]
        title = strip_markup(flowable.getPlainText())
        self.heading_counter += 1
        key = f"heading-{self.heading_counter}"
        self.canv.bookmarkPage(key)
        if level <= 1:
            try:
                self.canv.addOutlineEntry(title, key, level=level, closed=False)
            except ValueError:
                pass
        self.notify("TOCEntry", (level, title, self.page, key))


def cover_story() -> list:
    evidence_table = Table(
        [
            [
                Paragraph("<b>[D]</b> 官方直接披露", ST["Small"]),
                Paragraph("<b>[C]</b> 透明算术推导", ST["Small"]),
                Paragraph("<b>[R]</b> 工程建议", ST["Small"]),
            ]
        ],
        colWidths=[CONTENT_W / 3] * 3,
        hAlign="LEFT",
    )
    evidence_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), HexColor("#DCE8FF")),
                ("BACKGROUND", (1, 0), (1, 0), HexColor("#E2FAF5")),
                ("BACKGROUND", (2, 0), (2, 0), HexColor("#FFF1C8")),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return [
        Spacer(1, 38 * mm),
        Paragraph("FRONTIER POST-TRAINING SYSTEMS · RESEARCH REPORT 01", ST["CoverKicker"]),
        Paragraph("GLM-5.2、Kimi K3、<br/>DeepSeek V4 后训练路线图", ST["CoverTitle"]),
        Paragraph(
            "Training Report 对照研究、逐节点数据审计、<br/>12–16 周工程 Roadmap 与 4 周教学 Walk-through",
            ST["CoverSubtitle"],
        ),
        HRFlowable(width="100%", thickness=1.2, color=HexColor("#4C6EAA"), spaceAfter=15),
        Paragraph("研究时点：2026-08-03　｜　版本：1.0　｜　中文", ST["CoverMeta"]),
        Spacer(1, 12 * mm),
        evidence_table,
        Spacer(1, 13 * mm),
        Paragraph(
            "核心问题：三家如何设计后训练 pipeline？各 node 使用什么数据、奖励和 rollout 系统？哪些规模已公开，哪些仍然未知？",
            ST["CoverMeta"],
        ),
        PageBreak(),
    ]


def toc_story() -> list:
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOC1",
            fontName=FONT_BOLD,
            fontSize=10.2,
            leading=15,
            textColor=NAVY,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=5,
        ),
        ParagraphStyle(
            "TOC2",
            fontName=FONT,
            fontSize=8.7,
            leading=13,
            textColor=HexColor("#31527C"),
            leftIndent=14,
            firstLineIndent=0,
            spaceBefore=2,
        ),
        ParagraphStyle(
            "TOC3",
            fontName=FONT,
            fontSize=7.6,
            leading=11,
            textColor=MUTED,
            leftIndent=27,
            firstLineIndent=0,
            spaceBefore=1,
        ),
    ]
    return [
        Paragraph("目录", ST["TOCTitle"]),
        Paragraph(
            "本报告按“证据边界 → 单模型 pipeline → 横向比较 → 数据审计 → Roadmap → 教学计划”组织。",
            ST["Body"],
        ),
        Spacer(1, 5),
        toc,
        PageBreak(),
    ]


def flush_paragraph(buffer: list[str], story: list) -> None:
    if not buffer:
        return
    value = " ".join(part.strip() for part in buffer).strip()
    if value:
        story.append(Paragraph(inline_md(value), ST["Body"]))
    buffer.clear()


def parse_markdown(text: str) -> list:
    lines = text.splitlines()
    first_rule = next(index for index, line in enumerate(lines) if line.strip() == "---")
    lines = lines[first_rule + 1 :]
    story: list = []
    paragraph_buffer: list[str] = []
    index = 0
    first_h1_seen = False

    while index < len(lines):
        raw = lines[index]
        line = raw.rstrip()

        if not line.strip():
            flush_paragraph(paragraph_buffer, story)
            index += 1
            continue

        if line.strip() == "---":
            flush_paragraph(paragraph_buffer, story)
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.7,
                    color=LINE,
                    spaceBefore=7,
                    spaceAfter=10,
                )
            )
            index += 1
            continue

        if line.startswith("```"):
            flush_paragraph(paragraph_buffer, story)
            code_lines = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code_lines.append(lines[index].rstrip())
                index += 1
            index += 1
            code = html.escape("\n".join(code_lines), quote=False)
            code_block = XPreformatted(code, ST["Code"])
            box = Table([[code_block]], colWidths=[CONTENT_W], hAlign="LEFT")
            box.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), HexColor("#F1F5FA")),
                        ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#C7D4E5")),
                        ("LINEBEFORE", (0, 0), (0, -1), 3, CYAN),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                )
            )
            story.extend([box, Spacer(1, 7)])
            continue

        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            flush_paragraph(paragraph_buffer, story)
            hashes, title = heading.groups()
            level = len(hashes)
            if title == "Executive Summary":
                style_name = "H1"
            elif level == 1:
                style_name = "H1"
            elif level == 2:
                style_name = "H2"
            else:
                style_name = "H3"
            if style_name == "H1":
                if first_h1_seen:
                    story.append(PageBreak())
                first_h1_seen = True
                story.append(
                    HRFlowable(
                        width=32 * mm,
                        thickness=3,
                        color=CYAN,
                        hAlign="LEFT",
                        spaceAfter=8,
                    )
                )
            story.append(Paragraph(inline_md(title), ST[style_name]))
            index += 1
            continue

        if line.startswith("> "):
            flush_paragraph(paragraph_buffer, story)
            quote_lines = []
            while index < len(lines) and lines[index].startswith("> "):
                quote_lines.append(lines[index][2:].strip())
                index += 1
            quote = Paragraph(inline_md(" ".join(quote_lines)), ST["Quote"])
            callout = Table([[quote]], colWidths=[CONTENT_W], hAlign="LEFT")
            callout.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), PALE_CYAN),
                        ("LINEBEFORE", (0, 0), (0, -1), 3, CYAN),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, -1), 9),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                    ]
                )
            )
            story.extend([callout, Spacer(1, 6)])
            continue

        if line.startswith("|"):
            flush_paragraph(paragraph_buffer, story)
            raw_rows = []
            while index < len(lines) and lines[index].startswith("|"):
                raw_rows.append(lines[index].strip())
                index += 1
            rows: list[list[str]] = []
            for row_index, row in enumerate(raw_rows):
                cells = [cell.strip() for cell in row.strip("|").split("|")]
                if row_index == 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    continue
                rows.append(cells)
            if rows:
                story.extend([make_table(rows), Spacer(1, 8)])
            continue

        bullet_match = re.match(r"^\s*[-*]\s+(.+)$", line)
        number_match = re.match(r"^\s*(\d+)[.)]\s+(.+)$", line)
        if bullet_match or number_match:
            flush_paragraph(paragraph_buffer, story)
            ordered = bool(number_match)
            start_number = number_match.group(1) if number_match else None
            items = []
            while index < len(lines):
                candidate = lines[index].rstrip()
                match = (
                    re.match(r"^\s*(\d+)[.)]\s+(.+)$", candidate)
                    if ordered
                    else re.match(r"^\s*[-*]\s+(.+)$", candidate)
                )
                if not match:
                    break
                item_text = match.group(2) if ordered else match.group(1)
                items.append(
                    ListItem(
                        Paragraph(inline_md(item_text), ST["List"]),
                        leftIndent=12,
                    )
                )
                index += 1
                while index < len(lines) and not lines[index].strip():
                    index += 1
            list_kwargs = dict(
                bulletType="1" if ordered else "bullet",
                leftIndent=18,
                bulletFontName=FONT_BOLD,
                bulletFontSize=8.5,
                bulletColor=BLUE,
                spaceAfter=7,
            )
            if ordered:
                list_kwargs["start"] = start_number
            story.append(ListFlowable(items, **list_kwargs))
            continue

        paragraph_buffer.append(line)
        index += 1

    flush_paragraph(paragraph_buffer, story)
    return story


def build() -> Path:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    story = cover_story() + toc_story() + parse_markdown(markdown)
    doc = ReportDocTemplate(str(OUTPUT_PATH))
    doc.multiBuild(story)
    return OUTPUT_PATH


if __name__ == "__main__":
    output = build()
    print(output)

"""
Markdown raporu -> PDF (reportlab + DejaVu font, Türkçe tam destek).
Kullanım: python md_to_pdf.py RAPOR_SUNUM.md RAPOR_SUNUM.pdf
"""
import sys, os, re
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Preformatted, HRFlowable)

SRC = sys.argv[1] if len(sys.argv) > 1 else "RAPOR_SUNUM.md"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(SRC)[0] + ".pdf"

# --- Fontlar (Türkçe karakterler için DejaVu) ---
FD = "/usr/share/fonts/truetype/dejavu"
pdfmetrics.registerFont(TTFont("DV", f"{FD}/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", f"{FD}/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-M", f"{FD}/DejaVuSansMono.ttf"))

# Emoji -> DejaVu'nun desteklediği sade simgeler
EMOJI = {"✅": "[✓]", "❌": "[✗]", "⚠️": "[!]", "⚠": "[!]", "👉": "→",
         "🎯": "*", "🤖": "", "✔️": "✓"}
def deemoji(s):
    for k, v in EMOJI.items():
        s = s.replace(k, v)
    return s

def inline(s):
    """**bold** -> <b>, özel karakterleri escape et."""
    s = deemoji(s)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r'<font face="DV-M">\1</font>', s)
    return s

styles = getSampleStyleSheet()
def mk(name, **kw):
    kw.setdefault("fontName", "DV")
    return ParagraphStyle(name, **kw)

body = mk("body", fontSize=10, leading=15, spaceAfter=6, alignment=TA_LEFT)
h1 = mk("h1", fontName="DV-B", fontSize=18, leading=22, spaceBefore=6, spaceAfter=10, textColor=colors.HexColor("#1a3c5e"))
h2 = mk("h2", fontName="DV-B", fontSize=14, leading=18, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#21618c"))
h3 = mk("h3", fontName="DV-B", fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#2874a6"))
quote = mk("quote", fontSize=10, leading=14, leftIndent=14, textColor=colors.HexColor("#555555"), spaceAfter=6)
cell = mk("cell", fontSize=8.5, leading=11)
cellb = mk("cellb", fontName="DV-B", fontSize=8.5, leading=11, textColor=colors.white)
bullet = mk("bullet", fontSize=10, leading=14, leftIndent=14, bulletIndent=4, spaceAfter=2)

story = []
lines = open(SRC, encoding="utf-8").read().splitlines()
i = 0
def flush_table(rows):
    if not rows:
        return
    # ilk satır başlık
    data = []
    for r_idx, row in enumerate(rows):
        sty = cellb if r_idx == 0 else cell
        data.append([Paragraph(inline(c), sty) for c in row])
    ncol = max(len(r) for r in data)
    for r in data:
        while len(r) < ncol:
            r.append(Paragraph("", cell))
    avail = A4[0] - 4 * cm
    t = Table(data, colWidths=[avail / ncol] * ncol, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#21618c")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#aab7c4")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

while i < len(lines):
    line = lines[i]
    s = line.strip()

    # Kod bloğu
    if s.startswith("```"):
        block = []
        i += 1
        while i < len(lines) and not lines[i].strip().startswith("```"):
            block.append(deemoji(lines[i]))
            i += 1
        i += 1
        pre = Preformatted("\n".join(block), mk("code", fontName="DV-M", fontSize=7.5, leading=9.5),
                           )
        story.append(Table([[pre]], colWidths=[A4[0] - 4 * cm],
                           style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f6f8")),
                                             ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#ccd4dc")),
                                             ("LEFTPADDING", (0, 0), (-1, -1), 6),
                                             ("TOPPADDING", (0, 0), (-1, -1), 5),
                                             ("BOTTOMPADDING", (0, 0), (-1, -1), 5)])))
        story.append(Spacer(1, 8))
        continue

    # Tablo
    if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\|?[\s:|-]+\|?$", lines[i + 1].strip()):
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            r = lines[i].strip()
            if re.match(r"^\|?[\s:|-]+\|?$", r):  # ayraç satırı
                i += 1
                continue
            cells = [c.strip() for c in r.strip("|").split("|")]
            rows.append(cells)
            i += 1
        flush_table(rows)
        continue

    if not s:
        i += 1
        continue
    if s.startswith("# "):
        story.append(Paragraph(inline(s[2:]), h1))
    elif s.startswith("## "):
        story.append(Paragraph(inline(s[3:]), h2))
    elif s.startswith("### "):
        story.append(Paragraph(inline(s[4:]), h3))
    elif s.startswith("> "):
        story.append(Paragraph(inline(s[2:]), quote))
    elif s in ("---", "***", "___"):
        story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#ccd4dc"), spaceBefore=4, spaceAfter=8))
    elif re.match(r"^[-*]\s+", s):
        story.append(Paragraph(inline(re.sub(r"^[-*]\s+", "", s)), bullet, bulletText="•"))
    elif re.match(r"^\d+\.\s+", s):
        story.append(Paragraph(inline(s), bullet))
    else:
        story.append(Paragraph(inline(s), body))
    i += 1

doc = SimpleDocTemplate(OUT, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm,
                        leftMargin=2*cm, rightMargin=2*cm, title="AI Görüntü Analizi - Rapor")
doc.build(story)
print(f"PDF olusturuldu: {OUT}")

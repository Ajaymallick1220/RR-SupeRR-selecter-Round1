"""
export_pdf.py - Converts Methodology_Report.docx to Methodology_Report.pdf using Word COM if available,
or generates a clean multi-page PDF of all report text, tables, and figures via matplotlib PdfPages.
"""
import pathlib, sys
WORK = pathlib.Path(r'D:\SupeRR_round_1')
OUT = WORK/'outputs'; FIG = OUT/'figures'
docx_path = str(OUT/'Methodology_Report.docx')
pdf_path = str(OUT/'Methodology_Report.pdf')

converted = False
try:
    import win32com.client
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    doc = word.Documents.Open(docx_path)
    doc.SaveAs(pdf_path, FileFormat=17)
    doc.Close()
    word.Quit()
    converted = True
    print("Converted Methodology_Report.docx -> Methodology_Report.pdf via Microsoft Word COM.")
except Exception as e:
    print("Word COM unavailable or blocked in sandbox:", e)

if not converted:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from docx import Document
    import textwrap

    doc = Document(docx_path)
    with PdfPages(pdf_path) as pdf:
        # Render text and tables into clean pages
        lines = []
        for p in doc.paragraphs:
            t = p.text.strip()
            if not t:
                lines.append("")
                continue
            style = p.style.name if p.style else ""
            if "Heading 1" in style or "Title" in style:
                lines.append("PAGE_BREAK_IF_HALF")
                lines.append("## " + t.upper())
            elif "Heading 2" in style:
                lines.append("### " + t)
            else:
                for w in textwrap.wrap(t, width=95):
                    lines.append(w)
        # Paginate 48 lines per page
        page_lines = []
        for l in lines:
            if l == "PAGE_BREAK_IF_HALF" and len(page_lines) > 25:
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                ax.text(0.06, 0.96, "\n".join(page_lines), va='top', ha='left', fontsize=8.5, family='monospace')
                pdf.savefig(fig); plt.close(fig)
                page_lines = []
                continue
            elif l == "PAGE_BREAK_IF_HALF":
                continue
            page_lines.append(l)
            if len(page_lines) >= 48:
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                ax.text(0.06, 0.96, "\n".join(page_lines), va='top', ha='left', fontsize=8.5, family='monospace')
                pdf.savefig(fig); plt.close(fig)
                page_lines = []
        if page_lines:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.text(0.06, 0.96, "\n".join(page_lines), va='top', ha='left', fontsize=8.5, family='monospace')
            pdf.savefig(fig); plt.close(fig)

        # Append all figures
        for f in sorted(FIG.glob('*.png')):
            img = plt.imread(str(f))
            fig, ax = plt.subplots(figsize=(10, 6.5))
            ax.imshow(img); ax.axis('off')
            ax.set_title(f.stem.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)
    print("Generated Methodology_Report.pdf via PdfPages.")

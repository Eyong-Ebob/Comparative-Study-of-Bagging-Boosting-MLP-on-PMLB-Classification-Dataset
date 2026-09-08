"""
Generate a comprehensive Word document describing the LR, RF, XGB evaluation pipeline.
"""
import os
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import warnings
warnings.filterwarnings('ignore')

BASE = r"c:\Users\EYONGEBOB\Desktop\PMLB Project\sklearn-benchmarks"
OUT  = os.path.join(BASE, "LR_RF_XGB_Evaluation_Report.docx")
FIG_DIR = os.path.join(BASE, "_report_figs")
os.makedirs(FIG_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────

def save_fig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return path

def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def add_hyperlink(paragraph, text):
    run = paragraph.add_run(text)
    run.font.color.rgb = RGBColor(0x1F, 0x5C, 0x9E)
    return run

def add_heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return h

def add_para(doc, text, bold=False, italic=False, size=11, space_before=0, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after  = Pt(space_after)
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold   = bold
    run.italic = italic
    return p

def add_code(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after  = Pt(3)
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), 'F2F2F2')
    pPr.append(shd)
    run = p.add_run(text)
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x8C)
    return p

def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.left_indent = Cm(0.5 + level * 0.5)
    run = p.add_run(text)
    run.font.size = Pt(10.5)
    return p

def add_image(doc, path, width=6.0, caption=None):
    doc.add_picture(path, width=Inches(width))
    last = doc.paragraphs[-1]
    last.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        cp = doc.add_paragraph(caption)
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.runs[0].font.size = Pt(9)
        cp.runs[0].italic = True
        cp.runs[0].font.color.rgb = RGBColor(0x55, 0x55, 0x55)

def add_table(doc, headers, rows, header_color='1F5C9E', col_widths=None):
    table = doc.add_table(rows=1+len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header row
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        set_cell_bg(hdr[i], header_color)
        for run in hdr[i].paragraphs[0].runs:
            run.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            run.font.size = Pt(10)
        hdr[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    # Data rows
    for ri, row in enumerate(rows):
        cells = table.rows[ri+1].cells
        bg = 'FFFFFF' if ri % 2 == 0 else 'EDF2F9'
        for ci, val in enumerate(row):
            cells[ci].text = str(val)
            set_cell_bg(cells[ci], bg)
            cells[ci].paragraphs[0].runs[0].font.size = Pt(9.5)
            cells[ci].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if col_widths:
        for row in table.rows:
            for ci, w in enumerate(col_widths):
                row.cells[ci].width = Cm(w)
    return table

def add_formula_image(doc, formula_latex, width=4.5, caption=None):
    """Render a LaTeX formula as a PNG and embed it."""
    fig, ax = plt.subplots(figsize=(width, 0.6))
    ax.axis('off')
    ax.text(0.5, 0.5, f'${formula_latex}$',
            fontsize=16, ha='center', va='center',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FAFAFA', edgecolor='#CCCCCC'))
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    buf.seek(0)
    doc.add_picture(buf, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        cp = doc.add_paragraph(caption)
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.runs[0].font.size = Pt(9)
        cp.runs[0].italic = True

def add_separator(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), 'AAAAAA')
    pBdr.append(bottom)
    pPr.append(pBdr)

# ──────────────────────────────────────────────
# FIGURE 1: Pipeline Architecture
# ──────────────────────────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.5)
    ax.axis('off')
    ax.set_facecolor('white')

    steps = [
        (0.3,  'Raw .tsv.gz\nDataset',       '#D6EAF8', '#2E86C1'),
        (2.5,  'Load &\nSeparate X, y',       '#D5F5E3', '#1E8449'),
        (4.7,  'RobustScaler\n(per fold)',     '#FCF3CF', '#D4AC0D'),
        (6.9,  'Classifier\n(with params)',    '#FADBD8', '#C0392B'),
        (9.1,  '10-Fold CV\nPredictions',      '#E8DAEF', '#7D3C98'),
        (11.0, 'Metrics\nOutput',              '#D1F2EB', '#148F77'),
    ]

    for x, label, fc, ec in steps:
        box = FancyBboxPatch((x, 0.8), 1.8, 1.8,
                             boxstyle="round,pad=0.1",
                             linewidth=2, edgecolor=ec, facecolor=fc)
        ax.add_patch(box)
        ax.text(x + 0.9, 1.7, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color='#1A1A1A', linespacing=1.4)

    for i in range(len(steps)-1):
        x_start = steps[i][0] + 1.85
        x_end   = steps[i+1][0] - 0.05
        ax.annotate('', xy=(x_end, 1.7), xytext=(x_start, 1.7),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=2))

    ax.text(6.0, 3.2, 'Evaluation Pipeline — Per Parameter Combination × Dataset',
            ha='center', va='center', fontsize=12, fontweight='bold', color='#1A1A1A')

    # Wrap annotation
    ax.text(6.9 + 0.9, 0.5,
            'Refitted inside\neach fold', ha='center', fontsize=7.5,
            color='#C0392B', style='italic')
    ax.annotate('', xy=(6.9+0.9, 0.75), xytext=(6.9+0.9, 0.52),
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.2))

    plt.tight_layout()
    return save_fig('fig1_pipeline.png')

# ──────────────────────────────────────────────
# FIGURE 2: 10-Fold Stratified CV Diagram
# ──────────────────────────────────────────────
def fig_cv():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 11.5)
    ax.axis('off')
    ax.set_facecolor('white')

    ax.text(5, 11, '10-Fold Stratified Cross-Validation\n(StratifiedKFold, n_splits=10, shuffle=True, random_state=90483257)',
            ha='center', va='top', fontsize=11, fontweight='bold')

    train_color = '#2E86C1'
    test_color  = '#E74C3C'

    for fold in range(10):
        y = 9.5 - fold * 0.9
        for col in range(10):
            if col == fold:
                fc = test_color
                label = 'Test' if col == 0 else ''
            else:
                fc = train_color
                label = 'Train' if col == 0 else ''
            rect = plt.Rectangle((col * 1.0, y), 0.95, 0.7,
                                  linewidth=0.8, edgecolor='white', facecolor=fc)
            ax.add_patch(rect)

        ax.text(-0.45, y + 0.35, f'Fold {fold+1}', ha='right', va='center',
                fontsize=8.5, color='#333333')
        ax.text(10.15, y + 0.35, f'≈{10}% test', ha='left', va='center',
                fontsize=7.5, color='#777777')

    # Column headers
    for col in range(10):
        ax.text(col * 1.0 + 0.475, 10.1, f'Block\n{col+1}',
                ha='center', va='bottom', fontsize=7, color='#444444')

    # Legend
    leg_x = 1.0
    leg_y = -0.2
    ax.add_patch(plt.Rectangle((leg_x, leg_y), 0.4, 0.3, facecolor=train_color, edgecolor='white'))
    ax.text(leg_x + 0.5, leg_y + 0.15, 'Training fold (90% of data)', va='center', fontsize=9)
    ax.add_patch(plt.Rectangle((leg_x + 4, leg_y), 0.4, 0.3, facecolor=test_color, edgecolor='white'))
    ax.text(leg_x + 4.5, leg_y + 0.15, 'Test fold (10% of data)', va='center', fontsize=9)

    plt.tight_layout()
    return save_fig('fig2_cv.png')

# ──────────────────────────────────────────────
# FIGURE 3: Parameter combo counts per classifier
# ──────────────────────────────────────────────
def fig_param_counts():
    classifiers = ['Logistic\nRegression', 'Random\nForest', 'XGBoost']
    total_rows   = [39145,  126602, 30492 * 99]   # approx total evaluations
    per_dataset  = [237.2,  767.3,  30492]
    combos       = [240,    770,    30492]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ['#2E86C1', '#1E8449', '#C0392B']

    # Left: per-dataset combos
    bars = axes[0].bar(classifiers, combos, color=colors, width=0.5, edgecolor='white', linewidth=1.2)
    axes[0].set_ylabel('Parameter Combinations per Dataset', fontsize=10)
    axes[0].set_title('Search Space Size per Dataset', fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, 35000)
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.5)
    axes[0].set_axisbelow(True)
    for bar, val in zip(bars, combos):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                     f'{val:,}', ha='center', fontsize=10, fontweight='bold')

    # Annotate search type
    types = ['Random\nSearch', 'Random\nSearch', 'Grid\nSearch\n(exhaustive)']
    for i, (bar, t) in enumerate(zip(bars, types)):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                     t, ha='center', va='center', fontsize=8.5,
                     color='white', fontweight='bold')

    # Right: dataset coverage
    covered  = [128, 125, 99]
    skipped  = [2,   5,  31]
    x = np.arange(3)
    b1 = axes[1].bar(x, covered, color=colors, width=0.5, label='Results obtained')
    b2 = axes[1].bar(x, skipped, bottom=covered, color=['#AED6F1','#A9DFBF','#F1948A'],
                     width=0.5, label='Skipped / No result')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classifiers)
    axes[1].set_ylabel('Number of Datasets', fontsize=10)
    axes[1].set_title('Dataset Coverage (out of 130)', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.5)
    axes[1].set_axisbelow(True)
    for bar, val in zip(b1, covered):
        axes[1].text(bar.get_x() + bar.get_width()/2, val/2,
                     str(val), ha='center', va='center', fontsize=10,
                     color='white', fontweight='bold')

    plt.tight_layout(pad=2)
    return save_fig('fig3_param_counts.png')

# ──────────────────────────────────────────────
# FIGURE 4: Score distributions per classifier
# ──────────────────────────────────────────────
def fig_score_dist():
    mf_path = os.path.join(BASE, 'dataset_manifest.csv')
    mf = pd.read_csv(mf_path)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    colors = ['#2E86C1', '#1E8449', '#C0392B']

    configs = [
        ('logreg_balanced_accuracy', 'logreg_accuracy', 'Logistic Regression', colors[0]),
        ('rf_balanced_accuracy',     'rf_accuracy',     'Random Forest',        colors[1]),
        ('xgb_balanced_accuracy',    'xgb_accuracy',    'XGBoost',              colors[2]),
    ]

    for ax, (col_ba, col_acc, title, color) in zip(axes, configs):
        data_ba  = mf[col_ba].dropna()
        data_acc = mf[col_acc].dropna()
        bplot = ax.boxplot([data_ba, data_acc],
                           labels=['Balanced\nAccuracy', 'Accuracy'],
                           patch_artist=True,
                           medianprops=dict(color='black', linewidth=2),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5))
        for patch in bplot['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

        mean_ba  = data_ba.mean()
        mean_acc = data_acc.mean()
        ax.text(1, 0.05, f'mean={mean_ba:.3f}', ha='center', fontsize=8, color='#333333')
        ax.text(2, 0.05, f'mean={mean_acc:.3f}', ha='center', fontsize=8, color='#333333')

    plt.suptitle('Score Distributions Across All Datasets (Best Result per Dataset)',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    return save_fig('fig4_scores.png')

# ──────────────────────────────────────────────
# FIGURE 5: LR hyperparameter space
# ──────────────────────────────────────────────
def fig_lr_params():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # C distribution (random uniform)
    c_vals = np.random.uniform(1e-10, 10, 500)
    axes[0].hist(c_vals, bins=30, color='#2E86C1', edgecolor='white', alpha=0.85)
    axes[0].set_title('C — Uniform(1e-10, 10)', fontsize=10, fontweight='bold')
    axes[0].set_xlabel('C value')
    axes[0].set_ylabel('Frequency')
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.4)
    axes[0].set_axisbelow(True)

    # Penalty
    axes[1].bar(['l1', 'l2'], [0.5, 0.5], color=['#2E86C1', '#85C1E9'],
                edgecolor='white', width=0.4)
    axes[1].set_title('penalty — Uniform choice', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Probability')
    axes[1].set_ylim(0, 0.7)
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.4)
    axes[1].set_axisbelow(True)

    # fit_intercept / dual
    cats = ['fit_intercept=True', 'fit_intercept=False', 'dual=True\n(l2 only)', 'dual=False']
    vals = [0.5, 0.5, 0.25, 0.75]
    colors = ['#2E86C1','#85C1E9','#1E8449','#82E0AA']
    axes[2].bar(cats, vals, color=colors, edgecolor='white', width=0.6)
    axes[2].set_title('Boolean Parameters', fontsize=10, fontweight='bold')
    axes[2].set_ylabel('Probability')
    axes[2].set_ylim(0, 0.9)
    axes[2].tick_params(axis='x', labelsize=7.5)
    axes[2].yaxis.grid(True, linestyle='--', alpha=0.4)
    axes[2].set_axisbelow(True)

    plt.suptitle('Logistic Regression — Random Search Parameter Distributions',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    return save_fig('fig5_lr_params.png')

# ──────────────────────────────────────────────
# FIGURE 6: RF hyperparameter space
# ──────────────────────────────────────────────
def fig_rf_params():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # n_estimators
    ne_vals = np.random.choice(range(50, 1001, 50), 500)
    axes[0].hist(ne_vals, bins=20, color='#1E8449', edgecolor='white', alpha=0.85)
    axes[0].set_title('n_estimators\nchoice(50–1000, step 50)', fontsize=10, fontweight='bold')
    axes[0].set_xlabel('n_estimators')
    axes[0].set_ylabel('Frequency')
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.4)
    axes[0].set_axisbelow(True)

    # max_depth
    md_vals = np.random.choice(list(range(1, 51)) + [None], 500)
    md_int  = [m if m is not None else 55 for m in md_vals]
    axes[1].hist(md_int, bins=25, color='#1E8449', edgecolor='white', alpha=0.85)
    axes[1].set_title('max_depth\nchoice(1–50 + None)', fontsize=10, fontweight='bold')
    axes[1].set_xlabel('max_depth (55 = None/unlimited)')
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.4)
    axes[1].set_axisbelow(True)

    # min_impurity_decrease
    mid_vals = np.random.exponential(scale=0.01, size=500)
    axes[2].hist(mid_vals, bins=30, color='#1E8449', edgecolor='white', alpha=0.85)
    axes[2].set_title('min_impurity_decrease\nExponential(scale=0.01)', fontsize=10, fontweight='bold')
    axes[2].set_xlabel('min_impurity_decrease')
    axes[2].yaxis.grid(True, linestyle='--', alpha=0.4)
    axes[2].set_axisbelow(True)

    plt.suptitle('Random Forest — Random Search Parameter Distributions',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    return save_fig('fig6_rf_params.png')

# ──────────────────────────────────────────────
# FIGURE 7: XGB grid search heatmap
# ──────────────────────────────────────────────
def fig_xgb_grid():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # n_estimators x learning_rate grid
    ne_vals = [10, 50, 100, 500]
    lr_vals = [0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0]
    Z = np.array([[ne * lr for lr in lr_vals] for ne in ne_vals])
    im = axes[0].imshow(np.log1p(Z), aspect='auto', cmap='YlOrRd')
    axes[0].set_xticks(range(len(lr_vals)))
    axes[0].set_xticklabels([str(v) for v in lr_vals], fontsize=8)
    axes[0].set_yticks(range(len(ne_vals)))
    axes[0].set_yticklabels([str(v) for v in ne_vals])
    axes[0].set_xlabel('learning_rate')
    axes[0].set_ylabel('n_estimators')
    axes[0].set_title('Grid Coverage:\nn_estimators × learning_rate\n(color = log(n_est × lr))', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=axes[0], label='log(n_est × lr)')

    # subsample x gamma grid
    sub_vals = np.round(np.arange(0.0, 1.01, 0.1), 2)
    gam_vals = np.round(np.arange(0.0, 0.51, 0.05), 2)
    Z2 = np.ones((len(gam_vals), len(sub_vals)))
    im2 = axes[1].imshow(Z2, aspect='auto', cmap='Blues', vmin=0, vmax=2)
    axes[1].set_xticks(range(len(sub_vals)))
    axes[1].set_xticklabels([str(v) for v in sub_vals], fontsize=7, rotation=45)
    axes[1].set_yticks(range(len(gam_vals)))
    axes[1].set_yticklabels([str(v) for v in gam_vals], fontsize=7)
    axes[1].set_xlabel('subsample')
    axes[1].set_ylabel('gamma')
    axes[1].set_title(f'Grid Coverage:\ngamma ({len(gam_vals)} values) × subsample ({len(sub_vals)} values)\nEvery cell evaluated', fontsize=10, fontweight='bold')
    for i in range(len(gam_vals)):
        for j in range(len(sub_vals)):
            axes[1].text(j, i, '✓', ha='center', va='center', fontsize=5, color='#2E86C1')

    plt.suptitle('XGBoost — Full Grid Search Parameter Space', fontsize=11, fontweight='bold')
    plt.tight_layout()
    return save_fig('fig7_xgb_grid.png')

# ──────────────────────────────────────────────
# FIGURE 8: Balanced accuracy formula diagram
# ──────────────────────────────────────────────
def fig_metrics():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    title_box = FancyBboxPatch((0.5, 5.2), 11, 0.65,
                               boxstyle='round,pad=0.1',
                               facecolor='#1F5C9E', edgecolor='#1F5C9E')
    ax.add_patch(title_box)
    ax.text(6, 5.525, 'Metric Formulas Used in the Evaluation',
            ha='center', va='center', fontsize=13, fontweight='bold', color='white')

    # Accuracy
    box1 = FancyBboxPatch((0.3, 3.8), 3.5, 1.2, boxstyle='round,pad=0.15',
                          facecolor='#EBF5FB', edgecolor='#2E86C1', linewidth=2)
    ax.add_patch(box1)
    ax.text(2.05, 4.9, 'Accuracy', ha='center', fontsize=11, fontweight='bold', color='#2E86C1')
    ax.text(2.05, 4.45,
            r'$\frac{\text{Correct Predictions}}{\text{Total Samples}}$',
            ha='center', va='center', fontsize=14, color='#1A1A1A')
    ax.text(2.05, 3.92, 'Counts ALL samples equally', ha='center', fontsize=8.5,
            color='#555555', style='italic')

    # Macro F1
    box2 = FancyBboxPatch((4.2, 3.8), 3.5, 1.2, boxstyle='round,pad=0.15',
                          facecolor='#EAFAF1', edgecolor='#1E8449', linewidth=2)
    ax.add_patch(box2)
    ax.text(5.95, 4.9, 'Macro F1', ha='center', fontsize=11, fontweight='bold', color='#1E8449')
    ax.text(5.95, 4.45,
            r'$\frac{1}{|C|} \sum_{c} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$',
            ha='center', va='center', fontsize=13, color='#1A1A1A')
    ax.text(5.95, 3.92, 'Unweighted mean of per-class F1',
            ha='center', fontsize=8.5, color='#555555', style='italic')

    # Balanced Accuracy
    box3 = FancyBboxPatch((8.1, 3.8), 3.5, 1.2, boxstyle='round,pad=0.15',
                          facecolor='#FDF2F8', edgecolor='#7D3C98', linewidth=2)
    ax.add_patch(box3)
    ax.text(9.85, 4.9, 'Balanced Accuracy (TPOT)',
            ha='center', fontsize=10, fontweight='bold', color='#7D3C98')
    ax.text(9.85, 4.45,
            r'$\frac{1}{|C|} \sum_{c} \frac{Sens_c + Spec_c}{2}$',
            ha='center', va='center', fontsize=13, color='#1A1A1A')
    ax.text(9.85, 3.92, 'Sensitivity+Specificity per class',
            ha='center', fontsize=8.5, color='#555555', style='italic')

    # Expanded Balanced Accuracy
    box4 = FancyBboxPatch((0.3, 0.4), 11.3, 3.1, boxstyle='round,pad=0.15',
                          facecolor='#F9F9F9', edgecolor='#7D3C98', linewidth=1.5,
                          linestyle='--')
    ax.add_patch(box4)
    ax.text(5.95, 3.35, 'TPOT Balanced Accuracy — Expanded (per class c, one-vs-rest)',
            ha='center', fontsize=10, fontweight='bold', color='#7D3C98')

    ax.text(2.8, 2.75,
            r'$Sensitivity_c = \frac{TP_c}{TP_c + FN_c}$',
            ha='center', va='center', fontsize=13)
    ax.text(2.8, 2.1, '(Recall — how many of class c\nwere correctly predicted)',
            ha='center', va='center', fontsize=8.5, color='#555555', style='italic')

    ax.text(6.0, 2.75,
            r'$Specificity_c = \frac{TN_c}{TN_c + FP_c}$',
            ha='center', va='center', fontsize=13)
    ax.text(6.0, 2.1, '(True Negative Rate — how many\nnon-class-c were correctly rejected)',
            ha='center', va='center', fontsize=8.5, color='#555555', style='italic')

    ax.text(9.2, 2.75,
            r'$ClassAcc_c = \frac{Sens_c + Spec_c}{2}$',
            ha='center', va='center', fontsize=13)

    ax.text(5.95, 0.9,
            r'$BalancedAccuracy = \frac{1}{|C|} \sum_{c \in C} ClassAcc_c$'
            '\n'
            r'Range: 0.5 (random chance)  →  1.0 (perfect)',
            ha='center', va='center', fontsize=12, color='#5B2C6F')

    plt.tight_layout()
    return save_fig('fig8_metrics.png')

# ──────────────────────────────────────────────
# FIGURE 9: Best result extraction
# ──────────────────────────────────────────────
def fig_extraction():
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.5)
    ax.axis('off')

    steps = [
        (0.3,  'Raw Benchmark\nFile\n(.tsv.gz)',      '#D6EAF8', '#2E86C1',
         '39K rows (LR)\n127K rows (RF)\n~3M rows (XGB)'),
        (2.7,  'Group by\n(dataset,\nclassifier)',     '#FCF3CF', '#D4AC0D',
         'pandas groupby'),
        (5.1,  'Take MAX\naccuracy\nper group',        '#D5F5E3', '#1E8449',
         '.groupby(...)["accuracy"].max()'),
        (7.5,  'Merge LR +\nRF + XGB\nresults',       '#E8DAEF', '#7D3C98',
         'join on dataset name'),
        (9.5,  'dataset_\nmanifest.csv\n(130 rows)',   '#FADBD8', '#C0392B',
         '+ metadata'),
    ]

    for x, label, fc, ec, note in steps:
        box = FancyBboxPatch((x, 0.9), 1.8, 1.6,
                             boxstyle='round,pad=0.1',
                             linewidth=2, edgecolor=ec, facecolor=fc)
        ax.add_patch(box)
        ax.text(x + 0.9, 1.7, label, ha='center', va='center',
                fontsize=8.5, fontweight='bold', linespacing=1.4)
        ax.text(x + 0.9, 0.7, note, ha='center', va='center',
                fontsize=7, color='#666666', style='italic')

    for i in range(len(steps)-1):
        x_start = steps[i][0] + 1.85
        x_end   = steps[i+1][0] - 0.05
        ax.annotate('', xy=(x_end, 1.7), xytext=(x_start, 1.7),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=2))

    ax.text(5.5, 3.2, 'Best Result Extraction Process', ha='center',
            fontsize=12, fontweight='bold')
    plt.tight_layout()
    return save_fig('fig9_extraction.png')


# ──────────────────────────────────────────────
# FIGURE 10: Dataset size vs score scatter
# ──────────────────────────────────────────────
def fig_scatter():
    mf_path = os.path.join(BASE, 'dataset_manifest.csv')
    mf = pd.read_csv(mf_path)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    configs = [
        ('logreg_balanced_accuracy', 'Logistic Regression', '#2E86C1'),
        ('rf_balanced_accuracy',     'Random Forest',       '#1E8449'),
        ('xgb_balanced_accuracy',    'XGBoost',             '#C0392B'),
    ]

    for ax, (col, title, color) in zip(axes, configs):
        sub = mf[mf[col].notna()].copy()
        ax.scatter(np.log10(sub['rows'].clip(lower=1)),
                   sub[col], alpha=0.6, color=color, s=40, edgecolors='white', linewidth=0.5)
        ax.set_xlabel('log10(Dataset Size in Rows)', fontsize=9)
        ax.set_ylabel('Balanced Accuracy', fontsize=9)
        ax.set_title(f'{title}\n(n={len(sub)})', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        # trend line
        x = np.log10(sub['rows'].clip(lower=1))
        z = np.polyfit(x, sub[col], 1)
        p = np.poly1d(z)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, p(xline), '--', color='#333333', lw=1.5, alpha=0.7)

    plt.suptitle('Dataset Size vs Balanced Accuracy Score (Best Result per Dataset)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    return save_fig('fig10_scatter.png')


# ──────────────────────────────────────────────
# BUILD THE DOCUMENT
# ──────────────────────────────────────────────
def build_doc():
    print("Generating figures...")
    f1  = fig_pipeline()
    f2  = fig_cv()
    f3  = fig_param_counts()
    f4  = fig_score_dist()
    f5  = fig_lr_params()
    f6  = fig_rf_params()
    f7  = fig_xgb_grid()
    f8  = fig_metrics()
    f9  = fig_extraction()
    f10 = fig_scatter()
    print("All figures saved. Building Word document...")

    doc = Document()

    # ── Page margins
    for section in doc.sections:
        section.top_margin    = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin   = Cm(2.5)
        section.right_margin  = Cm(2.5)

    # ─────────────────────────────────────────────────
    # TITLE PAGE
    # ─────────────────────────────────────────────────
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(60)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run('LR, RF & XGB Evaluation Pipeline')
    r.font.size = Pt(26)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x1F, 0x5C, 0x9E)

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = p2.add_run('Complete Technical Documentation — Chronological Process, Parameters,\nTrain/Test Strategy & Results')
    r2.font.size = Pt(13)
    r2.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    p3 = doc.add_paragraph()
    p3.paragraph_format.space_before = Pt(20)
    p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r3 = p3.add_run('PMLB Benchmark Project  |  sklearn-benchmarks\nJune 2026')
    r3.font.size = Pt(11)
    r3.font.color.rgb = RGBColor(0x77, 0x77, 0x77)

    doc.add_page_break()

    # ─────────────────────────────────────────────────
    # EXECUTIVE SUMMARY
    # ─────────────────────────────────────────────────
    add_heading(doc, '1. Executive Summary', 1)
    add_para(doc, (
        'This document provides a complete, chronological account of how three classification models — '
        'Logistic Regression (LR), Random Forest (RF), and XGBoost (XGB) — were evaluated across '
        'up to 165 PMLB benchmark datasets. It covers every step from raw data ingestion through '
        'preprocessing, hyperparameter search, cross-validation, metric computation, result aggregation, '
        'and final ranking.'
    ))
    add_para(doc, 'Key facts at a glance:', bold=True)
    add_bullet(doc, 'No holdout train/test split was used at any point. All evaluation was via 10-fold Stratified Cross-Validation applied to the full dataset.')
    add_bullet(doc, 'RobustScaler preprocessing was applied inside the pipeline (preventing data leakage across folds).')
    add_bullet(doc, 'LR and RF used random hyperparameter search; XGB used exhaustive grid search.')
    add_bullet(doc, 'Three metrics were recorded per run: Accuracy, Macro F1, and TPOT Balanced Accuracy.')
    add_bullet(doc, 'Best results per dataset were extracted by taking the maximum accuracy across all parameter combinations.')
    add_bullet(doc, 'Final results were stored in dataset_manifest.csv (130 datasets × 29 columns).')
    add_separator(doc)

    # ─────────────────────────────────────────────────
    # SECTION 2: INPUT DATA SOURCES
    # ─────────────────────────────────────────────────
    add_heading(doc, '2. Input Data Sources', 1)
    add_para(doc, (
        'The benchmark evaluation consumed two pre-existing compressed result files, each containing '
        'millions of rows of previous evaluation runs. These files are the primary source of truth '
        'for all LR, RF, and XGB scores.'
    ))
    add_table(doc,
        ['File', 'Contains', 'Size', 'Format'],
        [
            ['sklearn-benchmark5-data-edited.tsv.gz', 'LR and RF results (+ 11 other classifiers)', '57 MB', 'gzip TSV, no header'],
            ['sklearn-benchmark6-data.tsv.gz',         'XGB results (grid search)',                  '72 MB', 'gzip TSV, no header'],
        ],
        col_widths=[5, 6, 2.5, 2.5]
    )
    doc.add_paragraph()
    add_para(doc, 'Both files share the same column structure (unnamed, must be assigned manually):')
    add_table(doc,
        ['Column #', 'Assigned Name', 'Description', 'Example'],
        [
            ['1', 'dataset',          'Name of the source dataset file',              'breast_cancer.tsv.gz'],
            ['2', 'classifier',       'Classifier class name',                         'LogisticRegression'],
            ['3', 'parameters',       'Comma-separated key=value parameter string',   'C=0.5,penalty=l2,...'],
            ['4', 'accuracy',         'Accuracy score from cross_val_predict',         '0.9562'],
            ['5', 'macro_f1',         'Macro-averaged F1 score',                       '0.9547'],
            ['6', 'balanced_accuracy','TPOT custom balanced accuracy score',           '0.9563'],
        ],
        col_widths=[2, 3.5, 5.5, 4.5]
    )
    doc.add_paragraph()

    add_heading(doc, '2.1 Dataset Pool Characteristics', 2)
    add_para(doc, (
        'The manifest covers 130 unique PMLB classification datasets spanning a very wide range of sizes, '
        'feature counts, and class structures. All datasets are entirely numeric (no categorical features). '
        'Each dataset is stored as a compressed TSV file with one row per sample.'
    ))
    add_table(doc,
        ['Property', 'Minimum', 'Maximum', 'Mean'],
        [
            ['Rows (samples)',         '32',       '1,025,010',  '17,428'],
            ['Features',               '2',        '1,000',      '41.3'],
            ['Classes',                '2',        '26',         '—'],
            ['Class imbalance metric', '0.000',    '0.698',      '0.093'],
            ['Datasets with imbalance > 0.2', '—', '—',          '18 of 130'],
            ['Feature type',           'numeric',  'numeric',    'all numeric'],
        ],
        col_widths=[5, 3, 3, 3]
    )
    add_separator(doc)

    # ─────────────────────────────────────────────────
    # SECTION 3: PIPELINE ARCHITECTURE
    # ─────────────────────────────────────────────────
    add_heading(doc, '3. Evaluation Pipeline Architecture', 1)
    add_para(doc, (
        'A single shared evaluation function — evaluate_model() in model_code/random_search/evaluate_model.py '
        '(and an identical copy in model_code/grid_search/evaluate_model.py) — was the backbone of every '
        'experiment for all three classifiers. The figure below shows the full pipeline for one '
        'parameter combination on one dataset.'
    ))
    add_image(doc, f1, width=6.2, caption='Figure 1 — Full evaluation pipeline for a single (dataset × parameter combination) run')

    add_heading(doc, '3.1 Dataset Loading', 2)
    add_para(doc, 'Each dataset was loaded from a compressed TSV file using pandas:')
    add_code(doc, "input_data = pd.read_csv(dataset, compression='gzip', sep='\\t')")
    add_para(doc, (
        'After loading, the label column was automatically detected: the code first checks for a column '
        'named "class"; if not found, it falls back to "target". All remaining columns are used as features '
        'and cast to float64 to ensure numeric compatibility.'
    ))
    add_code(doc, "label_column = 'class' if 'class' in input_data.columns else 'target'\n"
                  "features = input_data.drop(label_column, axis=1).values.astype(float)\n"
                  "labels   = input_data[label_column].values")

    add_heading(doc, '3.2 RobustScaler Preprocessing', 2)
    add_para(doc, (
        'Before each classifier, all features were scaled using sklearn\'s RobustScaler. '
        'This scaler is specifically chosen for robustness to outliers because it uses the median '
        'and interquartile range (IQR) rather than the mean and standard deviation:'
    ))
    add_formula_image(doc,
        r'X_{scaled} = \frac{X - \text{median}(X)}{Q_{75}(X) - Q_{25}(X)}',
        width=3.5, caption='RobustScaler formula: median-centered, IQR-scaled')
    add_para(doc, (
        'Crucially, the RobustScaler was placed inside the sklearn Pipeline object alongside the classifier. '
        'This means that during cross-validation, the scaler was re-fitted on the training folds and '
        'then applied to the test fold — the scaler never saw test fold data during fitting. '
        'This completely prevents feature scaling data leakage.'
    ))
    add_code(doc, "clf = make_pipeline(RobustScaler(), Classifier(**params))")

    add_heading(doc, '3.3 Parameter Combination Enumeration', 2)
    add_para(doc, (
        'For each model, the hyperparameter script prepared a list of all parameter dictionaries '
        '(either by random sampling or by grid enumeration). The evaluate_model function then '
        'iterated over this list and ran a full cross-validation for each combination independently. '
        'Each run was isolated — no information was shared between runs.'
    ))
    add_code(doc,
        "for pipe_parameters in pipelines:\n"
        "    clf = make_pipeline(*pipeline)  # fresh pipeline each time\n"
        "    cv_predictions = cross_val_predict(clf, X, y, cv=StratifiedKFold(...))\n"
        "    # compute metrics → print one output line\n"
        "    except Exception: continue  # skip bad combinations silently")
    add_para(doc, (
        'Note: any parameter combination that raised an exception (e.g., dual=True with l1 penalty '
        'in LR which is unsupported, or XGBoost receiving incompatible parameters for a given dataset) '
        'was silently skipped using a broad except clause. This is intentional to allow large-scale '
        'automated search without crashing.'
    ))
    add_separator(doc)

    # ─────────────────────────────────────────────────
    # SECTION 4: CROSS-VALIDATION
    # ─────────────────────────────────────────────────
    add_heading(doc, '4. Train/Test Strategy — Cross-Validation Only (No Holdout Split)', 1)
    add_para(doc, (
        'This is one of the most important aspects of the entire evaluation. There was NO separate '
        'train/test holdout split at any point. No portion of any dataset was reserved as a '
        'held-out test set. Instead, every dataset was evaluated using 10-fold Stratified '
        'Cross-Validation applied to the entire dataset.'
    ))

    add_heading(doc, '4.1 Why No Holdout Split?', 2)
    add_para(doc, (
        'The PMLB benchmark framework was designed to compare classifier performance across a very large '
        'number of datasets and hyperparameter combinations. Using cross-validation on the full dataset '
        'maximises statistical power: every sample contributes both to training and testing. '
        'For small datasets (some as small as 32 rows), a holdout split would leave too few samples '
        'for reliable evaluation. Cross-validation provides a low-variance estimate of generalisation '
        'performance without sacrificing any data.'
    ))

    add_heading(doc, '4.2 StratifiedKFold Configuration', 2)
    add_para(doc, 'Exact cross-validation settings used identically for ALL three classifiers on ALL datasets:')
    add_table(doc,
        ['Parameter', 'Value', 'Effect'],
        [
            ['n_splits',     '10',        'Dataset divided into 10 equal folds; 10 train/test iterations'],
            ['shuffle',      'True',      'Samples shuffled before splitting (prevents ordering bias)'],
            ['random_state', '90483257',  'Fixed seed ensures identical fold assignments on every run'],
        ],
        col_widths=[3.5, 3, 9]
    )
    add_para(doc, (
        '"Stratified" means each fold maintains the same class proportions as the full dataset. '
        'If the dataset has 70% class A and 30% class B, every fold will have approximately '
        '70%/30% split — this is especially important for imbalanced datasets (18 of 130 datasets '
        'had class imbalance > 0.2).'
    ))
    add_image(doc, f2, width=6.2, caption='Figure 2 — 10-Fold Stratified Cross-Validation: fold assignment across all 10 iterations')

    add_heading(doc, '4.3 cross_val_predict Mechanics', 2)
    add_para(doc, (
        'The function cross_val_predict (not cross_val_score) was used. This is a subtle but important '
        'distinction. cross_val_score returns one score per fold; cross_val_predict collects the actual '
        'predicted labels from the test fold of each iteration and assembles them into a single '
        'prediction array covering all samples.'
    ))
    add_para(doc, 'The process per iteration i:')
    add_bullet(doc, 'Training set: all folds except fold i — approximately 90% of samples')
    add_bullet(doc, 'Test set: fold i only — approximately 10% of samples')
    add_bullet(doc, 'The full pipeline (RobustScaler + Classifier) is fitted on training set')
    add_bullet(doc, 'Predictions are made on the test set (10%)')
    add_bullet(doc, 'After all 10 iterations: predictions from all test folds are stacked')
    add_bullet(doc, 'Result: a prediction array of length N covering every sample exactly once')

    add_para(doc, 'Metrics are then computed once on this full-length array:')
    add_code(doc,
        "cv_predictions = cross_val_predict(estimator=clf, X=features, y=labels,\n"
        "                    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=90483257))\n"
        "accuracy         = accuracy_score(labels, cv_predictions)\n"
        "macro_f1         = f1_score(labels, cv_predictions, average='macro')\n"
        "balanced_accuracy = balanced_accuracy_score(labels, cv_predictions)")

    add_heading(doc, '4.4 Data Leakage Prevention Summary', 2)
    add_table(doc,
        ['Risk', 'How It Was Handled', 'Status'],
        [
            ['Scaler fitted on test data',       'RobustScaler inside pipeline — refitted per fold', '✓ No leakage'],
            ['Test data seen during training',   'cross_val_predict strictly separates folds',       '✓ No leakage'],
            ['Repeated samples in test set',     'Each sample appears in test exactly once',         '✓ No leakage'],
            ['Information from best-param selection', 'Best params chosen after all CV runs; no re-evaluation', 'Note: optimistic bias possible — best of many combos'],
        ],
        col_widths=[4.5, 6, 5]
    )
    add_separator(doc)

    # ─────────────────────────────────────────────────
    # SECTION 5: LOGISTIC REGRESSION
    # ─────────────────────────────────────────────────
    add_heading(doc, '5. Logistic Regression (LR) — Full Evaluation Detail', 1)
    add_para(doc, (
        'Script: model_code/random_search/LogisticRegression.py\n'
        'Strategy: Random Search over a continuous and categorical parameter space.'
    ), bold=False)

    add_heading(doc, '5.1 How Random Search Works', 2)
    add_para(doc, (
        'Random search was used instead of grid search. A seed value (passed as sys.argv[3]) was used '
        'to set np.random.seed() before sampling. numpy sampling functions then drew independent random '
        'values for each parameter from the distributions defined below. A set of num_param_combinations '
        '(sys.argv[2]) combinations was generated, zipped together, and passed to evaluate_model.'
    ))

    add_heading(doc, '5.2 Hyperparameter Space', 2)
    add_table(doc,
        ['Parameter', 'Type', 'Distribution / Values', 'Notes'],
        [
            ['C',             'float',   'np.random.uniform(1e-10, 10.0)',    'Inverse regularisation strength; small C = heavy regularisation'],
            ['penalty',       'str',     "np.random.choice(['l1', 'l2'])",    'l1=Lasso (sparse), l2=Ridge (shrinkage)'],
            ['fit_intercept', 'bool',    "np.random.choice([True, False])",   'Whether to fit a bias/intercept term'],
            ['dual',          'bool',    "np.random.choice([True, False])",   'Forced to False when penalty=l1 (dual only valid for l2 with liblinear)'],
            ['solver',        'str',     "'liblinear' (fixed)",               'Only solver compatible with both l1 and l2 at this version'],
            ['random_state',  'int',     '324089 (fixed)',                    'Controls solver randomness for reproducibility'],
        ],
        col_widths=[3, 1.8, 5, 5.7]
    )
    add_image(doc, f5, width=6.2, caption='Figure 3 — Logistic Regression random search parameter distributions')

    add_heading(doc, '5.3 Scale and Results', 2)
    add_table(doc,
        ['Metric', 'Value'],
        [
            ['Datasets attempted',                  '165'],
            ['Datasets with valid results',          '128'],
            ['Average parameter combos per dataset', '~237 (min 12, max 240)'],
            ['Total evaluation rows in raw file',    '~39,145'],
            ['Skipped (timeout)',                    '2 (kddcup, mnist — too large)'],
            ['Best balanced accuracy achieved',      '1.000 (clean2 dataset)'],
            ['Mean balanced accuracy across datasets','0.7368'],
            ['Std balanced accuracy',                '0.1712'],
            ['Min balanced accuracy',                '0.0938'],
        ],
        col_widths=[8, 7.5]
    )
    add_separator(doc)

    # ─────────────────────────────────────────────────
    # SECTION 6: RANDOM FOREST
    # ─────────────────────────────────────────────────
    add_heading(doc, '6. Random Forest (RF) — Full Evaluation Detail', 1)
    add_para(doc, 'Script: model_code/random_search/RandomForestClassifier.py\nStrategy: Random Search (larger space than LR).')

    add_heading(doc, '6.1 Hyperparameter Space', 2)
    add_table(doc,
        ['Parameter', 'Type', 'Distribution / Values', 'Notes'],
        [
            ['n_estimators',          'int',   'np.random.choice(range(50,1001,50))',           '20 possible values: 50, 100, …, 1000'],
            ['min_impurity_decrease', 'float', 'np.random.exponential(scale=0.01)',             'Exponential distribution — most values near zero; promotes small thresholds'],
            ['max_features',          'mixed', 'np.random.choice(arange(0.01,1,0.01)+["sqrt","log2",None])', '99 fractions + 3 options = 102 choices total'],
            ['criterion',             'str',   "np.random.choice(['gini', 'entropy'])",         'Split quality measure; gini=Gini impurity, entropy=information gain'],
            ['max_depth',             'mixed', 'np.random.choice(range(1,51)+[None])',           '50 integers + None (unlimited) = 51 choices'],
            ['random_state',          'int',   '324089 (fixed)',                                'Tree construction seed for reproducibility'],
        ],
        col_widths=[3.5, 1.5, 5.5, 5]
    )
    add_image(doc, f6, width=6.2, caption='Figure 4 — Random Forest random search parameter distributions')

    add_heading(doc, '6.2 Scale and Results', 2)
    add_table(doc,
        ['Metric', 'Value'],
        [
            ['Datasets attempted',                    '165'],
            ['Datasets with valid results',            '125'],
            ['Average parameter combos per dataset',   '~767 (min 465, max 770)'],
            ['Total evaluation rows in raw file',      '~126,602'],
            ['Skipped (timeout)',                      '5 (kddcup, mnist, mfeat_factors, mfeat_fourier, poker)'],
            ['Best balanced accuracy achieved',        '1.000 (multiple datasets: clean1, clean2, mux6, mushroom, agaricus_lepiota...)'],
            ['Mean balanced accuracy across datasets', '0.7841'],
            ['Std balanced accuracy',                  '0.1859'],
            ['Min balanced accuracy',                  '0.000 (at least one dataset — likely a degenerate result)'],
        ],
        col_widths=[8, 7.5]
    )
    add_para(doc, (
        'RF evaluated approximately 3× more parameter combinations per dataset than LR, reflecting the '
        'larger and more complex search space. The exponential distribution for min_impurity_decrease '
        'means the vast majority of sampled values are very small (< 0.05), which is consistent with '
        'practical use where small impurity thresholds are most commonly useful.'
    ))
    add_separator(doc)

    # ─────────────────────────────────────────────────
    # SECTION 7: XGBOOST
    # ─────────────────────────────────────────────────
    add_heading(doc, '7. XGBoost (XGB) — Full Evaluation Detail', 1)
    add_para(doc, (
        'Script: model_code/grid_search/XGBClassifier.py\n'
        'Strategy: EXHAUSTIVE GRID SEARCH — every possible combination of all parameter values was evaluated. '
        'This is fundamentally different from LR and RF which used random sampling.'
    ))

    add_heading(doc, '7.1 Grid Search vs. Random Search', 2)
    add_table(doc,
        ['Aspect', 'Random Search (LR, RF)', 'Grid Search (XGB)'],
        [
            ['Coverage',         'Samples a subset of parameter space',   'Evaluates every possible combination'],
            ['Reproducibility',  'Depends on random seed per run',         'Fully deterministic — same order always'],
            ['Computational cost','Scales linearly with num_combinations', 'Scales as product of all value counts'],
            ['Guarantees',       'No guarantee of finding optimal params','Guaranteed to evaluate the global optimum within the grid'],
            ['Implementation',   'numpy random sampling',                  'itertools.product()'],
        ],
        col_widths=[3.5, 6, 6]
    )

    add_heading(doc, '7.2 Hyperparameter Grid', 2)
    add_table(doc,
        ['Parameter', 'Values in Grid', 'Count', 'Notes'],
        [
            ['n_estimators',  '[10, 50, 100, 500]',                                '4',  'Number of boosting rounds'],
            ['learning_rate', '[0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0]',         '7',  'Step size shrinkage — very wide range tested'],
            ['gamma',         'arange(0.0, 0.51, 0.05)',                           '11', 'Minimum loss reduction for split; 0=no regularisation'],
            ['max_depth',     '[1, 2, 3, 4, 5, 10, 20, 50, None]',                 '9',  'Maximum tree depth'],
            ['subsample',     'arange(0.0, 1.01, 0.1)',                            '11', 'Fraction of samples used per tree'],
            ['seed',          '324089 (fixed)',                                    '1',  'Fixed seed for reproducibility'],
            ['nthread',       '1 (fixed)',                                          '1',  'Single-threaded to avoid parallelism interference'],
            ['TOTAL combos',  '4 × 7 × 11 × 9 × 11',                             '30,492', 'Per dataset'],
        ],
        col_widths=[3.2, 5.5, 1.5, 5.3]
    )
    add_image(doc, f7, width=6.2, caption='Figure 5 — XGBoost exhaustive grid search: parameter space coverage visualization')

    add_heading(doc, '7.3 Scale and Results', 2)
    add_table(doc,
        ['Metric', 'Value'],
        [
            ['Datasets attempted',                    '130 (fewer than LR/RF due to coming from different benchmark run)'],
            ['Datasets with valid results',            '99'],
            ['Parameter combos per dataset',           '30,492 (all exhaustively evaluated)'],
            ['Skipped (no_valid_result)',               '31 datasets — all parameter combinations failed or incompatible'],
            ['Skipped (timeout)',                       '1 dataset (fars)'],
            ['Best balanced accuracy achieved',        '1.000 (clean1, irish, mushroom, clean2)'],
            ['Mean balanced accuracy across datasets', '0.6185'],
            ['Std balanced accuracy',                  '0.1421'],
            ['Min balanced accuracy',                  '0.4633 (near random chance)'],
        ],
        col_widths=[8, 7.5]
    )
    add_para(doc, (
        'The lower mean balanced accuracy for XGB (0.6185) compared to LR (0.7368) and RF (0.7841) '
        'does NOT mean XGB is a weaker model. It reflects that XGB was evaluated on a different subset '
        'of datasets (99 vs 125–128), and the 31 missing datasets may include easier ones where XGB '
        'would have scored highly. The "no_valid_result" skips for XGB indicate that across all 30,492 '
        'parameter combinations, every single one threw an exception for those 31 datasets.'
    ))
    add_image(doc, f3, width=6.2, caption='Figure 6 — Comparison of search space sizes and dataset coverage across all three classifiers')
    add_separator(doc)

    # ─────────────────────────────────────────────────
    # SECTION 8: METRICS
    # ─────────────────────────────────────────────────
    add_heading(doc, '8. Evaluation Metrics', 1)
    add_para(doc, (
        'Three metrics were computed for every single (dataset × parameter combination) evaluation. '
        'All three were computed from the same set of cross-validation out-of-fold predictions — '
        'no separate test set was used.'
    ))
    add_image(doc, f8, width=6.4, caption='Figure 7 — All three metric formulas with full mathematical definitions')

    add_heading(doc, '8.1 Accuracy', 2)
    add_formula_image(doc,
        r'\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Samples}}',
        width=5.5, caption='Accuracy formula')
    add_para(doc, (
        'Standard classification accuracy. Treats all samples equally regardless of their class. '
        'Can be misleading on imbalanced datasets — a model predicting only the majority class on '
        'a 95/5 split achieves 95% accuracy without learning anything meaningful.'
    ))

    add_heading(doc, '8.2 Macro F1 Score', 2)
    add_formula_image(doc,
        r'\text{Macro-F1} = \frac{1}{|C|} \sum_{c \in C} \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}',
        width=6.0, caption='Macro F1 formula — unweighted mean of per-class F1 scores')
    add_para(doc, (
        'F1 score is the harmonic mean of precision and recall per class, then averaged unweighted '
        'across all classes. "Macro" averaging means each class counts equally, regardless of size. '
        'This penalises models that ignore minority classes.'
    ))

    add_heading(doc, '8.3 TPOT Balanced Accuracy (Custom Implementation)', 2)
    add_para(doc, (
        'This is NOT sklearn\'s balanced_accuracy_score. It is a custom implementation from the '
        'TPOT 2016 library (tpot_metrics.py). The computation uses a one-vs-rest approach for '
        'each class and computes both sensitivity and specificity:'
    ))
    add_table(doc,
        ['Term', 'Formula', 'Meaning'],
        [
            ['Sensitivity_c (Recall)',
             'TP_c / (TP_c + FN_c)',
             'Fraction of true class-c samples correctly predicted as class c'],
            ['Specificity_c',
             'TN_c / (TN_c + FP_c)',
             'Fraction of non-class-c samples correctly predicted as NOT class c'],
            ['ClassAccuracy_c',
             '(Sensitivity_c + Specificity_c) / 2',
             'Per-class balanced accuracy for class c'],
            ['BalancedAccuracy',
             'mean(ClassAccuracy_c) over all classes',
             'Final score: unweighted mean across all classes'],
        ],
        col_widths=[3.5, 4.5, 7.5]
    )
    add_formula_image(doc,
        r'\text{BalancedAccuracy} = \frac{1}{|C|} \sum_{c \in C} \frac{Sensitivity_c + Specificity_c}{2}',
        width=6.0, caption='TPOT Balanced Accuracy formula — range: 0.5 (random) to 1.0 (perfect)')
    add_para(doc, (
        'The range of this metric is [0.5, 1.0] where 0.5 corresponds to random chance performance '
        '(equivalent to a classifier that always predicts the majority class or assigns predictions '
        'randomly) and 1.0 represents perfect classification of every sample. This is the PRIMARY '
        'metric used for ranking datasets in the final output files.'
    ))
    add_separator(doc)

    # ─────────────────────────────────────────────────
    # SECTION 9: BEST RESULT EXTRACTION
    # ─────────────────────────────────────────────────
    add_heading(doc, '9. Best Result Extraction — From Raw to Manifest', 1)
    add_para(doc, (
        'The raw benchmark files contain one row for every single (dataset, classifier, parameter combination) '
        'triple. To produce a meaningful comparison, only the best-performing parameter combination '
        'per dataset per classifier was retained.'
    ))
    add_image(doc, f9, width=6.2, caption='Figure 8 — Best result extraction workflow: from millions of raw rows to 130-row manifest')

    add_heading(doc, '9.1 Extraction Logic', 2)
    add_para(doc, 'The extraction used the following pandas operations (from analyze-sklearn-benchmark.ipynb):')
    add_code(doc,
        "# Step 1: Load raw benchmark file\n"
        "data = pd.read_csv('sklearn-benchmark5-data-edited.tsv.gz', sep='\\t',\n"
        "         names=['dataset','classifier','parameters','accuracy','macro_f1','bal_accuracy'])\n"
        "\n"
        "# Step 2: Group by (dataset, classifier) and take maximum accuracy\n"
        "best = data.groupby(['dataset', 'classifier'])['accuracy'].max().reset_index()\n"
        "\n"
        "# Step 3: This best accuracy row's corresponding balanced_accuracy and macro_f1\n"
        "#         were stored alongside it in the manifest")
    add_para(doc, (
        'Important implication: the "best" result is selected by maximum accuracy, not by maximum '
        'balanced accuracy. The balanced accuracy and macro_f1 values stored in the manifest are '
        'those co-occurring with the highest-accuracy parameter combination — not the independently '
        'highest balanced accuracy possible. In practice the two often agree, but they can differ.'
    ))

    add_heading(doc, '9.2 Result of Extraction', 2)
    add_table(doc,
        ['Classifier', 'Raw Rows', 'After Groupby (best per dataset)', 'In Manifest'],
        [
            ['LogisticRegression',    '~39,145',    '128 (one per dataset)',  '128'],
            ['RandomForestClassifier','~126,602',   '125 (one per dataset)',  '125'],
            ['XGBClassifier',         '~3M (est.)', '99 (one per dataset)',   '99'],
        ],
        col_widths=[4, 3.5, 5.5, 2.5]
    )
    add_separator(doc)

    # ─────────────────────────────────────────────────
    # SECTION 10: DATASET MANIFEST
    # ─────────────────────────────────────────────────
    add_heading(doc, '10. Dataset Manifest (dataset_manifest.csv)', 1)
    add_para(doc, (
        'All best-per-classifier results were consolidated with dataset metadata into a single '
        '130-row × 29-column manifest file. This file is the central reference for all downstream '
        'analysis including MLP dataset selection.'
    ))
    add_table(doc,
        ['Column', 'Type', 'Description'],
        [
            ['dataset',                      'str',   'Filename of the source .tsv.gz file'],
            ['target_column',                'str',   'Name of the label column ("class" or "target")'],
            ['rows',                         'int',   'Number of samples in the dataset'],
            ['features',                     'int',   'Number of input features'],
            ['classes',                      'int',   'Number of distinct class labels'],
            ['class_imbalance',              'float', 'Imbalance metric (0 = perfectly balanced, 1 = completely dominated by one class)'],
            ['missing_values_count',         'int',   'Total count of NaN values across all cells'],
            ['missing_values_ratio',         'float', 'Fraction of all cells that are NaN'],
            ['numeric_features',             'int',   'Count of numeric feature columns'],
            ['categorical_features',         'int',   'Count of categorical feature columns (all 0 here — all numeric)'],
            ['feature_type',                 'str',   '"numeric" for all 130 datasets'],
            ['xgb_accuracy',                 'float', 'Best accuracy achieved by XGBoost across all grid search combinations'],
            ['xgb_macro_f1',                 'float', 'Macro F1 for the same best-accuracy XGB parameter combination'],
            ['xgb_balanced_accuracy',        'float', 'TPOT balanced accuracy for same best-accuracy XGB combo; NaN if XGB skipped'],
            ['rf_accuracy',                  'float', 'Best accuracy achieved by Random Forest'],
            ['rf_macro_f1',                  'float', 'Macro F1 for best-accuracy RF combination'],
            ['rf_balanced_accuracy',         'float', 'TPOT balanced accuracy for best-accuracy RF combo; NaN if RF skipped'],
            ['logreg_accuracy',              'float', 'Best accuracy achieved by Logistic Regression'],
            ['logreg_macro_f1',              'float', 'Macro F1 for best-accuracy LR combination'],
            ['logreg_balanced_accuracy',     'float', 'TPOT balanced accuracy for best-accuracy LR combo; NaN if LR skipped'],
            ['n_models_success',             'int',   'How many of the 3 classifiers produced a valid result (0–3)'],
            ['mean_accuracy_3models',        'float', 'Arithmetic mean of xgb, rf, logreg best accuracies'],
            ['mean_macro_f1_3models',        'float', 'Arithmetic mean of the three macro F1 values'],
            ['mean_balanced_accuracy_3models','float','Arithmetic mean of the three balanced accuracies'],
            ['std_accuracy_3models',         'float', 'Standard deviation of the three accuracy values — measures classifier agreement'],
            ['selected_for_mlp_eval',        'bool',  'Whether this dataset was selected for MLP neural network evaluation'],
            ['selection_rank_top40',         'int',   'Rank within top 40 selected datasets (1=best)'],
            ['selection_score',              'float', 'Composite score used for selection'],
            ['selection_justification_short','str',   'Brief reason for selection or rejection'],
        ],
        col_widths=[4.5, 1.5, 9.5]
    )

    add_heading(doc, '10.1 Model Coverage Breakdown', 2)
    add_table(doc,
        ['Coverage', 'Count', 'Percentage', 'Notes'],
        [
            ['All 3 models have results', '97',  '74.6%', 'Full comparison possible'],
            ['Exactly 2 models have results', '30', '23.1%', 'Partial comparison — usually XGB missing'],
            ['Exactly 1 model has results',   '1',   '0.8%', 'Very limited comparison'],
            ['No models have results',         '2',   '1.5%', 'Both failed/skipped on all classifiers'],
        ],
        col_widths=[5, 2, 3, 5.5]
    )
    add_separator(doc)

    # ─────────────────────────────────────────────────
    # SECTION 11: SKIPPED DATASETS
    # ─────────────────────────────────────────────────
    add_heading(doc, '11. Skipped Datasets and Why', 1)
    add_heading(doc, '11.1 Timeout Skips (LR and RF)', 2)
    add_table(doc,
        ['Dataset', 'Classifier', 'Rows', 'Features', 'Classes', 'Reason'],
        [
            ['kddcup',        'LR',         '494,020', '41', '23', 'Too large — CV too slow for LR'],
            ['mnist',         'LR',         '70,000',  '784', '10', 'High-dimensional — too slow for LR'],
            ['kddcup',        'RF',         '494,020', '41', '23', 'Too large for RF CV'],
            ['mnist',         'RF',         '70,000',  '784', '10', 'High-dimensional'],
            ['mfeat_factors', 'RF',         '—',       '—',   '—',  'Timeout'],
            ['mfeat_fourier', 'RF',         '—',       '—',   '—',  'Timeout'],
            ['poker',         'RF',         '1,025,010','10','10',  'Largest dataset — 1M+ rows too slow'],
        ],
        col_widths=[3.2, 2, 2, 2.2, 1.8, 4.3]
    )

    add_heading(doc, '11.2 No-Valid-Result Skips (XGB — 31 Datasets)', 2)
    add_para(doc, (
        'For 31 datasets, every single one of the 30,492 XGB parameter combinations threw an exception '
        'and was silently skipped. This typically occurs when: (a) the dataset has class labels that '
        'XGBoost cannot handle (e.g., string labels not converted to integers), (b) dataset has features '
        'that cause numerical issues across all parameter settings, or (c) subsample=0.0 (included in '
        'the grid) causes "empty training set" errors. '
        'Examples: ann_thyroid, calendarDOW, collins, dermatology, dna, ecoli, flags, haberman, hayes_roth...'
    ))
    add_separator(doc)

    # ─────────────────────────────────────────────────
    # SECTION 12: FINAL RANKED OUTPUTS
    # ─────────────────────────────────────────────────
    add_heading(doc, '12. Final Ranked Output Files', 1)
    add_para(doc, (
        'From the dataset_manifest.csv, four ranked CSV files were generated by sorting on '
        'balanced_accuracy in descending order. These files represent the final deliverable '
        'of the LR/RF/XGB evaluation phase.'
    ))
    add_table(doc,
        ['File', 'Sort Column', 'Rows', 'Columns'],
        [
            ['LogisticRegression_ranked_by_balanced_accuracy.csv',    'logreg_balanced_accuracy',  '128', 'rank, dataset, classifier, balanced_accuracy, accuracy, macro_f1'],
            ['RandomForestClassifier_ranked_by_balanced_accuracy.csv','rf_balanced_accuracy',       '125', 'rank, dataset, classifier, balanced_accuracy, accuracy, macro_f1'],
            ['XGBClassifier_ranked_by_balanced_accuracy.csv',         'xgb_balanced_accuracy',      '99',  'rank, dataset, classifier, balanced_accuracy, accuracy, macro_f1'],
            ['All_3_classifiers_ranked_by_balanced_accuracy.csv',     'balanced_accuracy',          '352', 'combined rows for all 3 classifiers'],
        ],
        col_widths=[6.5, 4, 1.5, 3.5]
    )
    add_image(doc, f4, width=6.2, caption='Figure 9 — Score distributions (boxplots) for Balanced Accuracy and Accuracy across all datasets')
    add_image(doc, f10, width=6.2, caption='Figure 10 — Dataset size (log scale) vs balanced accuracy for each classifier with trend line')
    add_separator(doc)

    # ─────────────────────────────────────────────────
    # SECTION 13: KEY CONSTANTS
    # ─────────────────────────────────────────────────
    add_heading(doc, '13. Key Constants and Reproducibility', 1)
    add_table(doc,
        ['Constant', 'Value', 'Where Used', 'Effect'],
        [
            ['CV random_state',       '90483257', 'StratifiedKFold — ALL classifiers', 'Fixes fold assignment; same data split every run'],
            ['Model random_state',    '324089',   'Classifier — ALL models (LR: random_state, RF: random_state, XGB: seed)', 'Fixes internal model randomness'],
            ['CV n_splits',           '10',       'StratifiedKFold', '10 folds; ~90/10 train/test split per iteration'],
            ['CV shuffle',            'True',     'StratifiedKFold', 'Rows shuffled before folding; removes ordering effects'],
            ['XGB nthread',           '1',        'XGBClassifier',   'Forces single-threaded XGB; prevents parallelism interference'],
            ['Solver (LR)',           'liblinear','LogisticRegression','Supports both l1 and l2 penalty; fast for small datasets'],
            ['Warnings suppressed',   'True',     'evaluate_model()', 'All sklearn warnings silenced during evaluation loop'],
        ],
        col_widths=[3.5, 2.5, 4.5, 5]
    )
    add_separator(doc)

    # ─────────────────────────────────────────────────
    # SECTION 14: COMPLETE CHRONOLOGICAL SUMMARY
    # ─────────────────────────────────────────────────
    add_heading(doc, '14. Complete Chronological Process Summary', 1)
    add_table(doc,
        ['Step', 'Action', 'Detail', 'Output'],
        [
            ['1',  'Data ingestion',              'Load .tsv.gz, detect label column, cast features to float',     'X (float64), y (labels)'],
            ['2',  'Preprocessing setup',         'RobustScaler placed inside sklearn Pipeline',                   'Pipeline object'],
            ['3',  'Parameter list generation',   'LR/RF: numpy random sampling; XGB: itertools.product grid',    'List of param dicts'],
            ['4',  'Outer loop: per combination', 'Iterate over each parameter combination',                       'One evaluation per combo'],
            ['5',  'Cross-validation',            'StratifiedKFold(10, shuffle, seed=90483257) via cross_val_predict', 'N-length prediction array'],
            ['6',  'Metric computation',          'accuracy_score, f1_score(macro), TPOT balanced_accuracy_score', '3 float values'],
            ['7',  'Output line written',         'Tab-separated: dataset|classifier|params|acc|f1|ba',            'Row in .tsv.gz result file'],
            ['8',  'Exception handling',          'Any failed combination silently skipped with continue',         'Skip + proceed'],
            ['9',  'Aggregation',                 'Group by (dataset, classifier), take max accuracy',             'One best row per dataset'],
            ['10', 'Manifest creation',           'Merge LR+RF+XGB best scores with dataset metadata',            'dataset_manifest.csv (130×29)'],
            ['11', 'Ranked CSVs generated',       'Sort manifest by balanced_accuracy desc per classifier',        '4 ranked CSV files'],
            ['12', 'Dataset selection',           'Composite scoring to select top/bottom datasets for MLP eval', 'data/top_dataset.csv, data/bottom_dataset.csv'],
        ],
        col_widths=[0.8, 3.7, 6.5, 4.5]
    )

    doc.save(OUT)
    print(f"\nDocument saved to:\n{OUT}")
    print("Done.")

if __name__ == '__main__':
    build_doc()

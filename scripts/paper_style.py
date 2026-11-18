# scripts/paper_style.py
import matplotlib as mpl
import matplotlib.font_manager as fm

MM = 1 / 25.4
COL_W  = 88  * MM     # 单栏宽度 (in)，约 3.465
PAGE_W = 181 * 0.7 * MM     # 跨栏宽度 (in)，约 7.126

def apply(font_dir="fonts"):
    for f in ["TIMES.TTF", "TIMESBD.TTF", "TIMESI.TTF", "TIMESBI.TTF"]:
        fm.fontManager.addfont(f"{font_dir}/{f}")
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",      # 以防出现 $\alpha$ 等数学符号
        "pdf.fonttype": 42,              # TrueType 完整嵌入，而非默认 Type3 位图
        "svg.fonttype": "none",          # 仅供你自己编辑用，不提交给期刊
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 9,
    })
"""Render image annotations and fixed-coordinate FWHM distributions."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

METHODS = ("DEPICT", "BPM+TV", "Rytov")
COLORS = {"DEPICT": "#D55E00", "BPM+TV": "#0072B2", "Rytov": "#009E73"}
METHOD_NAMES = {"DEPICT": "DEPICT", "BPM+TV": "TV-BPM", "Rytov": "Rytov"}
WIDTH_IN = 8.5 / 2.54


def setup_style():
    font = Path(__file__).resolve().parents[3] / "postprocessing/fonts/Arial.TTF"
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": "Arial", "font.size": 6,
                         "svg.fonttype": "none", "svg.hashsalt": "DEPICT-Figure3",
                         "axes.linewidth": .5})


def draw_boxes(ax, values, panel, config, annotated=True):
    cells = panel["cells"]
    for method, offset in zip(METHODS, (-.24, 0, .24)):
        distributions = [values.loc[(values.cell == cell) & (values.method == method),
                                     "axial_lateral_ratio"].to_numpy() for cell in cells]
        ax.boxplot(distributions, positions=np.arange(len(cells)) + offset,
                   widths=.20, whis=1.5, patch_artist=True, showfliers=False,
                   manage_ticks=False, zorder=2,
                   boxprops=dict(facecolor=COLORS[method], edgecolor="#333333", linewidth=.5),
                   whiskerprops=dict(color="#333333", linewidth=.5),
                   capprops=dict(color="#333333", linewidth=.5),
                   medianprops=dict(color="#111111", linewidth=.5))
    ax.axhline(1, color="#333333", linestyle=(0, (3, 3)), linewidth=.5, alpha=.65)
    ax.set(xlim=(-.55, len(cells)-.45), ylim=(0, 3.2),
           xticks=np.arange(len(cells)), yticks=[1, 3])
    ax.tick_params(length=2, width=.5, pad=2, labelsize=5,
                   labelbottom=annotated, labelleft=annotated)
    ax.set_xticklabels([config["cell_names"][cell] for cell in cells])
    if annotated:
        ax.set_yticklabels(["1.0", "3.0"])
        ax.set_title(panel["title"], fontsize=6, pad=5)
        ax.legend(handles=[Patch(facecolor=COLORS[m], edgecolor="#333333",
                                 linewidth=.4, label=METHOD_NAMES[m]) for m in METHODS],
                  frameon=False, loc="upper left", fontsize=5, handlelength=2,
                  borderaxespad=.5, labelspacing=.3)


def save_boxplot(root, config, panel, values):
    output = root / ("panel_" + panel["label"])
    fig, ax = plt.subplots(figsize=(WIDTH_IN, WIDTH_IN/2))
    fig.subplots_adjust(left=.08, right=.985, bottom=.15, top=.97)
    draw_boxes(ax, values, panel, config, annotated=False)
    fig.savefig(output / ("panel_" + panel["label"] + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(WIDTH_IN, WIDTH_IN/2))
    fig.subplots_adjust(left=.085, right=.985, bottom=.15, top=.89)
    draw_boxes(ax, values, panel, config)
    fig.text(.005, .99, panel["label"], weight="bold", va="top")
    fig.savefig(output / ("panel_" + panel["label"] + "_with_text.svg"),
                metadata={"Date": None})
    plt.close(fig)


def draw_image(ax, image, panel, geometry):
    width, height = image.size
    ax.imshow(image, extent=(0, width, height, 0), interpolation="nearest")
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis("off")
    boxes = geometry["boxes_xyxy"]
    scale = height/450
    for name, label in (("xy", "DEPICT"), ("tide", "DEPICT"),
                        ("comparison", "Rytov"), ("zoom_rytov", "Rytov"),
                        ("zoom_bpm", "TV-BPM"), ("zoom_tide", "DEPICT")):
        x0, y0, x1, y1 = boxes[name]
        ax.text(x0+8*scale, y0+9*scale, label, color="white", fontsize=5,
                ha="left", va="top")
    x0, y0, x1, y1 = boxes["comparison"]
    ax.text(x1-8*scale, y1-8*scale, "TV-BPM", color="white", fontsize=5,
            ha="right", va="bottom")
    for name, horizontal in (("xy", "x"), ("tide", "z")):
        x0, y0, x1, y1 = boxes[name]
        x, y, length = x0+16*scale, y1-15*scale, 65*scale
        arrow = dict(arrowstyle="->", color="white", lw=.45, mutation_scale=4)
        ax.annotate("", (x+length, y), (x, y), arrowprops=arrow)
        ax.annotate("", (x, y-length), (x, y), arrowprops=arrow)
        ax.text(x+length+5*scale, y, horizontal, color="white", fontsize=5, va="center")
        ax.text(x, y-length-6*scale, "y", color="white", fontsize=5, ha="center")
    ax.set_title(panel["title"], fontsize=6, pad=3)
    ax.text(-.015, 1.03, panel["label"], transform=ax.transAxes,
            fontsize=7, weight="bold", va="bottom")


def save_image_svg(root, panel, geometry):
    path = root/("panel_"+panel["label"])
    with Image.open(path/("panel_"+panel["label"]+".tif")) as image:
        fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN*.51))
        ax = fig.add_axes([.015, .015, .97, .89])
        draw_image(ax, image, panel, geometry)
        fig.savefig(path/("panel_"+panel["label"]+"_with_text.svg"), dpi=300,
                    metadata={"Date": None})
        plt.close(fig)


def assemble(root, config, geometries, tables):
    fig = plt.figure(figsize=(2*WIDTH_IN+.22, 7.65))
    images = {p["label"]:p for p in config["panels"] if p["kind"] == "image"}
    for row, letters in enumerate((("a", "e"), ("b", "f"), ("c", "g"))):
        for column, letter in enumerate(letters):
            ax = fig.add_axes([.025+column*.50, .740-row*.225, .472, .200])
            with Image.open(root/("panel_"+letter)/("panel_"+letter+".tif")) as image:
                draw_image(ax, image, images[letter], geometries[letter]["geometry"])
    for column, letter in enumerate(("d", "h")):
        panel = next(p for p in config["panels"] if p["label"] == letter)
        ax = fig.add_axes([.055+column*.50, .055, .44, .18])
        draw_boxes(ax, tables[letter], panel, config)
        ax.text(-.08, 1.045, letter, transform=ax.transAxes, weight="bold", fontsize=7)
    cax = fig.add_axes([.25, .269, .50, .009])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(1.33, 1.39), cmap="gray"),
                       cax=cax, orientation="horizontal", ticks=[1.33, 1.39])
    bar.ax.tick_params(length=0, pad=2, labelsize=5)
    bar.set_label("Refractive index", fontsize=6, labelpad=-5)
    fig.text(.26, .978, "In-distribution", ha="center", weight="bold", fontsize=7)
    fig.text(.76, .978, "Out-of-distribution", ha="center", weight="bold", fontsize=7)
    fig.savefig(root/"Figure3_reproduced.pdf", dpi=300,
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

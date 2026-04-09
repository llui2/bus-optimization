import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def draw_node(ax, xy, label, radius=0.12, edgecolor="black", lw=1.8):
    circle = Circle(xy, radius, facecolor="white", edgecolor=edgecolor, linewidth=lw, zorder=3)
    ax.add_patch(circle)
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=12, zorder=4)

def draw_edge(ax, p1, p2, color="black", lw=2.2, linestyle="-", alpha=1.0, zorder=2):
    ax.plot(
        [p1[0], p2[0]], [p1[1], p2[1]],
        color=color, linewidth=lw, linestyle=linestyle, alpha=alpha, zorder=zorder
    )

def put_edge_label(ax, p1, p2, text, dx=0.0, dy=0.0, color="0.15", fontsize=12):
    mx = (p1[0] + p2[0]) / 2
    my = (p1[1] + p2[1]) / 2
    ax.text(mx + dx, my + dy, text, ha="center", va="center", fontsize=fontsize, color=color)

def main():
    os.makedirs("results", exist_ok=True)

    granate = "#7A0019"
    gris = "0.55"
    gris_suau = "0.78"
    negre_suau = "0.15"

    # únic camí mínim
    fig_a, ax = plt.subplots(figsize=(6, 5.3))

    pos_a = {
        "i": (0.0, 0.0),
        "a": (1.0, 0.0),
        "b": (2.0, 0.0),
        "j": (3.0, 0.0),
        "c": (1.5, -1.0)
    }

    draw_edge(ax, pos_a["i"], pos_a["a"], color=granate, lw=2.8)
    draw_edge(ax, pos_a["a"], pos_a["b"], color=granate, lw=2.8)
    draw_edge(ax, pos_a["b"], pos_a["j"], color=granate, lw=2.8)

    draw_edge(ax, pos_a["i"], pos_a["c"], color=gris_suau, lw=1.8, linestyle="--")
    draw_edge(ax, pos_a["c"], pos_a["j"], color=gris_suau, lw=1.8, linestyle="--")

    for node, xy in pos_a.items():
        draw_node(ax, xy, node)

    put_edge_label(ax, pos_a["i"], pos_a["a"], r"$D_{ij}/3$", dy=0.16)
    put_edge_label(ax, pos_a["a"], pos_a["b"], r"$D_{ij}/3$", dy=0.16)
    put_edge_label(ax, pos_a["b"], pos_a["j"], r"$D_{ij}/3$", dy=0.16)

    ax.text(1.5, 0.62, "Camí mínim únic", ha="center", fontsize=11, color=granate)
    ax.text(1.5, -1.38, "Camí alternatiu més llarg", ha="center", fontsize=10.5, color=gris)

    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(-1.6, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    plt.tight_layout()
    fig_a.savefig("results/unic_cami_minim.png", dpi=300, bbox_inches="tight")
    fig_a.savefig("results/unic_cami_minim.pdf", bbox_inches="tight")
    plt.close(fig_a)

    # dos camins mínims
    fig_b, ax = plt.subplots(figsize=(6, 5.3))

    pos_b = {
        "i": (0.0, 0.0),
        "a": (1.0, 0.8),
        "b": (1.0, -0.8),
        "j": (2.0, 0.0)
    }

    draw_edge(ax, pos_b["i"], pos_b["a"], color=granate, lw=2.8)
    draw_edge(ax, pos_b["a"], pos_b["j"], color=granate, lw=2.8)
    draw_edge(ax, pos_b["i"], pos_b["b"], color=granate, lw=2.8)
    draw_edge(ax, pos_b["b"], pos_b["j"], color=granate, lw=2.8)

    for node, xy in pos_b.items():
        draw_node(ax, xy, node)

    put_edge_label(ax, pos_b["i"], pos_b["a"], r"$D_{ij}/4$", dx=-0.05, dy=0.10)
    put_edge_label(ax, pos_b["a"], pos_b["j"], r"$D_{ij}/4$", dx=0.05, dy=0.10)
    put_edge_label(ax, pos_b["i"], pos_b["b"], r"$D_{ij}/4$", dx=-0.05, dy=-0.10)
    put_edge_label(ax, pos_b["b"], pos_b["j"], r"$D_{ij}/4$", dx=0.05, dy=-0.10)

    ax.text(1.0, 1.22, "Camí mínim 1", ha="center", fontsize=11, color=granate)
    ax.text(1.0, -1.28, "Camí mínim 2", ha="center", fontsize=11, color=granate)
    ax.text(1.0, 0.05, "La demanda es reparteix", ha="center", va="center", fontsize=11.5, color=negre_suau)

    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    plt.tight_layout()
    fig_b.savefig("results/dos_camins_minims.png", dpi=300, bbox_inches="tight")
    fig_b.savefig("results/dos_camins_minims.pdf", bbox_inches="tight")
    plt.close(fig_b)

    print("Figures generades correctament:")
    print(" - results/unic_cami_minim.png")
    print(" - results/unic_cami_minim.pdf")
    print(" - results/dos_camins_minims.png")
    print(" - results/dos_camins_minims.pdf")

if __name__ == "__main__":
    main()
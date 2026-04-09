import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch


def draw_node(ax, xy, label, radius=0.12, edgecolor="black", lw=1.8, fontsize=12):
    circle = Circle(xy, radius, facecolor="white", edgecolor=edgecolor, linewidth=lw, zorder=5)
    ax.add_patch(circle)
    ax.text(xy[0], xy[1] - 0.32, label, ha="center", va="center", fontsize=fontsize, zorder=6)


def draw_curved_arrow(ax, p1, p2, color="0.55", lw=2.0, rad=0.15, alpha=1.0, zorder=2, ms=16):
    arrow = FancyArrowPatch(
        p1, p2,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="->",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        alpha=alpha,
        zorder=zorder,
        shrinkA=10,
        shrinkB=10,
    )
    ax.add_patch(arrow)


def main():
    os.makedirs("results", exist_ok=True)

    granate = "#7A0019"
    gris = "0.55"
    gris_suau = "0.78"
    negre_suau = "0.15"

    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    pos = {
        "i_1": (-3.3, 1.7),
        "i_2": (-4.1, 0.1),
        "i_3": (-3.5, -1.8),
        "i_4": (-1.2, -0.5),
        "v_5": (1.2, -0.5),
        "j_1": (2.4, 1.4),
        "j_2": (3.8, -0.45),
        "j_3": (3.4, -1.35),
        "j_4": (3.2, -2.35),
    }

    # aresta central comuna
    ax.plot(
        [pos["i_4"][0], pos["v_5"][0]],
        [pos["i_4"][1], pos["v_5"][1]],
        color=granate,
        linewidth=3.0,
        zorder=3,
    )
    ax.text(0.02, -0.18, r"$e$", fontsize=13, color=negre_suau)

    # fluxos entrants cap a i4
    draw_curved_arrow(ax, pos["i_1"], pos["i_4"], color=gris, lw=2.1, rad=-0.22)
    draw_curved_arrow(ax, pos["i_2"], pos["i_4"], color=gris, lw=2.1, rad=-0.12)
    draw_curved_arrow(ax, pos["i_3"], pos["i_4"], color=gris, lw=2.1, rad=-0.10)

    # fluxos sortints des de v5
    draw_curved_arrow(ax, pos["v_5"], pos["j_1"], color=gris, lw=2.1, rad=-0.18)
    draw_curved_arrow(ax, pos["v_5"], pos["j_2"], color=gris, lw=2.1, rad=-0.06)
    draw_curved_arrow(ax, pos["v_5"], pos["j_3"], color=gris, lw=2.1, rad=0.08)
    draw_curved_arrow(ax, pos["v_5"], pos["j_4"], color=gris, lw=2.1, rad=0.18)

    # destacat visual del fet que diversos fluxos passen per e
    #ax.text(-0.2, -1.95, r"$p_e$: acumulació de demanda", fontsize=12, color=granate)

    # nodes
    draw_node(ax, pos["i_1"], r"$i_1$", edgecolor=negre_suau)
    draw_node(ax, pos["i_2"], r"$i_2$", edgecolor=negre_suau)
    draw_node(ax, pos["i_3"], r"$i_3$", edgecolor=negre_suau)
    draw_node(ax, pos["i_4"], r"$i_4$", edgecolor=granate, lw=2.0)
    draw_node(ax, pos["v_5"], r"$j_5$", edgecolor=granate, lw=2.0)
    draw_node(ax, pos["j_1"], r"$j_1$", edgecolor=negre_suau)
    draw_node(ax, pos["j_2"], r"$j_2$", edgecolor=negre_suau)
    draw_node(ax, pos["j_3"], r"$j_3$", edgecolor=negre_suau)
    draw_node(ax, pos["j_4"], r"$j_4$", edgecolor=negre_suau)

    ax.set_xlim(-4.8, 4.4)
    ax.set_ylim(-2.9, 2.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    plt.tight_layout()
    fig.savefig("results/flux_aresta_central.png", dpi=300, bbox_inches="tight")
    fig.savefig("results/flux_aresta_central.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Figures generades correctament:")
    print(" - results/flux_aresta_central.png")
    print(" - results/flux_aresta_central.pdf")


if __name__ == "__main__":
    main()

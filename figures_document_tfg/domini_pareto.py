import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    os.makedirs("results", exist_ok=True)

    # -----------------------------
    # Dades de les solucions
    # -----------------------------
    # Eix X: cost (minimitzar)
    # Eix Y: servei (maximitzar)
    K1 = (4.0, 8.0)   # domina K2
    K2 = (6.5, 5.5)   # dominada per K1
    K3 = (2.8, 6.2)   # no comparable amb K1

    granate = "#7A0019"
    gris = "0.45"
    gris_suau = "0.82"

    fig, ax = plt.subplots(figsize=(7.2, 5.8))

    # -----------------------------
    # Zona dominada per K1
    # -----------------------------
    # Tots els punts amb cost >= K1_x i servei <= K1_y
    rect = Rectangle(
        (K1[0], 0),
        width=10 - K1[0],
        height=K1[1],
        facecolor=granate,
        alpha=0.08,
        edgecolor="none",
        zorder=0
    )
    ax.add_patch(rect)

    # -----------------------------
    # Línies auxiliars
    # -----------------------------
    for x, y in [K1, K2, K3]:
        ax.plot([x, x], [0, y], linestyle="--", color=gris_suau, linewidth=1.0, zorder=1)
        ax.plot([0, x], [y, y], linestyle="--", color=gris_suau, linewidth=1.0, zorder=1)

    # -----------------------------
    # Punts
    # -----------------------------
    ax.scatter(K1[0], K1[1], s=130, color=granate, edgecolors="white", linewidths=0.9, zorder=3)
    ax.scatter(K2[0], K2[1], s=130, color=gris, edgecolors="white", linewidths=0.9, zorder=3)
    ax.scatter(K3[0], K3[1], s=130, color="0.65", edgecolors="white", linewidths=0.9, zorder=3)

    # -----------------------------
    # Etiquetes dels punts
    # -----------------------------
    ax.text(K1[0] + 0.15, K1[1] + 0.15, r"$K_1$", fontsize=12)
    ax.text(K2[0] + 0.15, K2[1] - 0.45, r"$K_2$", fontsize=12)
    ax.text(K3[0] - 0.55, K3[1] + 0.15, r"$K_3$", fontsize=12)

    # -----------------------------
    # Anotacions
    # -----------------------------
    ax.annotate(
        r"$K_1$ domina $K_2$",
        xy=K2,
        xytext=(6.5, 8.3),
        arrowprops=dict(arrowstyle="->", lw=1.2, color=granate),
        fontsize=11,
        color=granate
    )

    # Segment entre K1 i K3 -> indicar que no són comparables
    ax.plot(
        [K3[0], K1[0]],
        [K3[1], K1[1]],
        linestyle="--",
        color="0.45",
        linewidth=1.2,
        zorder=2
    )

    ax.annotate(
        r"$K_1$ i $K_3$ no són comparables",
        xy=((K1[0] + K3[0]) / 2, (K1[1] + K3[1]) / 2),
        xytext=(0.9, 9.2),
        arrowprops=dict(arrowstyle="->", lw=1.0, color="0.35"),
        fontsize=11,
        color="0.25"
    )
    # -----------------------------
    # Configuració dels eixos
    # -----------------------------
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.set_xlabel(r"Cost $C(K)$", fontsize=12)
    ax.set_ylabel(r"Servei $P(K)$", fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    fig.savefig("results/dominancia_pareto.png", dpi=300, bbox_inches="tight")
    fig.savefig("results/dominancia_pareto.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Figures generades correctament:")
    print(" - results/dominancia_pareto.png")
    print(" - results/dominancia_pareto.pdf")


if __name__ == "__main__":
    main()
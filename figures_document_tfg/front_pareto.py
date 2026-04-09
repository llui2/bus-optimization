import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    os.makedirs("results", exist_ok=True)

    granate = "#7A0019"
    gris = "0.55"
    gris_suau = "0.82"

    # -----------------------------
    # Solucions candidates
    # Eix X: cost (minimitzar)
    # Eix Y: servei (maximitzar)
    # -----------------------------
    dominated = np.array([
        [2.4, 4.8],
        [3.2, 5.1],
        [3.8, 6.0],
        [4.7, 5.7],
        [5.1, 6.6],
        [5.8, 6.2],
        [6.4, 7.0],
        [7.1, 6.8],
        [7.6, 7.6]
    ])

    pareto = np.array([
        [2.0, 5.6],
        [2.8, 6.4],
        [3.7, 7.1],
        [4.8, 7.8],
        [6.0, 8.5],
        [7.2, 9.1]
    ])

    fig, ax = plt.subplots(figsize=(7.4, 5.8))

    # Línies auxiliars molt subtils
    ax.grid(False)

    # Solucions dominades
    ax.scatter(
        dominated[:, 0], dominated[:, 1],
        s=85,
        color="0.72",
        edgecolors="white",
        linewidths=0.8,
        zorder=2,
        label="Solucions dominades"
    )

    # Solucions no dominades
    ax.scatter(
        pareto[:, 0], pareto[:, 1],
        s=115,
        color=granate,
        edgecolors="white",
        linewidths=0.9,
        zorder=4,
        label="Solucions no dominades"
    )

    # Front de Pareto
    ax.plot(
        pareto[:, 0], pareto[:, 1],
        linestyle="--",
        linewidth=1.6,
        color=granate,
        zorder=3
    )

    # Etiquetes d'alguns punts del front
    labels = [r"$K_a$", r"$K_b$", r"$K_c$", r"$K_d$", r"$K_e$", r"$K_f$"]
    offsets = [(0.10, 0.18), (0.10, 0.18), (0.10, 0.18), (0.10, 0.18), (0.10, 0.18), (0.10, 0.18)]
    for (x, y), lab, (dx, dy) in zip(pareto, labels, offsets):
        ax.text(x + dx, y + dy, lab, fontsize=11)

    # Anotacions generals
    ax.annotate(
        "Front de Pareto",
        xy=(5.4, 8.2),
        xytext=(5.8, 9.45),
        arrowprops=dict(arrowstyle="->", lw=1.1, color=granate),
        fontsize=11,
        color=granate
    )

    ax.annotate(
        "Solucions dominades",
        xy=(5.8, 6.2),
        xytext=(7.0, 5.2),
        arrowprops=dict(arrowstyle="->", lw=1.0, color="0.35"),
        fontsize=11,
        color="0.25"
    )

    # Eixos
    ax.set_xlim(1.2, 8.4)
    ax.set_ylim(4.2, 9.8)

    ax.set_xlabel(r"Cost $C(K)$", fontsize=12)
    ax.set_ylabel(r"Servei $P(K)$", fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    fig.savefig("results/front_pareto.png", dpi=300, bbox_inches="tight")
    fig.savefig("results/front_pareto.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Figures generades correctament:")
    print(" - results/front_pareto.png")
    print(" - results/front_pareto.pdf")

if __name__ == "__main__":
    main()
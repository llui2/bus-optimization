import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    os.makedirs("results", exist_ok=True)

    granate = "#7A0019"
    gris = "0.72"
    gris_fosc = "0.35"
    gris_suau = "0.85"

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

    # Tres solucions representatives del front
    K_a = pareto[0]   # cost baix, servei moderat
    K_b = pareto[3]   # compromís intermedi
    K_c = pareto[5]   # servei alt, cost alt

    fig, ax = plt.subplots(figsize=(7.4, 5.8))

    # Solucions dominades
    ax.scatter(
        dominated[:, 0], dominated[:, 1],
        s=80,
        color=gris,
        edgecolors="white",
        linewidths=0.8,
        zorder=1
    )

    # Front complet amb menys protagonisme
    ax.plot(
        pareto[:, 0], pareto[:, 1],
        linestyle="--",
        linewidth=1.5,
        color=granate,
        alpha=0.75,
        zorder=2
    )

    ax.scatter(
        pareto[:, 0], pareto[:, 1],
        s=70,
        color=granate,
        edgecolors="white",
        linewidths=0.8,
        alpha=0.65,
        zorder=3
    )

    # Ressaltar 3 punts clau
    selected = np.array([K_a, K_b, K_c])
    ax.scatter(
        selected[:, 0], selected[:, 1],
        s=150,
        color=granate,
        edgecolors="white",
        linewidths=1.0,
        zorder=4
    )

    # Etiquetes dels punts
    ax.text(K_a[0] - 0.28, K_a[1] + 0.22, r"$K_{\mathrm{A}}$", fontsize=12)
    ax.text(K_b[0] + 0.10, K_b[1] + 0.20, r"$K_{\mathrm{B}}$", fontsize=12)
    ax.text(K_c[0] + 0.10, K_c[1] + 0.10, r"$K_{\mathrm{C}}$", fontsize=12)

    # Anotacions explicatives
    ax.annotate(
        "Cost baix\nServei moderat",
        xy=K_a,
        xytext=(1.55, 7.25),
        arrowprops=dict(arrowstyle="->", lw=1.0, color=gris_fosc),
        fontsize=10.5,
        color=gris_fosc,
        ha="left"
    )

    ax.annotate(
        "Compromís\nintermedi",
        xy=K_b,
        xytext=(4.15, 9.1),
        arrowprops=dict(arrowstyle="->", lw=1.0, color=gris_fosc),
        fontsize=10.5,
        color=gris_fosc,
        ha="center"
    )

    ax.annotate(
        "Servei alt\nCost elevat",
        xy=K_c,
        xytext=(7.55, 8.15),
        arrowprops=dict(arrowstyle="->", lw=1.0, color=gris_fosc),
        fontsize=10.5,
        color=gris_fosc,
        ha="left"
    )

    # Indicació del sentit del compromís
    ax.annotate(
        "Augmentar el servei\nimplica assumir més cost",
        xy=(5.2, 8.0),
        xytext=(2.55, 9.55),
        arrowprops=dict(arrowstyle="->", lw=1.1, color=granate),
        fontsize=10.5,
        color=granate,
        ha="left"
    )

    # Eixos
    ax.set_xlim(1.2, 8.4)
    ax.set_ylim(4.2, 10.0)

    ax.set_xlabel(r"Cost $C(K)$", fontsize=12)
    ax.set_ylabel(r"Servei $P(K)$", fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    fig.savefig("results/tradeoff_pareto.png", dpi=300, bbox_inches="tight")
    fig.savefig("results/tradeoff_pareto.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Figures generades correctament:")
    print(" - results/tradeoff_pareto.png")
    print(" - results/tradeoff_pareto.pdf")

if __name__ == "__main__":
    main()
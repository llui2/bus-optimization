import os
import random
import matplotlib.pyplot as plt


def build_grid_graph(n=10):
    V = []
    E = []

    for i in range(n):
        for j in range(n):
            V.append((i, j))

    for i in range(n):
        for j in range(n):
            if i + 1 < n:
                E.append(((i, j), (i + 1, j)))
            if j + 1 < n:
                E.append(((i, j), (i, j + 1)))

    return V, E


def choose_bus_stops_spread(V, n_stops=16, seed=7):
    rng = random.Random(seed)
    chosen = []

    if n_stops > 0 and len(V) > 0:
        first = rng.choice(V)
        chosen.append(first)

        while len(chosen) < min(n_stops, len(V)):
            best_p = None
            best_score = -1.0

            for p in V:
                if p in chosen:
                    continue

                dmin = None
                for c in chosen:
                    dx = p[0] - c[0]
                    dy = p[1] - c[1]
                    d2 = dx * dx + dy * dy

                    if dmin is None or d2 < dmin:
                        dmin = d2

                if dmin is not None and dmin > best_score:
                    best_score = dmin
                    best_p = p

            if best_p is not None:
                chosen.append(best_p)
            else:
                break

    return set(chosen)


def generate_positions(V, shift=0.0, seed=7):
    rng = random.Random(seed)
    pos = {}

    for (i, j) in V:
        if shift == 0:
            x = float(i)
            y = float(j)
        else:
            x = i + rng.uniform(-shift, shift)
            y = j + rng.uniform(-shift, shift)

        pos[(i, j)] = (x, y)

    return pos


def plot_network(ax, V, E, pos, bus_stops, title=None):
    granate = "#7A0019"

    # Arestes / carreteres
    for (u, v) in E:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2],
                color="0.72",
                linewidth=0.9,
                alpha=0.85,
                zorder=1)

    # Nodes normals (opcionales, molt subtils)
    x_all = [pos[p][0] for p in V]
    y_all = [pos[p][1] for p in V]
    ax.scatter(x_all, y_all,
               s=10,
               color="0.82",
               edgecolors="none",
               zorder=2)

    # Parades de bus
    xb = [pos[p][0] for p in bus_stops]
    yb = [pos[p][1] for p in bus_stops]
    ax.scatter(xb, yb,
               s=60,
               color=granate,
               edgecolors="white",
               linewidths=0.6,
               zorder=3)

    if title is not None:
        ax.set_title(title, fontsize=12)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def save_single_figure(V, E, pos, bus_stops, out_png, out_pdf, title=None):
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    plot_network(ax, V, E, pos, bus_stops, title=title)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def save_comparison_figure(V, E, pos0, pos1, bus_stops, out_png, out_pdf):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    plot_network(
        axes[0], V, E, pos0, bus_stops,
        title=r"shift=0"
    )
    plot_network(
        axes[1], V, E, pos1, bus_stops,
        title=r"shift>0"
    )

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs("results", exist_ok=True)

    n = 10
    seed_nodes = 7
    seed_stops = 11
    n_stops = 16

    # Pots canviar aquest valor per fer més o menys visible el soroll
    shift_positive = 0.22

    V, E = build_grid_graph(n=n)

    # Si vols evitar parades massa al marge:
    V_inner = []
    for (i, j) in V:
        if 1 <= i <= n - 2 and 1 <= j <= n - 2:
            V_inner.append((i, j))

    bus_stops = choose_bus_stops_spread(V_inner, n_stops=n_stops, seed=seed_stops)

    pos_shift_0 = generate_positions(V, shift=0.0, seed=seed_nodes)
    pos_shift_pos = generate_positions(V, shift=shift_positive, seed=seed_nodes)

    save_single_figure(
        V, E, pos_shift_0, bus_stops,
        out_png="results/graf_shift_0.png",
        out_pdf="results/graf_shift_0.pdf",
        title=None
    )

    save_single_figure(
        V, E, pos_shift_pos, bus_stops,
        out_png="results/graf_shift_pos.png",
        out_pdf="results/graf_shift_pos.pdf",
        title=None
    )

    save_comparison_figure(
        V, E, pos_shift_0, pos_shift_pos, bus_stops,
        out_png="results/graf_shift_comparacio.png",
        out_pdf="results/graf_shift_comparacio.pdf"
    )

    print("Figures generades correctament:")
    print(" - results/graf_shift_0.png / .pdf")
    print(" - results/graf_shift_pos.png / .pdf")
    print(" - results/graf_shift_comparacio.png / .pdf")


if __name__ == "__main__":
    main()
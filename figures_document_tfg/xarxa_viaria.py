import random
import matplotlib.pyplot as plt
import os, json

def build_grid_graph(n=20):
    V = [(i, j) for i in range(n) for j in range(n)]
    E = []
    for i in range(n):
        for j in range(n):
            if i + 1 < n:
                E.append(((i, j), (i + 1, j)))
            if j + 1 < n:
                E.append(((i, j), (i, j + 1)))
    return V, E


def sqdist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def choose_bus_stops_spread(V, n_stops=22, seed=7):
    rng = random.Random(seed)

    if n_stops <= 0:
        return set()

    first = rng.choice(V)
    chosen = [first]

    while len(chosen) < min(n_stops, len(V)):
        best_p = None
        best_score = -1

        for p in V:
            if p in chosen:
                continue

            dmin = min(sqdist(p, c) for c in chosen)
            if dmin > best_score:
                best_score = dmin
                best_p = p

        if best_p is None:
            break

        chosen.append(best_p)

    return set(chosen)


def jittered_polyline(u, v, rng, amp=0.14, kinks=2):
    x1, y1 = u
    x2, y2 = v

    pts = [(x1, y1)]
    for t in range(1, kinks + 1):
        alpha = t / (kinks + 1)

        xb = x1 + alpha * (x2 - x1)
        yb = y1 + alpha * (y2 - y1)

        dx = x2 - x1
        dy = y2 - y1
        px, py = -dy, dx

        j_perp = (rng.uniform(-amp, amp), rng.uniform(-amp, amp))
        j_free = (rng.uniform(-amp, amp) * 0.5, rng.uniform(-amp, amp) * 0.5)

        xj = xb + px * j_perp[0] + j_free[0]
        yj = yb + py * j_perp[1] + j_free[1]
        pts.append((xj, yj))

    pts.append((x2, y2))
    return pts


def plot_noisy_disconnected_network(V, E, V_B,
                                    out_png="road_network_noisy.png",
                                    out_pdf="road_network_noisy.pdf",
                                    seed=7,
                                    drop_edge_prob=0.18,
                                    noise_amp=0.14,
                                    kinks=2):
    rng = random.Random(seed)
    fig, ax = plt.subplots()

    # Arestes: eliminació aleatòria per generar talls
    E_kept = []
    for e in E:
        if rng.random() >= drop_edge_prob:
            E_kept.append(e)

    # Carreteres: gris + més subtil (alpha + linewidth)
    for (u, v) in E_kept:
        pts = jittered_polyline(u, v, rng, amp=noise_amp, kinks=kinks)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color="0.72", linewidth=0.8, alpha=0.75)

    # Parades: granate
    granate = "#7A0019"
    xb = [p[0] for p in V_B]
    yb = [p[1] for p in V_B]
    #ax.scatter(xb, yb, s=70, color=granate, edgecolors="white", linewidths=0.6, zorder=3)
    ax.scatter(xb, yb, s=60, color=granate, edgecolors="white", linewidths=0.6, zorder=3)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return E_kept


def main():
    os.makedirs("results", exist_ok=True)
    seed = 7
    n = 20
    B = 22

    drop_edge_prob = 0.18
    noise_amp = 0.14
    kinks = 2

    V, E = build_grid_graph(n=n)

    V_inner = [(i, j) for (i, j) in V if 1 <= i <= n - 2 and 1 <= j <= n - 2]
    V_B = choose_bus_stops_spread(V_inner, n_stops=B, seed=seed)

    E_kept = plot_noisy_disconnected_network(
        V, E, V_B,
        out_png="results/road_network.png",
        out_pdf="results/road_network.pdf",
        seed=seed,
        drop_edge_prob=drop_edge_prob,
        noise_amp=noise_amp,
        kinks=kinks
    )

    data = {"n": n,
            "seed": seed,
            "V": V,
            "E_roads": E_kept,
            "V_B": list(V_B)  # IMPORTANT
            }
    with open("results/base_map.json", "w", encoding="utf-8") as f:
        json.dump(data, f)

if __name__ == "__main__":
    main()
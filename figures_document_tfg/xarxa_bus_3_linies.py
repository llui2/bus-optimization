import os
import json
import random
import matplotlib.pyplot as plt
from collections import deque


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
        j_free = (rng.uniform(-amp, amp) * 0.35, rng.uniform(-amp, amp) * 0.35)

        xj = xb + px * j_perp[0] + j_free[0]
        yj = yb + py * j_perp[1] + j_free[1]
        pts.append((xj, yj))

    pts.append((x2, y2))
    return pts


def build_adjacency(E):
    adj = {}
    for u, v in E:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    return adj


def norm_edge(a, b):
    return (a, b) if a <= b else (b, a)


def bfs_shortest_path_avoiding(adj, start, goal, banned_edges):
    if start == goal:
        return [start]

    q = deque([start])
    parent = {start: None}

    found = False
    while q and not found:
        u = q.popleft()
        for v in adj.get(u, []):
            if norm_edge(u, v) in banned_edges:
                continue
            if v not in parent:
                parent[v] = u
                if v == goal:
                    found = True
                    break
                q.append(v)

    if goal not in parent:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def edges_from_path(path):
    edges = []
    if path is not None:
        for i in range(len(path) - 1):
            edges.append(norm_edge(path[i], path[i + 1]))
    return edges


def pick_node_in_region(V, rng, x_min, x_max, y_min, y_max):
    candidates = [(x, y) for (x, y) in V if x_min <= x <= x_max and y_min <= y <= y_max]
    if len(candidates) == 0:
        return None
    return rng.choice(candidates)


def build_one_line(adj, V_inner, rng, region_a, region_b, banned_edges, hub, max_tries=200):
    tries = 0
    path = None
    a = None
    b = None

    while tries < max_tries and path is None:
        a = pick_node_in_region(V_inner, rng, *region_a)
        b = pick_node_in_region(V_inner, rng, *region_b)
        if a is not None and b is not None:
            p1 = bfs_shortest_path_avoiding(adj, a, hub, banned_edges)
            p2 = bfs_shortest_path_avoiding(adj, hub, b, banned_edges)

            if p1 is not None and p2 is not None:
                # concatena evitant duplicar el hub
                path = p1 + p2[1:]
        tries += 1

    # fallback sense evitar (si hi ha molts talls)
    if path is None and a is not None and b is not None:
        p1 = bfs_shortest_path_avoiding(adj, a, hub, set())
        p2 = bfs_shortest_path_avoiding(adj, hub, b, set())
        if p1 is not None and p2 is not None:
            path = p1 + p2[1:]

    return a, b, path


def plot_map_and_3_lines(E_roads, V_B, lines, out_png, out_pdf, rng, noise_amp=0.14, kinks=2):
    fig, ax = plt.subplots()

    # 1) Guardem el traçat (amb soroll) de cada aresta del mapa gris
    edge_poly = {}

    # Mapa base (gris)
    for (u, v) in E_roads:
        e = norm_edge(u, v)
        pts = jittered_polyline(u, v, rng, amp=noise_amp, kinks=kinks)
        edge_poly[e] = pts

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color="0.72", linewidth=0.8, alpha=0.75, zorder=1)

    # Totes les parades (V_B) en granat
    granate = "#7A0019"
    xb = [p[0] for p in V_B]
    yb = [p[1] for p in V_B]
    ax.scatter(xb, yb, s=60, color=granate, edgecolors="white", linewidths=0.6, zorder=2)

    # 2) Línies (colors) REUTILITZANT el mateix traçat que el mapa gris
    for line in lines:
        color = line["color"]
        for (u, v) in line["edges"]:
            e = norm_edge(u, v)
            pts = edge_poly.get(e)

            # fallback (no hauria de passar)
            if pts is None:
                pts = [u, v]

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=color, linewidth=2.6, alpha=0.95, zorder=3)

        # Parades d’inici/final de cada línia amb el mateix color
        stops = line["stops"]
        xs = [p[0] for p in stops]
        ys = [p[1] for p in stops]
        ax.scatter(xs, ys, s=90, color=color, edgecolors="white", linewidths=0.9, zorder=4)

    # Llegenda
    handles = []
    labels = []
    for line in lines:
        h = ax.plot([], [], color=line["color"], linewidth=3)[0]
        handles.append(h)
        labels.append(line["name"])
    #ax.legend(handles, labels, frameon=False, loc="upper right")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    #plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    os.makedirs("results", exist_ok=True)

    # Carrega mapa base (G)
    with open("results/base_map.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    n = data["n"]
    seed = data["seed"]

    rng = random.Random(seed)

    V = [tuple(x) for x in data["V"]]
    V_B = [tuple(x) for x in data["V_B"]]

    hub = rng.choice(V_B)

    E_roads_raw = data["E_roads"]
    if E_roads_raw is None:
        raise ValueError("E_roads és null. Torna a generar base_map.json després d'afegir return E_kept.")

    E_roads = [(tuple(a), tuple(b)) for a, b in E_roads_raw]
    adj = build_adjacency(E_roads)

    # Evitem la vora per parades
    V_inner = [(i, j) for (i, j) in V if 1 <= i <= n - 2 and 1 <= j <= n - 2]

    # 3 línies ben separades (zones diferents)
    regions = [
        ((2, 4, n - 4, n - 2), (n - 4, n - 2, n - 4, n - 2)),  # dalt esq -> dalt dreta
        ((2, 4, 2, 4), (n - 4, n - 2, 2, 4)),                  # baix esq -> baix dreta
        ((2, 4, n // 2 - 2, n // 2 + 2), (n - 4, n - 2, n // 2 - 2, n // 2 + 2)),  # mig esq -> mig dreta
    ]

    colors = ["#1f77b4", "#2ca02c", "#7A0019"]  # blau, verd, granat
    names = ["Línia 1", "Línia 2", "Línia 3"]

    lines = []
    banned_edges = set()

    for i in range(3):
        a, b, path = build_one_line(adj, V_inner, rng, regions[i][0], regions[i][1], banned_edges, hub)
        edges = edges_from_path(path)

        # Evita superposició entre línies
        for e in edges:
            banned_edges.add(e)

        line = {
            "name": names[i],
            "color": colors[i],
            "stops": [a, b] if a is not None and b is not None else [],
            "edges": edges
        }
        lines.append(line)

    noise_amp = 0.14
    kinks = 2

    plot_map_and_3_lines(
        E_roads=E_roads,
        V_B=V_B,
        lines=lines,
        out_png="results/bus_network_3_lines.png",
        out_pdf="results/bus_network_3_lines.pdf",
        rng=rng,
        noise_amp=noise_amp,
        kinks=kinks
    )


if __name__ == "__main__":
    main()
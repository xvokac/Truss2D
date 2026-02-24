import numpy as np
import matplotlib.pyplot as plt
import json
from truss_solver import solve_truss


def load_model(filename):
    with open(filename) as f:
        return json.load(f)


def plot_truss(filename, scale_load=0.2):
    model = load_model(filename)
    u, R, N = solve_truss(filename)

    nodes = np.array(model["nodes"])
    members = model["members"]

    fig, ax = plt.subplots(figsize=(8, 6))

    # ======================
    # Draw members
    # ======================
    for i, (n1, n2) in enumerate(members):
        x = [nodes[n1, 0], nodes[n2, 0]]
        y = [nodes[n1, 1], nodes[n2, 1]]

        force = N[i]

        if force > 0:
            color = "red"       # tension
        else:
            color = "blue"      # compression

        ax.plot(x, y, color=color, linewidth=2)

        # label force at midpoint
        xm = np.mean(x)
        ym = np.mean(y)
        ax.text(xm, ym, f"{force:.2f}", fontsize=9,
                ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    # ======================
    # Draw nodes
    # ======================
    ax.scatter(nodes[:, 0], nodes[:, 1], color="black", zorder=5)

    for i, (x, y) in enumerate(nodes):
        ax.text(x, y, f" {i}", fontsize=10, color="black")

    # ======================
    # Draw supports
    # ======================
    for sup in model.get("supports", []):
        node = sup["node"]
        fix_x, fix_y = sup["fix"]

        x, y = nodes[node]

        if fix_x and fix_y:
            ax.plot(x, y, marker="s", markersize=10, color="green")
        elif fix_y:
            ax.plot(x, y, marker="^", markersize=10, color="green")
        elif fix_x:
            ax.plot(x, y, marker=">", markersize=10, color="green")

    # ======================
    # Draw loads
    # ======================
    for load in model.get("loads", []):
        node = load["node"]
        fx = load.get("fx", 0.0)
        fy = load.get("fy", 0.0)

        x, y = nodes[node]

        ax.arrow(x, y,
                 fx * scale_load,
                 fy * scale_load,
                 head_width=0.1,
                 length_includes_head=True,
                 color="magenta")

    # ======================
    # Formatting
    # ======================
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Truss analysis")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.show()


if __name__ == "__main__":
    plot_truss("model.json")

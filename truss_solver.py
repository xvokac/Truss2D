####################
## truss_solver.py


import json
import numpy as np


class TrussError(Exception):
    pass


def load_model(filename):
    with open(filename) as f:
        return json.load(f)


def check_model(model):
    nodes = model["nodes"]
    members = model["members"]

    if len(nodes) == 0:
        raise TrussError("No nodes defined.")

    if len(members) == 0:
        raise TrussError("No members defined.")

    # zero length check
    for i, (n1, n2) in enumerate(members):
        p1 = np.array(nodes[n1])
        p2 = np.array(nodes[n2])
        if np.linalg.norm(p2 - p1) == 0:
            raise TrussError(f"Member {i} has zero length.")


def assemble_global_stiffness(nodes, members):
    n_nodes = len(nodes)
    ndof = 2 * n_nodes
    K = np.zeros((ndof, ndof))

    for (n1, n2) in members:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]

        dx = x2 - x1
        dy = y2 - y1
        L = np.sqrt(dx**2 + dy**2)

        c = dx / L
        s = dy / L

        k_local = (1 / L) * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])

        dofs = [
            2*n1, 2*n1+1,
            2*n2, 2*n2+1
        ]

        for i in range(4):
            for j in range(4):
                K[dofs[i], dofs[j]] += k_local[i, j]

    return K


def assemble_load_vector(model):
    n_nodes = len(model["nodes"])
    F = np.zeros(2 * n_nodes)

    for load in model.get("loads", []):
        node = load["node"]
        F[2*node] += load.get("fx", 0.0)
        F[2*node+1] += load.get("fy", 0.0)

    return F


def apply_supports(K, F, supports):
    ndof = len(F)
    fixed = []

    for sup in supports:
        node = sup["node"]
        fix = sup["fix"]

        if fix[0]:
            fixed.append(2*node)
        if fix[1]:
            fixed.append(2*node + 1)

    free = np.setdiff1d(np.arange(ndof), fixed)

    Kff = K[np.ix_(free, free)]
    Ff = F[free]

    return Kff, Ff, free, fixed


def solve_displacements(Kff, Ff):
    if np.linalg.matrix_rank(Kff) < Kff.shape[0]:
        raise TrussError("Structure is unstable (singular stiffness matrix).")

    return np.linalg.solve(Kff, Ff)


def compute_reactions(K, u, F):
    return K @ u - F


def compute_member_forces(nodes, members, u):
    forces = []

    for (n1, n2) in members:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]

        dx = x2 - x1
        dy = y2 - y1
        L = np.sqrt(dx**2 + dy**2)

        c = dx / L
        s = dy / L

        u_e = np.array([
            u[2*n1], u[2*n1+1],
            u[2*n2], u[2*n2+1]
        ])

        N = (1 / L) * np.array([-c, -s, c, s]) @ u_e
        forces.append(N)

    return np.array(forces)


def solve_truss(filename):
    model = load_model(filename)
    check_model(model)

    nodes = model["nodes"]
    members = model["members"]

    K = assemble_global_stiffness(nodes, members)
    F = assemble_load_vector(model)

    Kff, Ff, free, fixed = apply_supports(K, F, model["supports"])

    uf = solve_displacements(Kff, Ff)

    u = np.zeros(len(F))
    u[free] = uf

    reactions = compute_reactions(K, u, F)
    member_forces = compute_member_forces(nodes, members, u)

    return u, reactions, member_forces


if __name__ == "__main__":
    u, R, N = solve_truss("model.json")

    print("\n=== Member forces ===")
    for i, n in enumerate(N):
        state = "Tension" if n > 0 else "Compression"
        print(f"Member {i:3d}: {n:12.4f}   {state}")


    print("\n=== Reactions ===")
    for i, r in enumerate(R):
        if abs(r) > 1e-8:
            node = i // 2
            direction = "X" if i % 2 == 0 else "Y"
            print(f"DOF {i:3d} (node {node}+{direction}): {r:12.4f}")



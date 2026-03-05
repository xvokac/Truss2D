####################
## truss_solver.py


import json
import numpy as np


class TrussError(Exception):
    pass


def load_model(filename):
    with open(filename) as f:
        return json.load(f)


def normalize_model(model):
    """Return model with contiguous 0..N-1 node indices.

    Supported input formats:
    - nodes as [[x, y], ...] and indices already matching list positions
    - nodes as [{"id": <node_id>, "x": <x>, "y": <y>}, ...] with arbitrary IDs
    """
    nodes_raw = model.get("nodes", [])
    if not nodes_raw:
        return {
            "nodes": [],
            "members": [],
            "supports": [],
            "loads": [],
        }, {}

    first_node = nodes_raw[0]
    id_to_new = {}
    nodes = []

    if isinstance(first_node, dict):
        for i, node in enumerate(nodes_raw):
            if "id" not in node or "x" not in node or "y" not in node:
                raise TrussError(
                    f"Node at index {i} must contain keys: id, x, y."
                )
            original_id = int(node["id"])
            if original_id in id_to_new:
                raise TrussError(f"Duplicate node id {original_id}.")
            id_to_new[original_id] = len(nodes)
            nodes.append([float(node["x"]), float(node["y"])])
    else:
        nodes = [[float(x), float(y)] for x, y in nodes_raw]
        id_to_new = {i: i for i in range(len(nodes))}

    def remap_node(node_id):
        key = int(node_id)
        if key not in id_to_new:
            raise TrussError(f"Unknown node id {node_id}.")
        return id_to_new[key]

    members = []
    for i, (n1, n2) in enumerate(model.get("members", [])):
        try:
            members.append([remap_node(n1), remap_node(n2)])
        except TrussError as e:
            raise TrussError(f"Invalid member {i}: {e}") from e

    supports = []
    for i, sup in enumerate(model.get("supports", [])):
        try:
            supports.append({
                "node": remap_node(sup["node"]),
                "fix": [bool(sup["fix"][0]), bool(sup["fix"][1])],
            })
        except (KeyError, IndexError, TypeError) as e:
            raise TrussError(f"Invalid support {i} format.") from e
        except TrussError as e:
            raise TrussError(f"Invalid support {i}: {e}") from e

    loads = []
    for i, load in enumerate(model.get("loads", [])):
        try:
            loads.append({
                "node": remap_node(load["node"]),
                "fx": float(load.get("fx", 0.0)),
                "fy": float(load.get("fy", 0.0)),
            })
        except (KeyError, TypeError, ValueError) as e:
            raise TrussError(f"Invalid load {i} format.") from e
        except TrussError as e:
            raise TrussError(f"Invalid load {i}: {e}") from e

    return {
        "nodes": nodes,
        "members": members,
        "supports": supports,
        "loads": loads,
    }, id_to_new


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


def solve_truss(filename_or_model):
    if isinstance(filename_or_model, str):
        model = load_model(filename_or_model)
    else:
        model = filename_or_model

    model, _ = normalize_model(model)
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



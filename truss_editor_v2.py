import sys
import json
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox, QDialog, QLabel,
    QLineEdit, QCheckBox, QDialogButtonBox, QTableWidget, QTableWidgetItem
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

from truss_solver import assemble_global_stiffness, solve_displacements, compute_member_forces


# ================= MODEL =================

class TrussModel:
    def __init__(self):
        self.nodes = []
        self.members = []
        self.supports = []
        self.loads = []

    def to_dict(self):
        return self.__dict__

    def from_dict(self, d):
        self.nodes = d["nodes"]
        self.members = d["members"]
        self.supports = d["supports"]
        self.loads = d["loads"]


# ================= DIALOGS =================

class NodeDialog(QDialog):
    def __init__(self, x, y):
        super().__init__()
        self.setWindowTitle("Node")

        layout = QVBoxLayout(self)
        self.x = QLineEdit(str(round(x, 3)))
        self.y = QLineEdit(str(round(y, 3)))

        layout.addWidget(QLabel("X"))
        layout.addWidget(self.x)
        layout.addWidget(QLabel("Y"))
        layout.addWidget(self.y)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)


class SupportDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Support")

        layout = QVBoxLayout(self)
        self.fx = QCheckBox("Fix X")
        self.fy = QCheckBox("Fix Y")

        layout.addWidget(self.fx)
        layout.addWidget(self.fy)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)


class LoadDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Load")

        layout = QVBoxLayout(self)
        self.fx = QLineEdit("0")
        self.fy = QLineEdit("0")

        layout.addWidget(QLabel("Fx"))
        layout.addWidget(self.fx)
        layout.addWidget(QLabel("Fy"))
        layout.addWidget(self.fy)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)


class ViewScaleDialog(QDialog):
    def __init__(self, xlim, ylim, load_arrow_scale):
        super().__init__()
        self.setWindowTitle("View / Scale")

        layout = QVBoxLayout(self)
        self.xmin = QLineEdit(str(round(xlim[0], 3)))
        self.xmax = QLineEdit(str(round(xlim[1], 3)))
        self.ymin = QLineEdit(str(round(ylim[0], 3)))
        self.ymax = QLineEdit(str(round(ylim[1], 3)))
        self.load_scale = QLineEdit(str(round(load_arrow_scale, 3)))

        layout.addWidget(QLabel("X min"))
        layout.addWidget(self.xmin)
        layout.addWidget(QLabel("X max"))
        layout.addWidget(self.xmax)
        layout.addWidget(QLabel("Y min"))
        layout.addWidget(self.ymin)
        layout.addWidget(QLabel("Y max"))
        layout.addWidget(self.ymax)
        layout.addWidget(QLabel("Load arrow scale (same for Fx and Fy)"))
        layout.addWidget(self.load_scale)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)


# ================= CANVAS =================

class TrussCanvas(Canvas):
    def __init__(self, editor):
        self.fig = Figure(figsize=(6, 6))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.editor = editor

        self.default_xlim = (-20, 20)
        self.default_ylim = (-20, 20)
        self.default_load_arrow_scale = 0.2

        self.ax.set_xlim(*self.default_xlim)
        self.ax.set_ylim(*self.default_ylim)
        self.load_arrow_scale = self.default_load_arrow_scale
        self.zoom_step = 0.15

        self.mpl_connect("button_press_event", self.click)
        self.mpl_connect("scroll_event", self.on_scroll)

    def redraw(self, forces=None):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.clear()

        m = self.editor.model

        # members
        for i, (n1, n2) in enumerate(m.members):
            x = [m.nodes[n1][0], m.nodes[n2][0]]
            y = [m.nodes[n1][1], m.nodes[n2][1]]

            if forces is None:
                col = "black"
            else:
                col = "red" if forces[i] > 0 else "blue"

            self.ax.plot(x, y, color=col, lw=2)
            mx, my = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2
            self.ax.text(mx, my, f"m{i}", color="darkgreen", fontsize=9)

        # nodes
        if m.nodes:
            pts = np.array(m.nodes)
            self.ax.scatter(pts[:, 0], pts[:, 1], color="black")
            for i, (x, y) in enumerate(m.nodes):
                self.ax.text(x, y, f" {i}")

        # supports
        for s in m.supports:
            x, y = m.nodes[s["node"]]
            fx, fy = s["fix"]
            if fx and fy:
                self.ax.plot(x, y, "gs", ms=10)
            elif fy:
                self.ax.plot(x, y, "g^", ms=10)
            elif fx:
                self.ax.plot(x, y, "g>", ms=10)

        # loads
        for l in m.loads:
            x, y = m.nodes[l["node"]]
            dx, dy = l["fx"] * self.load_arrow_scale, l["fy"] * self.load_arrow_scale
            self.ax.arrow(x, y, dx, dy,
                          color="magenta", head_width=0.3)
            self.ax.text(x + dx, y + dy, f"[{l['fx']:.2f}, {l['fy']:.2f}]",
                         color="magenta", fontsize=9)

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_aspect("equal")
        self.ax.grid(True)
        self.draw()

    def reset_view_scale(self):
        self.load_arrow_scale = self.default_load_arrow_scale
        self.ax.set_xlim(*self.default_xlim)
        self.ax.set_ylim(*self.default_ylim)
        self.redraw()

    def on_scroll(self, e):
        if e.inaxes != self.ax or e.xdata is None or e.ydata is None:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        if e.button == "up":
            factor = 1 - self.zoom_step
        elif e.button == "down":
            factor = 1 + self.zoom_step
        else:
            return

        new_x_range = x_range * factor
        new_y_range = y_range * factor

        relx = (e.xdata - xlim[0]) / x_range if x_range else 0.5
        rely = (e.ydata - ylim[0]) / y_range if y_range else 0.5

        xmin = e.xdata - new_x_range * relx
        xmax = e.xdata + new_x_range * (1 - relx)
        ymin = e.ydata - new_y_range * rely
        ymax = e.ydata + new_y_range * (1 - rely)

        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.redraw()

    # ---------- click logic ----------
    def click(self, e):
        if e.inaxes is None:
            return

        x, y = e.xdata, e.ydata
        ed = self.editor
        m = ed.model

        if ed.mode == "node":
            dlg = NodeDialog(x, y)
            if dlg.exec_():
                m.nodes.append([float(dlg.x.text()), float(dlg.y.text())])
                self.redraw()
                ed.statusBar().showMessage(
                    f"Node {len(m.nodes) - 1} added.",
                    4000,
                )

        elif ed.mode == "member":
            idx = ed.find_node(x, y)
            if idx is None:
                ed.statusBar().showMessage(
                    "Member mode: click an existing node.",
                    4000,
                )
                return
            ed.sel.append(idx)
            if len(ed.sel) == 2:
                m.members.append(ed.sel.copy())
                ed.sel.clear()
                self.redraw()
                ed.statusBar().showMessage("Member added.", 4000)
            else:
                ed.statusBar().showMessage("First node selected. Click second node.")

        elif ed.mode == "support":
            idx = ed.find_node(x, y)
            if idx is None:
                ed.statusBar().showMessage(
                    "Support mode: click an existing node.",
                    4000,
                )
                return
            dlg = SupportDialog()
            if dlg.exec_():
                m.supports.append({"node": idx,
                                   "fix": [dlg.fx.isChecked(), dlg.fy.isChecked()]})
                self.redraw()
                ed.statusBar().showMessage(f"Support set at node {idx}.", 4000)

        elif ed.mode == "load":
            idx = ed.find_node(x, y)
            if idx is None:
                ed.statusBar().showMessage(
                    "Load mode: click an existing node.",
                    4000,
                )
                return
            dlg = LoadDialog()
            if dlg.exec_():
                m.loads.append({"node": idx,
                                "fx": float(dlg.fx.text()),
                                "fy": float(dlg.fy.text())})
                self.redraw()
                ed.statusBar().showMessage(f"Load set at node {idx}.", 4000)

        elif ed.mode == "delete":
            # delete member first
            for i, (n1, n2) in enumerate(m.members):
                p1, p2 = np.array(m.nodes[n1]), np.array(m.nodes[n2])
                if np.linalg.norm(np.cross(p2-p1, p1-[x,y]))/np.linalg.norm(p2-p1) < 0.3:
                    m.members.pop(i)
                    self.redraw()
                    ed.statusBar().showMessage(f"Member {i} deleted.", 4000)
                    return

            # delete node
            idx = ed.find_node(x, y)
            if idx is not None:
                m.nodes.pop(idx)
                # remove members connected to deleted node and reindex the rest
                updated_members = []
                for n1, n2 in m.members:
                    if idx in (n1, n2):
                        continue

                    updated_members.append([
                        n1 - 1 if n1 > idx else n1,
                        n2 - 1 if n2 > idx else n2,
                    ])
                m.members = updated_members

                # supports/loads must also be remapped to new node indices
                updated_supports = []
                for s in m.supports:
                    if s["node"] == idx:
                        continue
                    s["node"] = s["node"] - 1 if s["node"] > idx else s["node"]
                    updated_supports.append(s)
                m.supports = updated_supports

                updated_loads = []
                for l in m.loads:
                    if l["node"] == idx:
                        continue
                    l["node"] = l["node"] - 1 if l["node"] > idx else l["node"]
                    updated_loads.append(l)
                m.loads = updated_loads

                self.redraw()
                ed.statusBar().showMessage(f"Node {idx} deleted.", 4000)
                return

            ed.statusBar().showMessage("Delete mode: no nearby member or node.", 4000)


# ================= MAIN =================

class Editor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Truss Editor v2")
        self.resize(1200, 700)

        self.model = TrussModel()
        self.mode = "node"
        self.sel = []

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        left = QVBoxLayout()
        layout.addLayout(left, 3)

        toolbar = QHBoxLayout()
        left.addLayout(toolbar)

        for name in ["node","member","support","load","delete"]:
            b = QPushButton(name.capitalize())
            toolbar.addWidget(b)
            b.clicked.connect(lambda _, m=name: self.set_mode(m))

        solve = QPushButton("Solve")
        toolbar.addWidget(solve)
        solve.clicked.connect(self.solve)

        save = QPushButton("Save file (JSON)")
        toolbar.addWidget(save)
        save.clicked.connect(self.save)

        load = QPushButton("Load file (JSON)")
        toolbar.addWidget(load)
        load.clicked.connect(self.load)

        view_scale = QPushButton("View/Scale")
        toolbar.addWidget(view_scale)
        view_scale.clicked.connect(self.change_view_scale)

        reset_view_scale = QPushButton("Reset view/scale")
        toolbar.addWidget(reset_view_scale)

        self.canvas = TrussCanvas(self)
        left.addWidget(self.canvas)
        reset_view_scale.clicked.connect(self.canvas.reset_view_scale)

        # result table
        self.table = QTableWidget()
        layout.addWidget(self.table, 2)

        self.statusBar().showMessage("Mode: Node. Click in canvas to add a node.")

    def set_mode(self, m):
        self.mode = m
        self.sel.clear()
        self.set_status_for_mode()

    def set_status_for_mode(self):
        messages = {
            "node": "Mode: Node. Click in canvas to add a node.",
            "member": "Mode: Member. Click two existing nodes to create a member.",
            "support": "Mode: Support. Click an existing node to define support.",
            "load": "Mode: Load. Click an existing node to define load.",
            "delete": "Mode: Delete. Click a member or node to remove it.",
        }
        self.statusBar().showMessage(messages.get(self.mode, "Ready."))

    def find_node(self, x, y):
        if not self.model.nodes:
            return None
        pts = np.array(self.model.nodes)
        d = np.linalg.norm(pts-[x,y], axis=1)
        i = np.argmin(d)
        return i if d[i] < 0.5 else None

    # ---------- IO ----------
    def save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save", "", "JSON (*.json)")
        if path:
            data = {
                    "nodes": [[float(x), float(y)] for x, y in self.model.nodes],

                    "members": [[int(a), int(b)] for a, b in self.model.members],

                    "supports": [
                        {"node": int(s["node"]), "fix": [bool(s["fix"][0]), bool(s["fix"][1])]}
                        for s in self.model.supports
                    ],

                    "loads": [
                        {"node": int(l["node"]), "fx": float(l["fx"]), "fy": float(l["fy"])}
                        for l in self.model.loads
                    ],
                }
            
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            self.statusBar().showMessage(f"Saved model to: {path}", 6000)
##            with open(f,"w") as fh:
##                json.dump(self.model.to_dict(),fh,indent=2)

    def load(self):
        f,_ = QFileDialog.getOpenFileName(self,"Load","","JSON (*.json)")
        if f:
            with open(f) as fh:
                self.model.from_dict(json.load(fh))
            self.canvas.redraw()
            self.statusBar().showMessage(f"Loaded model from: {f}", 6000)

    def change_view_scale(self):
        dlg = ViewScaleDialog(
            self.canvas.ax.get_xlim(),
            self.canvas.ax.get_ylim(),
            self.canvas.load_arrow_scale,
        )
        if not dlg.exec_():
            return

        try:
            xmin = float(dlg.xmin.text())
            xmax = float(dlg.xmax.text())
            ymin = float(dlg.ymin.text())
            ymax = float(dlg.ymax.text())
            load_scale = float(dlg.load_scale.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid values", "Please enter numeric values.")
            return

        if xmin >= xmax or ymin >= ymax:
            QMessageBox.warning(self, "Invalid range", "Min value must be lower than max value.")
            return

        if load_scale <= 0:
            QMessageBox.warning(self, "Invalid scale", "Load arrow scale must be > 0.")
            return

        self.canvas.load_arrow_scale = load_scale
        self.canvas.ax.set_xlim(xmin, xmax)
        self.canvas.ax.set_ylim(ymin, ymax)
        self.canvas.redraw()
        self.statusBar().showMessage("View/scale updated.", 4000)

    # ---------- SOLVER ----------
    def solve(self):
        m = self.model
        if not m.members:
            self.statusBar().showMessage("Cannot solve: add at least one member first.", 5000)
            return

        n = len(m.nodes)
        K = assemble_global_stiffness(m.nodes, m.members)
        F = np.zeros(2*n)

        for l in m.loads:
            F[2*l["node"]] += l["fx"]
            F[2*l["node"]+1] += l["fy"]

        fixed=[]
        for s in m.supports:
            if s["fix"][0]: fixed.append(2*s["node"])
            if s["fix"][1]: fixed.append(2*s["node"]+1)

        free = np.setdiff1d(np.arange(2*n), fixed)
        Kff = K[np.ix_(free,free)]
        Ff = F[free]

        uf = solve_displacements(Kff,Ff)
        u = np.zeros(2*n); u[free]=uf

        forces = compute_member_forces(m.nodes,m.members,u)
        self.canvas.redraw(forces)

        # fill table
        self.table.setRowCount(len(forces))
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Member","Force"])

        for i,f in enumerate(forces):
            self.table.setItem(i,0,QTableWidgetItem(str(i)))
            self.table.setItem(i,1,QTableWidgetItem(f"{f:.3f}"))

        self.statusBar().showMessage("Solve finished. Results table updated.", 5000)


# ================= RUN =================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Editor()
    w.show()
    sys.exit(app.exec_())

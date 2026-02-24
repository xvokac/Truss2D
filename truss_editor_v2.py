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


# ================= CANVAS =================

class TrussCanvas(Canvas):
    def __init__(self, editor):
        self.fig = Figure(figsize=(6, 6))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.editor = editor

        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)

        self.mpl_connect("button_press_event", self.click)

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
            self.ax.arrow(x, y, l["fx"]*0.2, l["fy"]*0.2,
                          color="magenta", head_width=0.3)

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_aspect("equal")
        self.ax.grid(True)
        self.draw()

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

        elif ed.mode == "member":
            idx = ed.find_node(x, y)
            if idx is None:
                return
            ed.sel.append(idx)
            if len(ed.sel) == 2:
                m.members.append(ed.sel.copy())
                ed.sel.clear()
                self.redraw()

        elif ed.mode == "support":
            idx = ed.find_node(x, y)
            if idx is None:
                return
            dlg = SupportDialog()
            if dlg.exec_():
                m.supports.append({"node": idx,
                                   "fix": [dlg.fx.isChecked(), dlg.fy.isChecked()]})
                self.redraw()

        elif ed.mode == "load":
            idx = ed.find_node(x, y)
            if idx is None:
                return
            dlg = LoadDialog()
            if dlg.exec_():
                m.loads.append({"node": idx,
                                "fx": float(dlg.fx.text()),
                                "fy": float(dlg.fy.text())})
                self.redraw()

        elif ed.mode == "delete":
            # delete member first
            for i, (n1, n2) in enumerate(m.members):
                p1, p2 = np.array(m.nodes[n1]), np.array(m.nodes[n2])
                if np.linalg.norm(np.cross(p2-p1, p1-[x,y]))/np.linalg.norm(p2-p1) < 0.3:
                    m.members.pop(i)
                    self.redraw()
                    return

            # delete node
            idx = ed.find_node(x, y)
            if idx is not None:
                m.nodes.pop(idx)
                m.members = [mbr for mbr in m.members if idx not in mbr]
                self.redraw()


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

        save = QPushButton("Save")
        toolbar.addWidget(save)
        save.clicked.connect(self.save)

        load = QPushButton("Load")
        toolbar.addWidget(load)
        load.clicked.connect(self.load)

        self.canvas = TrussCanvas(self)
        left.addWidget(self.canvas)

        # result table
        self.table = QTableWidget()
        layout.addWidget(self.table, 2)

    def set_mode(self, m):
        self.mode = m
        self.sel.clear()

    def find_node(self, x, y):
        if not self.model.nodes:
            return None
        pts = np.array(self.model.nodes)
        d = np.linalg.norm(pts-[x,y], axis=1)
        i = np.argmin(d)
        return i if d[i] < 0.5 else None

    # ---------- IO ----------
    def save(self):
        f,_ = QFileDialog.getSaveFileName(self,"Save","","JSON (*.json)")
        if f:
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
            print(data)

            with open(path, "w") as f:
                json.dump(data, f, indent=2)
##            with open(f,"w") as fh:
##                json.dump(self.model.to_dict(),fh,indent=2)

    def load(self):
        f,_ = QFileDialog.getOpenFileName(self,"Load","","JSON (*.json)")
        if f:
            with open(f) as fh:
                self.model.from_dict(json.load(fh))
            self.canvas.redraw()

    # ---------- SOLVER ----------
    def solve(self):
        m = self.model
        if not m.members:
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


# ================= RUN =================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Editor()
    w.show()
    sys.exit(app.exec_())

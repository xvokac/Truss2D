import sys
import json
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QFileDialog, QTextEdit, QLabel, QMessageBox,
    QHBoxLayout, QDoubleSpinBox, QCheckBox
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

from truss_solver import solve_truss


class TrussCanvas(Canvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 5))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

    def plot(self, model, member_forces, load_scale_x=1.0, load_scale_y=1.0,
             dynamic_view=True):
        self.ax.clear()

        nodes = np.array(model["nodes"])
        members = model["members"]

        # draw members
        for i, (n1, n2) in enumerate(members):
            x = [nodes[n1, 0], nodes[n2, 0]]
            y = [nodes[n1, 1], nodes[n2, 1]]

            force = member_forces[i]
            color = "red" if force > 0 else "blue"

            self.ax.plot(x, y, color=color, linewidth=2)

            xm = np.mean(x)
            ym = np.mean(y)
            self.ax.text(xm, ym, f"{force:.2f}",
                         ha="center", va="center",
                         bbox=dict(facecolor="white", alpha=0.7))

        # nodes
        self.ax.scatter(nodes[:, 0], nodes[:, 1], color="black")

        for i, (x, y) in enumerate(nodes):
            self.ax.text(x, y, f" {i}")

        # supports
        for sup in model.get("supports", []):
            node = sup["node"]
            fix_x, fix_y = sup["fix"]
            x, y = nodes[node]

            if fix_x and fix_y:
                self.ax.plot(x, y, "gs", markersize=10)
            elif fix_y:
                self.ax.plot(x, y, "g^", markersize=10)
            elif fix_x:
                self.ax.plot(x, y, "g>", markersize=10)

        # loads
        load_endpoints = []
        for load in model.get("loads", []):
            node = load["node"]
            fx = load.get("fx", 0)
            fy = load.get("fy", 0)

            x, y = nodes[node]
            fx_vis = fx * load_scale_x
            fy_vis = fy * load_scale_y

            load_endpoints.append((x + fx_vis, y + fy_vis))
            self.ax.arrow(x, y, fx_vis, fy_vis,
                          head_width=0.1, color="magenta")
            self.ax.text(
                x + fx_vis,
                y + fy_vis,
                f"[{fx:.2f}, {fy:.2f}]",
                color="magenta",
                fontsize=9,
                ha="left",
                va="bottom",
            )

        if dynamic_view:
            min_x, max_x = np.min(nodes[:, 0]), np.max(nodes[:, 0])
            min_y, max_y = np.min(nodes[:, 1]), np.max(nodes[:, 1])

            if load_endpoints:
                load_endpoints = np.array(load_endpoints)
                min_x = min(min_x, np.min(load_endpoints[:, 0]))
                max_x = max(max_x, np.max(load_endpoints[:, 0]))
                min_y = min(min_y, np.min(load_endpoints[:, 1]))
                max_y = max(max_y, np.max(load_endpoints[:, 1]))

            pad_x = max((max_x - min_x) * 0.15, 0.5)
            pad_y = max((max_y - min_y) * 0.15, 0.5)
            self.ax.set_xlim(min_x - pad_x, max_x + pad_x)
            self.ax.set_ylim(min_y - pad_y, max_y + pad_y)

        self.ax.set_aspect("equal")
        self.ax.grid(True)
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("2D Truss Solver")
        self.resize(900, 700)

        self.model = None
        self.member_forces = None

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        self.load_btn = QPushButton("Load JSON")
        self.solve_btn = QPushButton("Solve")

        layout.addWidget(self.load_btn)
        layout.addWidget(self.solve_btn)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Force scale fx:"))
        self.fx_scale = QDoubleSpinBox()
        self.fx_scale.setRange(0.0, 10.0)
        self.fx_scale.setSingleStep(0.05)
        self.fx_scale.setValue(0.2)
        controls.addWidget(self.fx_scale)

        controls.addWidget(QLabel("Force scale fy:"))
        self.fy_scale = QDoubleSpinBox()
        self.fy_scale.setRange(0.0, 10.0)
        self.fy_scale.setSingleStep(0.05)
        self.fy_scale.setValue(0.2)
        controls.addWidget(self.fy_scale)

        self.dynamic_view = QCheckBox("Dynamic graph scale")
        self.dynamic_view.setChecked(True)
        controls.addWidget(self.dynamic_view)
        controls.addStretch()
        layout.addLayout(controls)

        self.canvas = TrussCanvas()
        layout.addWidget(self.canvas)

        layout.addWidget(QLabel("Results:"))

        self.results = QTextEdit()
        layout.addWidget(self.results)

        self.load_btn.clicked.connect(self.load_file)
        self.solve_btn.clicked.connect(self.solve)
        self.fx_scale.valueChanged.connect(self.update_plot)
        self.fy_scale.valueChanged.connect(self.update_plot)
        self.dynamic_view.toggled.connect(self.update_plot)

    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open JSON", "", "JSON Files (*.json)"
        )

        if fname:
            with open(fname) as f:
                self.model = json.load(f)

            self.filename = fname
            self.results.setText("Model loaded.")

    def solve(self):
        if not self.model:
            QMessageBox.warning(self, "Error", "Load a model first.")
            return

        try:
            u, reactions, member_forces = solve_truss(self.filename)
        except Exception as e:
            QMessageBox.critical(self, "Solver error", str(e))
            return

        self.member_forces = member_forces
        self.update_plot()

        # print results
        text = "MEMBER FORCES:\n\n"
        members = self.model["members"]
        for i, f in enumerate(member_forces):
            state = "Tension" if f > 0 else "Compression"
            n1, n2 = members[i]
            text += f"Member {i:3d} (nodes {n1}-{n2}): {f:10.3f}  {state}\n"

        text += "\nREACTIONS:\n\n"
        for i, r in enumerate(reactions):
            if abs(r) > 1e-8:
                node = i // 2
                direction = "X" if i % 2 == 0 else "Y"
                text += f"DOF {i:3d} (node {node}+{direction}): {r:10.3f}\n"

        self.results.setText(text)

    def update_plot(self):
        if self.model is None or self.member_forces is None:
            return

        self.canvas.plot(
            self.model,
            self.member_forces,
            load_scale_x=self.fx_scale.value(),
            load_scale_y=self.fy_scale.value(),
            dynamic_view=self.dynamic_view.isChecked(),
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())



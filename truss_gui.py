import sys
import json
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QFileDialog, QTextEdit, QLabel, QMessageBox
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

from truss_solver import solve_truss


class TrussCanvas(Canvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 5))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

    def plot(self, model, member_forces):
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
        for load in model.get("loads", []):
            node = load["node"]
            fx = load.get("fx", 0)
            fy = load.get("fy", 0)

            x, y = nodes[node]
            scale = 0.2

            self.ax.arrow(x, y, fx*scale, fy*scale,
                          head_width=0.1, color="magenta")

        self.ax.set_aspect("equal")
        self.ax.grid(True)
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("2D Truss Solver")
        self.resize(900, 700)

        self.model = None

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        self.load_btn = QPushButton("Load JSON")
        self.solve_btn = QPushButton("Solve")

        layout.addWidget(self.load_btn)
        layout.addWidget(self.solve_btn)

        self.canvas = TrussCanvas()
        layout.addWidget(self.canvas)

        layout.addWidget(QLabel("Results:"))

        self.results = QTextEdit()
        layout.addWidget(self.results)

        self.load_btn.clicked.connect(self.load_file)
        self.solve_btn.clicked.connect(self.solve)

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

        # plot
        self.canvas.plot(self.model, member_forces)

        # print results
        text = "MEMBER FORCES:\n\n"
        for i, f in enumerate(member_forces):
            state = "Tension" if f > 0 else "Compression"
            text += f"Member {i:3d}: {f:10.3f}  {state}\n"

        text += "\nREACTIONS:\n\n"
        for i, r in enumerate(reactions):
            if abs(r) > 1e-8:
                text += f"DOF {i:3d}: {r:10.3f}\n"

        self.results.setText(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

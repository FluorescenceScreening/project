import sys
from string import ascii_uppercase

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QTableWidgetItem
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import random

Ui_MainWindow, QtBaseClass = uic.loadUiType("window4.ui")


class MyWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.efficiency= random.sample(range(0, 300), 96)
        self.samples= list(range(1, 97))
        self.pushButton_generate_random_signal.clicked.connect(self.update_graph)
        self.pushButton_generate_random_signal.clicked.connect(self.update_table)
        self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))
        self.setWindowTitle("Fluorescence  Efficiency Evaluation")
        self.init_graph()
        self.datatable.setHorizontalHeaderLabels(['Sample\nIndex','Relatine\nEfficiency (%)'])

    def init_graph(self):
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.set_title('Relative Efficiency of each sample')
        self.MplWidget.canvas.axes._axes.set_ylabel('Relative Efficiency (%)')
        self.MplWidget.canvas.axes._axes.set_xlabel('Sample\n\n')
        self.MplWidget.canvas.axes._axes.set_xticks(list(range(1, 97, 12)))
        self.MplWidget.canvas.axes._axes.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        # 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'B1', 'B2', 'B3', 'B4', 'B5',
        # 'B6',
        # 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
        # 'C12', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'E1', 'E2', 'E3', 'E4',
        # 'E5',
        # 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
        # 'F11',
        # 'F12', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'H1', 'H2', 'H3', 'H4',
        # 'H5',
        # 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12'])

    def update_graph(self):
        eff = self.efficiency
        eff[0] = 100
        samp = self.samples
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.bar(samp, eff)
        self.MplWidget.canvas.axes.set_title('Relative Efficiency of each sample')
        self.MplWidget.canvas.axes._axes.set_ylabel('Relative Efficiency (%)')
        self.MplWidget.canvas.axes._axes.set_xlabel('Sample\n\n')
        self.MplWidget.canvas.axes._axes.set_xticks(list(range(1, 97, 12)))
        self.MplWidget.canvas.axes._axes.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        # 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'B1', 'B2', 'B3', 'B4', 'B5',
        # 'B6',
        # 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
        # 'C12', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'E1', 'E2', 'E3', 'E4',
        # 'E5',
        # 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
        # 'F11',
        # 'F12', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'H1', 'H2', 'H3', 'H4',
        # 'H5',
        # 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12'])
        self.MplWidget.canvas.draw()

    def update_table(self):
        eff = self.efficiency
        i = 0
        for c in ascii_uppercase:
            for x in range(1, 13):
                self.datatable.setItem(i, 1, QTableWidgetItem(str(eff[i])))
                self.datatable.setItem(i, 0, QTableWidgetItem(c + str(x)))
                i=i+1
            if c == "H":
                break
        self.datatable.setItem(0, 1, QTableWidgetItem('Reference'))
        self.progressBar.setValue(100)
        self.efficiency= random.sample(range(0, 300), 96)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())

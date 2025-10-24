from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton,QLabel, QLineEdit, QVBoxLayout, QWidget, QMenu, QAction
from PyQt5.QtCore import QSize
import sys


app = QApplication(sys.argv)
class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.isChecked = False
        self.setWindowTitle("My PyQt5 Window")
        self.button = QPushButton("please end me")
        self.button.clicked.connect(self.end)

        self.label = QLabel("Enter something:")
        self.input = QLineEdit()
        self.input.textChanged.connect(self.label.setText)


        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.input)
        layout.addWidget(self.label)
        

        container = QWidget()
        container.setLayout(layout)

        self.setMouseTracking(True)
        container.setMouseTracking(True)
        self.label.setMouseTracking(True)
        self.button.setMouseTracking(True)
        self.input.setMouseTracking(True)

        self.setCentralWidget(container)
        self.setMinimumSize(QSize(500, 300))

    def mouseMoveEvent(self, e):
        print(f"Mouse at: {e.x()}, {e.y()}")
        self.label.setText(f"Mouse at: {e.x()}, {e.y()}")
    
    def mousePressEvent(self, e):
        print(f"Mouse clicked at: {e.x()}, {e.y()}")
        self.label.setText(f"Mouse clicked at: {e.x()}, {e.y()}")
        print("Clicked by button:", e.button())

    def contextMenuEvent(self, e):
        contextMenu = QMenu(self)
        contextMenu.addAction(QAction("New", self))
        contextMenu.addAction(QAction("Open", self))
        contextMenu.addAction(QAction("Quit", self))
        action = contextMenu.exec(e.globalPos())
        if action == contextMenu.actions()[2]:
            self.end()


    def end(self):
        app.quit()










def main():
    window = MyWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()




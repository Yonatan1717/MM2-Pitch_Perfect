from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QLabel,
    QLineEdit,
    QHBoxLayout,
    QWidget,
    QMenu,
    QAction,
    QStackedLayout,
)
from PyQt5.QtCore import QSize, Qt
import sys


app = QApplication(sys.argv)
class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My PyQt5 Window")
        self.whitekeys = []
        for i in range(52):
            key = QWidget()
            key.setStyleSheet("background-color: white; border: 1px solid black;")
            key.setFixedSize(QSize(20, 200))
            self.whitekeys.append(key)

        self.blackkeys = []
        for i in range(36):
            key = QWidget()
            key.setStyleSheet("background-color: black; border: 1px solid black;")
            key.setFixedSize(QSize(15, 120))
            self.blackkeys.append(key)

        def midi_to_name(m):
            pcs = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
            pc = pcs[m % 12]
            octv = m // 12 - 1
            return f"{pc}{octv}"

        note_names = [midi_to_name(m) for m in range(21, 109)]  # A0..C8
        note_to_widget = {}
        w_i = 0  # white index
        b_i = 0  # black index
        for name in note_names:
            if "#" in name:
                note_to_widget[name] = self.blackkeys[b_i]
                b_i += 1
            else:
                note_to_widget[name] = self.whitekeys[w_i]
                w_i += 1

        flat_to_sharp = {"Db":"C#","Eb":"D#","Gb":"F#","Ab":"G#","Bb":"A#"}
        for name in list(note_to_widget.keys()):
            if "#" in name:
                base = name[:-1]  # e.g., A#
                octv = name[-1]
                for fl, sh in flat_to_sharp.items():
                    if base == sh:
                        note_to_widget[f"{fl}{octv}"] = note_to_widget[name]

        self.desiredbox = note_to_widget

        self._default_white_style = "background-color: white; border: 1px solid black;"
        self._default_black_style = "background-color: black; border: 1px solid black;"

        def set_note_color(note: str, color: str):
            n = note.strip().upper().replace('B', 'B') 
            for fl, sh in flat_to_sharp.items():
                n = n.replace(fl, sh)
            w = self.desiredbox.get(n)
            if not w:
                return
            w.setStyleSheet(f"background-color: {color}; border: 1px solid black;")

        self.set_note_color = set_note_color  

        layout_white = QHBoxLayout()
        layout_white.setContentsMargins(0, 0, 0, 0)
        layout_white.setSpacing(0)
        for key in self.whitekeys:
            layout_white.addWidget(key, alignment=Qt.AlignLeft | Qt.AlignTop)
        layout_white.addStretch(1)

        white_layer = QWidget()
        white_layer.setLayout(layout_white)

        black_layer = QWidget()
        black_layer.setAttribute(Qt.WA_StyledBackground, True)
        black_layer.setStyleSheet("background: transparent;")

        white_w = 20
        black_w = 15

        whites_in_order = [n for n in note_names if "#" not in n]
        has_black_after_white_flags = [w[0] not in ("E", "B") for w in whites_in_order[:-1]]

        positions = [i for i, flag in enumerate(has_black_after_white_flags) if flag]
        positions = positions[:len(self.blackkeys)]

        for key, i_between in zip(self.blackkeys, positions):
            key.setParent(black_layer)
            x = int((i_between + 1) * white_w - black_w / 2)
            key.move(x, 0)

        stacked = QStackedLayout()
        stacked.setStackingMode(QStackedLayout.StackAll)
        stacked.addWidget(white_layer)
        stacked.addWidget(black_layer)
        stacked.setCurrentWidget(black_layer)  

        container = QWidget()
        container.setLayout(stacked)

        self.setCentralWidget(container)
        self.setMinimumSize(QSize(500, 300))

        self.set_note_color("A4", "red")


    def end(self):
        app.quit()


def main():
    window = MyWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()




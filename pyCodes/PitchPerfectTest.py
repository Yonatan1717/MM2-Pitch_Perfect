from queue import Queue, Full, Empty
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import (    
    QStackedLayout,
    QApplication, 
    QMainWindow,
    QPushButton, 
    QVBoxLayout, 
    QHBoxLayout,
    QSizePolicy,
    QWidget,
    QLabel
)
import numpy as np
import threading
import pyaudio
import sys
import math
import time

CHUNK = 1024                                 # antall prøver per buffer
CHANNELS = 1                                 # mono
RATE = 24000                                 # sample per sekund (r)
FFT_SIZE = 2048                              # størrelse på FFT-vindu (N)
HOP_SIZE = 512                               # hop størrelse
MAX_FREQ = 9000                              # maksimal frekvens å analysere
MIN_FREQ = 16                                # minimal frekvens å analysere
INT16_MAX = 32767                            # maksimal verdi for int16
NOISE = 0.003 * INT16_MAX                    # initial støyterskel
ALPHA = 0.995                                # glatt faktor
NOISE_MULTIPLIER = 2                         # justerbar multiplikator for støyterskel
FIXED_GUI_SIZE = (1500, 800)                 # fast størrelse på GUI
FONT_SIZE = 10                               # skriftstørrelse for labels

app = QApplication(sys.argv)

class AudioRecorderProducer(threading.Thread):
    def __init__(self, queue, chunk=CHUNK, channels=CHANNELS, rate=RATE):
        super().__init__(daemon=True)
        self.queue = queue
        self.chunk = chunk
        self.channels = channels
        self.rate = rate
        self._stop_producer = False
        self._pause = False
        self._wake_event = threading.Event()

    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk)
        try:
            while not self._stop_producer:

                if self._pause:
                    self._wake_event.clear()
                    self._wake_event.wait()  # vent til vi blir vekket

                data = np.frombuffer(stream.read(self.chunk, exception_on_overflow=False), dtype=np.int16)

                # prov å legg data til køen, hvis full, fjern eldste element og prøv igjen

                try:
                    self.queue.put(data, timeout=0.1)
                except Full:
                    try:
                        _ =  self.queue.get_nowait()
                    except Empty:
                    
                    # prøv igjen å legg til data
                        pass
                    try:
                        self.queue.put_nowait(data)
                    except Full:
                        pass
        finally:
            stream.stop_stream()
            stream.close() 
            p.terminate()

            try:
                self.queue.put_nowait(None)  # signal for å stoppe forbrukeren
            except Full:
                pass
            
    def start(self):
        self._stop_producer = False
        return super().start()

    def pause(self):
        self._pause = True

    def unpause(self):
        self._pause = False
        self._wake_event.set()

    def stop(self):
        self._stop_producer = True
                
class AudioVisualizerConsumer(threading.Thread):
    def __init__(self, queue, my_window=None):
        super().__init__(daemon=True)
        self.queue = queue
        self.buffer = np.zeros(0, dtype=np.float32)
        self.window = np.hanning(FFT_SIZE).astype(np.float32)
        self.max_k = np.floor(MAX_FREQ / (RATE / FFT_SIZE)).astype(int)
        self.min_k = np.ceil(MIN_FREQ / (RATE / FFT_SIZE)).astype(int)
        self.noise = NOISE
        self.alpha = ALPHA
        self.noise_multiplier = NOISE_MULTIPLIER
        self.last_note = None
        self.last_print = 0 
        self.my_window = my_window
        self._stop_consumer = False
        self._pause = False
        self._wake_event = threading.Event()

    def run(self):
        while not self._stop_consumer:
            if self._pause:
                self._wake_event.clear()
                self._wake_event.wait()  # vent til vi blir vekket

            item = self.queue.get()
            if item is None:
                break # no more data to process

            self.buffer = np.concatenate((self.buffer, item.astype(np.float32)))

            while len(self.buffer) >= FFT_SIZE:
                data = self.buffer[:FFT_SIZE]
                self.buffer = self.buffer[HOP_SIZE:]

                data_windowed = data * self.window

                win_rms = np.sqrt(np.mean(self.window**2))
                rms = np.sqrt(np.mean(data_windowed**2)) / win_rms


                if rms < self.noise_multiplier * self.noise:
                    self.noise = self.alpha * self.noise + (1 - self.alpha) * rms

                RMS_THRESHOLD =  self.noise_multiplier * self.noise

                if rms < RMS_THRESHOLD:
                    continue # skip lav effekts rammer 

                freq_domain = np.fft.rfft(data_windowed, n=FFT_SIZE)
                mags = np.abs(freq_domain)

                kmax = int(max(self.max_k, len(mags) - 1))
                if kmax <= self.min_k + 1:
                    continue # ikke interessant

                k_top10 = np.argsort(mags[self.min_k:kmax])[-10:][::-1] + self.min_k
                
                # Kvadratisk interpolasjon for bedre frekvensestimat
                delta_k_top_10 = np.array([self.quad_interpolate(mags, k) for k in k_top10]) 
                freq = delta_k_top_10 * (RATE / FFT_SIZE)

                now = time.time()
                if now - self.last_print > 0.4:
                    for i, label in enumerate(self.my_window.labels):
                        note_name, cents, note_freq, error_hz = self.freq_to_note(freq[i])
                        label.setText(f"Top {i + 1}:\n\t Note: {note_name}  \n\t Cents: {cents:.2f} \n\t Error: {error_hz:.2f} Hz \n\t Ideell Freq: {note_freq:.2f} Hz \n\t Actual Freq: {freq[i]:.2f} Hz \n\t Magnitude: {mags[k_top10[i]]:.2f}")
                    
                    max_freq_note = self.freq_to_note(freq[0])[0]
                    if max_freq_note != self.last_note:
                        # restet forrige notat
                        if self.last_note:
                            if self.last_note in self.my_window.desiredbox:
                                w = self.my_window.desiredbox[self.last_note]
                                if "#" in self.last_note:
                                    w.setStyleSheet(self.my_window._default_black_style)
                                else:
                                    w.setStyleSheet(self.my_window._default_white_style)

                        # highlight nåværende notat
                        if max_freq_note in self.my_window.desiredbox:
                            w = self.my_window.desiredbox[max_freq_note]
                            w.setStyleSheet(f"background-color: red; border: 1px solid black;")
                            self.last_note = max_freq_note

    def freq_to_note(self, freq, a4=440.0, prefer_sharps=True):
        note_names_sharp = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        note_names_flat  = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
        if not math.isfinite(freq) or freq <= 0:
            return float('nan'), float('nan'), float('nan'), float('nan')

        # MIDI note nummer
        n = 69 + 12 * math.log2(freq / a4)
        n_round = int(round(n))                     # nærmeste MIDI-note
        cents = 100.0 * (n - n_round)               # avvik i cents

        names = note_names_sharp if prefer_sharps else note_names_flat
        note_name = f"{names[n_round % 12]}{(n_round // 12) - 1}"

        # Ideell frekvens for denne noten
        note_freq = a4 * (2 ** ((n_round - 69) / 12))
        error_hz = freq - note_freq

        return note_name, cents, note_freq, error_hz

    def quad_interpolate(self, mags, k):
        if k <= 0 or k >= len(mags) - 1:
            return 0  # Kan ikke interpolere ved kantene
        m_b = mags[k - 1]
        m_m = mags[k]
        m_n = mags[k + 1]
        denominator = (m_b - 2 * m_m + m_n)
        if denominator == 0:
            return 0  # Unngå deling på null
            
        delta = 0.5 * (m_b - m_n) / denominator
        return k + delta

    def pause(self):
        self._pause = True

    def unpause(self):
        self._pause = False
        self._wake_event.set()

    def stop(self):
        self._stop_consumer = True

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pitch Perfect - Audio Visualizer")
        

        self.button_unpause = QPushButton("Start Audio Processing")
        self.button_pause = QPushButton("Stop Audio Processing")
        self.button_unpause.clicked.connect(self.unpause_audio_processing)
        self.button_pause.clicked.connect(self.pause_audio_processing)
        self.button_unpause.setEnabled(True)
        self.button_pause.setEnabled(False)
        self.labels = [QLabel(f"{i}: N/A") for i in range(1, 11)]
        font = self.labels[0].font()
        font.setPointSize(FONT_SIZE)

        layoutH1 = QHBoxLayout()
        for i in range(len(self.labels)//2):
            label = self.labels[i]
            label.setFont(font)
            layoutH1.addWidget(label)

        layoutH2 = QHBoxLayout()
        for i in range(len(self.labels)//2, len(self.labels)):
            label = self.labels[i]
            label.setFont(font)
            layoutH2.addWidget(label)


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

        # Build mapping from musical note names (e.g., "A4", "C#5") to the corresponding QWidget key
        # The 88-key piano range is MIDI 21 (A0) to 108 (C8). We'll construct names using sharps.
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

        # Support flats as aliases (e.g., Bb4 -> A#4)
        self.flat_to_sharp = {"Db":"C#","Eb":"D#","Gb":"F#","Ab":"G#","Bb":"A#"}
        for name in list(note_to_widget.keys()):
            if "#" in name:
                base = name[:-1]  # e.g., A#
                octv = name[-1]
                for fl, sh in self.flat_to_sharp.items():
                    if base == sh:
                        note_to_widget[f"{fl}{octv}"] = note_to_widget[name]

        # Expose a friendly mapping requested: desiredbox["A4"] -> QWidget
        self.desiredbox = note_to_widget

        # Remember default styles so we can toggle highlights easily later
        self._default_white_style = "background-color: white; border: 1px solid black;"
        self._default_black_style = "background-color: black; border: 1px solid black;"

        # bunn lag hvite taster
        layout_white = QHBoxLayout()
        layout_white.setContentsMargins(0, 0, 0, 0)
        layout_white.setSpacing(0)
        for key in self.whitekeys:
            layout_white.addWidget(key, alignment=Qt.AlignLeft | Qt.AlignTop)
        layout_white.addStretch(1)

        white_layer = QWidget()
        white_layer.setLayout(layout_white)

    # tip lag svart lag over hvit
        black_layer = QWidget()
        black_layer.setAttribute(Qt.WA_StyledBackground, True)
        black_layer.setStyleSheet("background: transparent;")

        white_w = 20
        black_w = 15

        whites_in_order = [n for n in note_names if "#" not in n]
        # svart existerer etter hvit hvis hvit ikke er E eller B
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

        center = QWidget()
        center.setLayout(stacked)
        center.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        layoutH3 = QHBoxLayout()
        layoutH3.addWidget(center, alignment=Qt.AlignCenter)

        container = QWidget()
        layoutV = QVBoxLayout(container)
        layoutV.addLayout(layoutH1)
        layoutV.addLayout(layoutH2)
        layoutV.addLayout(layoutH3)
        layoutV.addWidget(self.button_unpause)
        layoutV.addWidget(self.button_pause)

        container.setLayout(layoutV)
        self.setCentralWidget(container)
        self.setMinimumSize(QSize(*FIXED_GUI_SIZE))
        
        self.init_audio_processing()

    def init_audio_processing(self):
        print("Audio processing started...")
        self.queue = Queue(maxsize=31)
        self.producer = AudioRecorderProducer(self.queue)
        self.consumer = AudioVisualizerConsumer(self.queue, my_window=self)
        self.producer.start()
        self.consumer.start()
        self.pause_audio_processing()

    def set_note_color(self, note: str, color: str):
            n = note.strip().upper().replace('B', 'B')  # keep case predictable
            # Normalize flats to sharps where applicable
            for fl, sh in self.flat_to_sharp.items():
                n = n.replace(fl, sh)
            w = self.desiredbox.get(n)
            if not w:
                return
            # Keep border visible while changing fill color
            w.setStyleSheet(f"background-color: {color}; border: 1px solid black;")
    
    def start_audio_processing(self):
        self.button_start.setEnabled(False)
        self.button_stop.setEnabled(True)
        self.producer.start()
        self.consumer.start()

    def pause_audio_processing(self):
        self.button_unpause.setEnabled(True)
        self.button_pause.setEnabled(False)
        if hasattr(self, 'producer'):
            self.producer.pause()
        if hasattr(self, 'consumer'):
            self.consumer.pause()

    def unpause_audio_processing(self):
        self.button_unpause.setEnabled(False)
        self.button_pause.setEnabled(True)
        if hasattr(self, 'producer'):
            self.producer.unpause()
        if hasattr(self, 'consumer'):
            self.consumer.unpause()

    def stop_audio_processing(self):
        self.button_unpause.setEnabled(True)
        self.button_pause.setEnabled(False)
        if hasattr(self, 'producer'):
            self.producer.stop()
        if hasattr(self, 'consumer'):
            self.consumer.stop()

    def closeEvent(self, event):
        self.pause_audio_processing()
        event.accept()

def main():
    my_window = MyWindow() 
    my_window.show()
    app.exec()

if __name__ == "__main__":
    main()
from queue import Queue, Full, Empty
from PyQt5.QtCore import QSize, Qt
import matplotlib.pyplot as plt
from collections import deque
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

                                
CHANNELS = 1                                    # mono
RATE = 48000                                    # sample per sekund (r)
FFT_SIZE = 2048                                 # størrelse på FFT-vindu (N)
HOP_SIZE = 256                                  # hop størrelse
CHUNK = HOP_SIZE                                # antall prøver per buffer
MAX_FREQ = 8000                                 # maksimal frekvens å analysere
MIN_FREQ = 20                                   # minimal frekvens å analysere
INT16_MAX = 32767                               # maksimal verdi for int16
NOISE = 0.004 * INT16_MAX                       # initial støyterskel
ALPHA = 0.99                                    # glatt faktor
NOISE_MULTIPLIER = 3                            # justerbar multiplikator for støyterskel
FIXED_GUI_SIZE = (1500, 900)                    # fast størrelse på GUI
FONT_SIZE = 10                                  # skriftstørrelse for labels
EXCLUSION_BINS = 3                              # 2–4 er bra for Hann-vindu (undertrykk nabo-binner)
PADDING_FACTOR = 4                              # zero-padding faktor for FFT
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
        self._pa = None
        self._stream = None

    # --- PyAudio callback: blir kalt i PortAudio-tråd ---
    def _cb(self, in_data, frame_count, time_info, status):
        # Ikke blokker her! (callback må være superrask)
        if not self._pause and not self._stop_producer:
            arr = np.frombuffer(in_data, dtype=np.int16).copy()  # kopier ut av PA-buffer
            try:
                self.queue.put_nowait(arr)
            except Full:
                # dropp eldste for å holde lav latens
                try:
                    _ = self.queue.get_nowait()
                except Empty:
                    pass
                try:
                    self.queue.put_nowait(arr)
                except Full:
                    pass
        # fortsett streamen
        return (None, pyaudio.paContinue)

    def run(self):
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,   # du kan sette = HOP_SIZE for litt lavere latens
            stream_callback=self._cb
        )
        self._stream.start_stream()

        try:
            # hold tråden i live til vi skal stoppe
            while not self._stop_producer:
                if self._pause:
                    # sov litt mens vi er pauset (callback kjører fortsatt, men dropper data)
                    self._wake_event.wait(timeout=0.05)
                else:
                    time.sleep(0.05)
        finally:
            # rydd opp pent
            if self._stream is not None:
                try:
                    self._stream.stop_stream()
                except Exception:
                    pass
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

            if self._pa is not None:
                try:
                    self._pa.terminate()
                except Exception:
                    pass
                self._pa = None

            # signaliser til consumer at vi er ferdige
            try:
                self.queue.put_nowait(None)
            except Full:
                pass

    def start(self):
        self._stop_producer = False
        return super().start()

    def pause(self):
        self._pause = True
        # ikke clear() her—callback sjekker bare flagget

    def unpause(self):
        self._pause = False
        self._wake_event.set()

    def stop(self):
        self._stop_producer = True
        self._wake_event.set()

                
class AudioVisualizerConsumer(threading.Thread):

    def __init__(self, queue, my_window=None):
        super().__init__(daemon=True)
        self.queue = queue
        self.chuncks = deque()
        self.total = 0
        self.winbuff = np.empty(FFT_SIZE, dtype=np.float32)
        self.window = np.hanning(FFT_SIZE).astype(np.float32)
        self.win_rms2 = np.mean(self.window**2)
        self.max_k = np.floor(MAX_FREQ / (RATE / FFT_SIZE)).astype(int)
        self.min_k = np.ceil(MIN_FREQ / (RATE / FFT_SIZE)).astype(int)
        self.noise = float(NOISE**2)  # initial støyterskel i effekt
        self.alpha = ALPHA
        self.noise_multiplier = NOISE_MULTIPLIER
        self.last_note = None
        self.last_print = 0 
        self.my_window = my_window
        self._stop_consumer = False
        self._pause = False
        self._wake_event = threading.Event()
        self.mags = np.zeros(self.max_k + 1, dtype=np.float32)
        self.last_wind_data = None
        self.M = FFT_SIZE * PADDING_FACTOR

    def run(self):               
        
        while not self._stop_consumer:
            if self._pause:
                self._wake_event.clear()
                self._wake_event.wait()  # vent til vi blir vekket

            item = self.queue.get()
            if item is None:
                break # no more data to process

            self.append_chunk(item.astype(np.float32))

            while self.total >= FFT_SIZE:
                data = self.build_window()
                data_windowed = data * self.window
                self.consume_left(HOP_SIZE)

                rms = float(np.mean(data_windowed**2) / self.win_rms2)


                if rms < (self.noise_multiplier**2) * self.noise:
                    self.noise = self.alpha * self.noise + (1 - self.alpha) * rms

                RMS_THRESHOLD =  (self.noise_multiplier**2) * self.noise

                if rms < RMS_THRESHOLD:
                    continue # skip lav effekts rammer 

                freq_domain = np.fft.rfft(data_windowed, n=FFT_SIZE)
                self.mags = np.abs(freq_domain)
                mags = self.mags[:self.max_k + 1]

                kmax = int(min(self.max_k, len(mags) - 1))
                if kmax <= self.min_k + 1:
                    continue # ikke interessant

                k_top10 = self._pick_peaks_nms(mags, self.min_k, kmax, K=10, exclusion=EXCLUSION_BINS)
                if k_top10.size == 0:
                    continue

                # Kvadratisk interpolasjon for frekvensestimat
                delta_k_top_10 = np.array([self.quad_interpolate(mags, k) for k in k_top10], dtype=np.float32)
                freq = delta_k_top_10 * (RATE / FFT_SIZE)

                # Sorter toppene etter styrke (samme som før)
                order = np.argsort(mags[k_top10])[::-1]
                k_top10 = k_top10[order]
                freq = freq[order]

                now = time.time()
                if now - self.last_print > 0.08:
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
                            self.my_window.NoteLabel.setText(f"{max_freq_note}")
                            self.last_note = max_freq_note

                    self.last_print = now
                    self.last_wind_data = data_windowed.copy()

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
    
    def _pick_peaks_nms(self, mags, k_min, k_max, K=10, exclusion=EXCLUSION_BINS):
        """Velg maks K topper uten nære duplikater (NMS i bin-rom)."""
        region = mags[k_min:k_max]
        if region.size == 0:
            return np.array([], dtype=int)
        # hent mange kandidater, sorter sterkest først
        cand_rel = np.argpartition(region, -K*8)[-K*8:]
        cand = cand_rel + k_min
        cand = cand[np.argsort(mags[cand])[::-1]]

        selected = []
        for k in cand:
            if all(abs(k - s) > exclusion for s in selected):
                selected.append(k)
                if len(selected) == K:
                    break
        return np.array(selected, dtype=int)
    
    def append_chunk(self, chunk):
        self.chuncks.append(chunk)
        self.total += len(chunk)
        
    def build_window(self):
        filled = 0
        for c in self.chuncks:
            take = min(len(c), FFT_SIZE - filled)
            self.winbuff[filled:filled+take] = c[:take]
            filled += take
            if filled == FFT_SIZE:
                break
        
        return self.winbuff.copy()
    
    def consume_left(self, n: int):
        while n > 0 and self.chuncks:
            c = self.chuncks[0]
            if len(c) <= n:
                n -= len(c)
                self.total -= len(c)
                self.chuncks.popleft()
            else:
                self.chuncks[0] = c[n:]
                self.total -= n
                n = 0
        
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
        
        self.plotButton = QPushButton("Plot Frequency Spectrum of the last FFT Window")
        self.button_unpause = QPushButton("Start Audio Processing")
        self.button_pause = QPushButton("Stop Audio Processing")
        
        self.button_unpause.clicked.connect(self.unpause_audio_processing)
        self.button_pause.clicked.connect(self.pause_audio_processing)    
        self.plotButton.clicked.connect(self.plotLastFFT)
        
        self.button_unpause.setEnabled(True)
        self.button_pause.setEnabled(False)
        self.plotButton.setEnabled(False)
        
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
        w_i = 0  # hvit index
        b_i = 0  # svart index
        for name in note_names:
            if "#" in name:
                note_to_widget[name] = self.blackkeys[b_i]
                b_i += 1
            else:
                note_to_widget[name] = self.whitekeys[w_i]
                w_i += 1

        self.flat_to_sharp = {"Db":"C#","Eb":"D#","Gb":"F#","Ab":"G#","Bb":"A#"}
        for name in list(note_to_widget.keys()):
            if "#" in name:
                base = name[:-1]  # f.eks. A#
                octv = name[-1]
                for fl, sh in self.flat_to_sharp.items():
                    if base == sh:
                        note_to_widget[f"{fl}{octv}"] = note_to_widget[name]

        self.desiredbox = note_to_widget

        # default
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

        self.NoteLabel = QLabel("N/A")
        notefont = self.NoteLabel.font()
        notefont.setPointSize(30)
        self.NoteLabel.setFont(notefont)
        self.NoteLabel.setAlignment(Qt.AlignCenter)
        
        container = QWidget()
        layoutV = QVBoxLayout(container)
        layoutV.addLayout(layoutH1)
        layoutV.addLayout(layoutH2)
        layoutV.addWidget(self.NoteLabel)
        layoutV.addLayout(layoutH3)
        layoutV.addWidget(self.button_unpause)
        layoutV.addWidget(self.button_pause)
        layoutV.addWidget(self.plotButton)

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
        self.plotButton.setEnabled(False)

    def set_note_color(self, note: str, color: str):
            n = note.strip().upper().replace('B', 'B') 
            for fl, sh in self.flat_to_sharp.items():
                n = n.replace(fl, sh)
            w = self.desiredbox.get(n)
            if not w:
                return
            w.setStyleSheet(f"background-color: {color}; border: 1px solid black;")
    
    def start_audio_processing(self):
        self.button_start.setEnabled(False)
        self.button_stop.setEnabled(True)
        self.producer.start()
        self.consumer.start()

    def pause_audio_processing(self):
        self.button_unpause.setEnabled(True)
        self.button_pause.setEnabled(False)
        self.plotButton.setEnabled(True)
        if hasattr(self, 'producer'):
            self.producer.pause()
        if hasattr(self, 'consumer'):
            self.consumer.pause()

    def unpause_audio_processing(self):
        self.button_unpause.setEnabled(False)
        self.button_pause.setEnabled(True)
        self.plotButton.setEnabled(False)
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
        
    def plotLastFFT(self):
        fig = plt.figure("Frequency Spectrum", figsize=(10, 6))
        def on_close(event):
            self.plotButton.setEnabled(True)
    
        fig.canvas.mpl_connect('close_event', on_close)

        ax = fig.add_subplot(1, 1, 1)

        self.plotButton.setEnabled(False)
        fft = np.fft.rfft(self.consumer.last_wind_data, n=self.consumer.M)
        freqs_full = np.fft.rfftfreq(self.consumer.M, d=1.0 / RATE)
        fft_mags = np.abs(fft)

        min_k_plot = int(np.searchsorted(freqs_full, MIN_FREQ, side='left'))
        k_max_plot = int(np.searchsorted(freqs_full, MAX_FREQ, side='right')) - 1
        k_max_plot = max(min_k_plot, min(k_max_plot, len(freqs_full) - 1, len(fft_mags) - 1))
        
        if k_max_plot < 1:
            return  # nothing meaningful to plot

        freqs = freqs_full[min_k_plot: k_max_plot + 1]
        plot_mags = fft_mags[min_k_plot: k_max_plot + 1]

        ax.plot(freqs, plot_mags)
        ax.set_title("Frequency Spectrum of Last FFT Window")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.grid()
        plt.show()

    def plotLastSpectrogram(self):
        pass  # Placeholder for future implementation
def main():
    my_window = MyWindow() 
    my_window.show()
    app.exec()

if __name__ == "__main__":
    main()
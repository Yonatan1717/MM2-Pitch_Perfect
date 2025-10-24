import pyaudio
import numpy as np
import threading
from queue import Queue
from queue import Queue, Full, Empty
import time


CHUNK = 1024
CHANNELS = 1
RATE = 24000
FFT_SIZE = 2048
HOP_SIZE = 512
MAX_FREQ = 9000
MIN_FREQ = 16
INT16_MAX = 32767

class AudioRecorderProducer(threading.Thread):
    def __init__(self, queue, chunk=CHUNK, channels=CHANNELS, rate=RATE):
        super().__init__(daemon=True)
        self.queue = queue
        self.chunk = chunk
        self.channels = channels
        self.rate = rate
        self._stop = threading.Event()

    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk)
        try:
            while not self._stop.is_set():
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
            
            
    def stop(self):
        self._stop.set()
                


class AudioVisualizerConsumer(threading.Thread):
    def __init__(self, queue):
        super().__init__(daemon=True)
        self.queue = queue
        self.buffer = np.zeros(0, dtype=np.float32)
        self.window = np.hanning(FFT_SIZE).astype(np.float32)
        self.max_k = np.floor(MAX_FREQ / (RATE / FFT_SIZE)).astype(int)
        self.min_k = np.ceil(MIN_FREQ / (RATE / FFT_SIZE)).astype(int)
        self.last_note = None
        self.last_print = 0 

    def run(self):
        while True:
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

                noise = 0.003 * INT16_MAX # initial støyterskel
                alpha = 0.995 # glatt faktor
                noise_multiplier = 4 # justerbar multiplikator for støyterskel

                if rms < noise_multiplier * noise:
                    noise = alpha * noise + (1 - alpha) * rms

                RMS_THRESHOLD =  noise_multiplier * noise

                if rms < RMS_THRESHOLD:
                    continue # skip lav effekts rammer 

                freq_domain = np.fft.rfft(data_windowed, n=FFT_SIZE)
                mags = np.abs(freq_domain)

                kmax = int(max(self.max_k, len(mags) - 1))
                if kmax <= self.min_k + 1:
                    continue # ikke interessant

                k = np.argmax(mags[self.min_k:kmax]) + self.min_k
                
                # Kvadratisk interpolasjon for bedre frekvensestimat
                delta_k = self.quad_interpolate(mags, k)
                freq = delta_k * (RATE / FFT_SIZE)

                now = time.time()
                if self.last_note != freq or (now - self.last_print) > 0.4:
                    print(f"Frekvens: {freq:.2f} Hz")
                    self.last_note = freq
                    self.last_print = now



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



            


            

            




def main():
    queue = Queue(maxsize=32)
    producer = AudioRecorderProducer(queue)
    consumer = AudioVisualizerConsumer(queue)
    try:
        consumer.start()
        
        # Hold hovedtråden i live
        producer.start()
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("stopper...")
    finally:
        producer.stop()
        producer.join(timeout=1.0)
        try:
            queue.put_nowait(None)
        except Full:
            pass
        consumer.join(timeout=1.0)

if __name__ == "__main__":
    main()
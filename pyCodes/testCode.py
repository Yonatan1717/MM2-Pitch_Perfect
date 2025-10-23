import pyaudio
import numpy as np
import threading
from queue import Queue
import time


class AudioRecorderProducer(threading.Thread):
    def __init__(self, queue, chunk=1024, channels=1, rate=16000):
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
                data = np.frombuffer(stream.read(self.chunk), dtype=np.int16)
                try:
                    self.queue.put(data, timeout=0.1)
                except Full:
                    try:
                        _ =  self.queue.get_nowait()
                    except Empty:
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
                self.queue.put_nowait(None)  # Signal the consumer to exit
            except Full:
                pass
            
            
    def stop(self):
        self._stop.set()
                


class AudioVisualizerConsumer(threading.Thread):
    def __init__(self, queue):
        super().__init__(daemon=True)
        self.queue = queue

    def run(self):
        while True:
            if self.queue.empty():
                pass
            else:
                data = self.queue.get()
                top3_freqs = []
                window = np.hanning(len(data))
                windowed_data = data * window
                fft_data = np.fft.rfft(windowed_data)
                magnitude = np.abs(fft_data)
                greatest_freq_index = np.argmax(magnitude)
                freq_bin = np.fft.rfftfreq(len(windowed_data), d=1/16000)
                greatest_freq = freq_bin[greatest_freq_index]
                for i in range(3):
                    top3_freqs.append(int(greatest_freq))
                    magnitude = np.delete(magnitude, greatest_freq_index)
                    freq_bin = np.delete(freq_bin, greatest_freq_index)
                    greatest_freq_index = np.argmax(magnitude)
                    greatest_freq = freq_bin[greatest_freq_index]
                average_freq = sum(top3_freqs) / len(top3_freqs)
                print("Top 3 frequencies: ", top3_freqs, "avaerage frequency: ", int(average_freq))
                

def main():
    queue = Queue(maxsize=32)
    producer = AudioRecorderProducer(queue)
    consumer = AudioVisualizerConsumer(queue)
    try:
        consumer.start()
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
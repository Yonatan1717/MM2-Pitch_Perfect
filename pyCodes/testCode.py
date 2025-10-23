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

    def run(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk)
        while True:
            data = np.frombuffer(self.stream.read(self.chunk), dtype=np.int16)
            self.queue.put(data)
        
    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


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
    queue = Queue()
    producer = AudioRecorderProducer(queue)
    consumer = AudioVisualizerConsumer(queue)
    try:
        consumer.start()
        producer.start()
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        producer.stop()
        queue.put(None)  # Signal the consumer to exit
        consumer.join()

if __name__ == "__main__":
    main()
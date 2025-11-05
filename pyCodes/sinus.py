import numpy as np
import matplotlib.pyplot as plt

def leakage_demo(f0=25.0, fs=10_000, T=5, pad=20):
    # Rektangulær ramme (gir diskontinuitet i periodisk forlengelse)
    t = np.arange(0, T, 1/fs)
    x = np.sin(2*np.pi*f0*t)

    N = len(x)

    M = int(pad * 2**np.ceil(np.log2(N)))

    X = np.fft.rfft(x, n=M)
    f = np.fft.rfftfreq(M, d=1/fs)

    mag = np.abs(X) / (N/2)
    mag[0] /= 2 

    if N % 2 == 0:
        mag[-1] /= 2  
    db_ref = mag.max()
    mag_db = 20 * np.log10(mag / db_ref + 1e-20)  # i dB

    Xbins = np.fft.rfft(x)
    fbins = np.fft.rfftfreq(N, d=1/fs)
    magbins = np.abs(Xbins) / (N/2)
    magbins[0] /= 2


    # Sammenlikning mot Hann (viser hvordan lekkasje dempes – valgfritt)
    w = np.hanning(N)
    Xw = np.fft.rfft(x*w, n=M)
    magw = np.abs(Xw) / (N/2); magw[0] /= 2
    if N % 2 == 0:
        magw[-1] /= 2
    
    magw_db = 20 * np.log10(magw / db_ref + 1e-12)  # i dB

    plt.figure(figsize=(10,4))
    plt.plot(f, mag_db, label="Rectangular")
    plt.plot(f, magw_db, label="Hann")
    plt.xlim(f0-5, f0+5)
    plt.ylim(-100, 3); 
    plt.grid(True); plt.legend()
    plt.title("Lekkasje: Rektangulær vs Hann")
    plt.grid(False)
    plt.xlabel("Frekvens [Hz]"); plt.ylabel("Relativ magnitude [dB]")
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.savefig("./Rapport/Media/frekvens_spekter.png", dpi=500)

leakage_demo()

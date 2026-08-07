import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob, os
CS8_DC = 0.6 #CF32 (or 32 bit) centers on 127.4, CS8 (or 8 bit) centers on 128 


def load_iq_files(meta_file):
    data = np.load(meta_file, allow_pickle = True)
    iq = data["iq_files"]
    base = os.path.dirname(meta_file)
    iq_file = []
    i=0
    for file_name in iq:
        raw = (np.fromfile(os.path.join(base, file_name), dtype = data["dtype"][0]))

        if data["dtype"][0] == 'int8':
            raw = ((raw[0::2] + CS8_DC) + 1j*(raw[1::2] + CS8_DC)) / 128
            raw = raw.astype(np.complex64)
        iq_file.append(raw)
        print(f"{len(iq_file[i])} samples, var {np.var(iq_file[i]):.7f}")
        i+=1
    return iq_file


def windowed_delay(a, b, maxlag, n_chunks = 16):
    assert len(a) == len(b)
    c = np.array_split(a, n_chunks)
    d = np.array_split(b, n_chunks)
    peak_lag = []
    ratio = []
    for i in range(n_chunks):
        lags, scores = xcorr_fft(c[i], d[i], maxlag)
        mag = np.abs(scores)
        peak_lag.append(lags[np.argmax(mag)])
        peak = np.abs(scores[np.argmax(np.abs(scores))])
        floor = np.sqrt(np.mean(np.delete(mag, np.argmax(mag))**2))
        ratio.append(peak/floor)
    chunk_seconds = (len(a)/2.048e6)/n_chunks
    times = ((np.arange(n_chunks) + 0.5) * chunk_seconds)
    return peak_lag, ratio, times
          

def xcorr_fft(a, b, maxlag):
    assert len(a) == len(b)
    N=len(a)
    Var_ceiling = np.ceil(np.log2(N + maxlag))
    power = 2 ** int(Var_ceiling)
    c = np.fft.ifft(np.fft.fft(a, power) * np.conj(np.fft.fft(b, power)))
    lags = np.arange(-maxlag, maxlag+1)
    scores = c[(-lags) % power]
    
    return lags, scores

          
def xcorr_brute(a, b, maxlag):
    assert len(a) == len(b)
    L_scores = []
    N = len(a)
    for L in range(-maxlag, maxlag+1):
        if L >= 0:
            slice_a = a[:N-L]
            slice_b = b[L:]
        else:
            slice_a = a[-L:]
            slice_b = b[:N+L]
        L_scores.append(np.sum(slice_a * np.conj(slice_b)))
    lags = np.arange(-maxlag, maxlag+1)
    L_scores = np.array(L_scores)
    return lags, L_scores
        
def plot_corr(times, slope, intercept, peak_lag):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    ax.plot(times, peak_lag, "o", color="#2e6fb7", label="measured delay (per 0.25 s chunk)")
    ax.plot(times, slope * times + intercept,
            color="#c2503b", linewidth=2,
            label=f"fit: {slope:.2f} samples/s  →  {abs(slope)/2.048:.2f} ppm")
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Measured delay (samples)")
    ax.set_title("Delay drift between free-running receivers (1 ppm injected)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig("drift_staircase.png", dpi=150)

def rotation_correction(a, b, maxlag):
    '''
    returns b -> rotated so its peak phase matches a's peak phase
    '''
    lags, scores = xcorr_fft(a, b, maxlag)
    peak_score = scores[np.argmax(np.abs(scores))]
    theta = np.angle(peak_score)
    b_fixed = b * np.exp(1j * theta)
    return theta, b_fixed

def acquire(a, b):
    assert len(a) == len(b)
    N=len(a)
    a = a - np.mean(a)
    b = b - np.mean(b)
    Var_ceiling = np.ceil(np.log2(N*2))
    power = 2 ** int(Var_ceiling)
    c = np.fft.ifft(np.fft.fft(a, power) * np.conj(np.fft.fft(b, power)))
    mag = np.abs(c)
    m = np.argmax(mag)
    lag = -m if m<=power//2 else power-m
    floor = np.sqrt(np.mean(np.delete(mag, m)**2))
    peak = mag[m]
    ratio = peak/floor
    return lag, ratio

def align(a, b, lag):
    if lag > 0:
        a_al = a[:-lag]
        b_al = b[lag:]
    elif lag < 0:
        a_al = a[-lag:]
        b_al = b[:lag]
    else:
        a_al = a
        b_al = b
    return a_al, b_al


def cross_spectrum(a, b, veclength = 1024, samp_rate = 2.048e6):
    n_chunks = len(a)//veclength
    a = a[:n_chunks*veclength]
    b = b[:n_chunks*veclength]
    c = np.array_split(a, n_chunks)
    d = np.array_split(b, n_chunks)
    X_ab = []
    window = np.blackman(veclength)
    for i in range(n_chunks):
        c_mean = np.mean(c[i])
        d_mean = np.mean(d[i])
        c[i] = c[i] - c_mean
        d[i] = d[i] - d_mean
        X_ab.append(np.fft.fft(c[i] * window) * np.conj(np.fft.fft(d[i] * window)))
    X_ab = np.mean(X_ab, axis = 0)
    freqs = np.fft.fftfreq(veclength, 1/samp_rate)
    return freqs, X_ab


def delay_from_spectrum(freqs, X_ab, tau_max = 0.5, dtau = 0.001):
    #we already removed the integer lag with aquire, so tau can only span ±0.5 or else it would round  
    samp_rate = len(freqs) * (freqs[1]-freqs[0]) # we space by samp_rate in cross_spectrum so we can find it again here
    freq_norm = freqs / samp_rate
    keep = (np.abs(freq_norm) > 0.01) & (np.abs(freq_norm) < 0.4) #RTLs have inherent DC spikes at the beginning and attenuates at the end
    freq_norm = freq_norm[keep]
    X_ab  = X_ab[keep]
    tau_grid = np.arange(-tau_max, tau_max, dtau)
    Search = []
    for i in range(len(tau_grid)):
        Search.append(np.abs(np.sum(X_ab* np.exp(-2j* np.pi * freq_norm * tau_grid[i])))) #searching the full tau range. the correct tau value will align in all bins and add, the wrong tau value will cancel
    Search = np.array(Search)
    peak = np.argmax(Search)
    tau = tau_grid[peak]
    floor = np.sqrt(np.sum(np.abs(X_ab)**2))    
    ratio = Search[peak]/floor
    return tau, ratio, tau_grid, Search

def anti_slip(a, b):
    pass


if __name__ == "__main__":
    meta_file = max(glob.glob("*.metadata.npz"), key = os.path.getmtime)
    print(f"Loading {meta_file}")
    data = load_iq_files(meta_file)    
    if len(data) < 2:
        raise ValueError(f"need at least 2 antennas to correlate, {meta_file} has {len(data)}")
    '''
    theta, b_fixed = rotation_correction(data[0], data[1], maxlag=100)
    a = data[0]
    b = b_fixed
    lags, scores = xcorr_fft(a, b, maxlag=100)
    lag, ratio = acquire(a,b)
    
    print(f"b is {lag} after a")
    peak = np.abs(scores[np.argmax(np.abs(scores))])
    print(f" Peak : {peak}")
    print(f" Ratio: {ratio}") 
    print(f"Angle: {theta}")
    
    peak_lag, ratio, times = windowed_delay(a, b, maxlag=50)
    slope, intercept = np.polyfit(times, peak_lag, 1)
    
    plot_corr(times, slope, intercept, peak_lag)
'''

    a = data[0]
    b = data[1]
    lag, ratio = acquire(a,b)
    
    print(f"b is {lag} after a before align ratio:{ratio}")

    a, b = align(a, b,lag)
    lag, ratio = acquire(a,b)
    cross_spectrum(a, b)
    
    print(f"b is {lag} after a after align ratio:{ratio}")

    


    

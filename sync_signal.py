import numpy as np
import adi
from time import sleep

# Use an ADALM-Pluto SDR to transmit a roughly white-noise signal
# to synchronize two chart horns.

sample_rate = 2e6 # Hz
center_freq = 1420e6 # Hz

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)
sdr.tx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
sdr.tx_lo = int(center_freq)
sdr.tx_hardwaregain_chan0 = 0 # Increase to increase tx power, valid range is -90 to 0 dB

N = 10000 # number of samples to transmit at once

# transmit batches of signals, alternating on/off
print('transmitting')
for i in range(200):  # 1 s
    samples = 2**14*(np.random.rand(N)-.5)
    sdr.tx(samples) # transmit the batch of samples once
    
sleep(.5) # .5 s
for i in range(250): # 1.25s
    samples = 2**14*(np.random.rand(N)-.5)
    sdr.tx(samples) # transmit the batch of samples once
    
sleep(1) # 1 s
for i in range(100): # .5s
    samples = 2**14*(np.random.rand(N)-.5)
    sdr.tx(samples) # transmit the batch of samples once
    
sleep(.25) # .25 s
for i in range(300): # 1.5s
    samples = 2**14*(np.random.rand(N)-.5)
    sdr.tx(samples) # transmit the batch of samples once
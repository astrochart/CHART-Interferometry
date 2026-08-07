''' Simulate a multi-antenna interferometer capture.'''
from gnuradio import gr, blocks, analog, filter, soapy
import numpy as np
import argparse
import time
import glob, os
import datetime
CS8_DC = 0.6 #CF32 (or 32 bit) centers on 127.4, CS8 (or 8 bit) centers on 128 


class TopBlock(gr.top_block):
    def __init__(self, sample_rate = 2.048e6, seconds = 1, rx_amp = 1, delay_len = None, sky_seed = 45, drift_ppm = None, source = 'sim', center_freq = 1419.99e6, gain = 40.0, serials = None, biastee = True):
        '''Initialize the collect top block. 
        Parameters:
        -----------
        sample_rate: float, optional
            Sample rate of radio in Hz, default 2.048e6
        seconds: int, optional
            How long data is being taken, units: seconds, default: 1
        rx_amp: float, optional
            The amplitude of each receiver's own noise source, default: 1
        delay_len: list of int, optional
            Per-antenna delay applied to the shared sky signal, in samples.
            Models the geometric delay: the same wavefront reaches each
            antenna at a slightly different time because the antennas are in
            different locations. Receiver noise is not delayed. Must have one
            entry per antenna (length num_ANT), default: [0,37]
        sky_seed: int, optional
            The random seed that determines the noise of the sky source
        drift_ppm: float, optional
            Per-antenna drift caused by systematic noise in the internal crystal. 
            Must have one entry per antenna (can be 0 in all antennae), default: [0,0]
        center_freq: float, optional
            Center frequency, in Hz. Default is 1419.99e6, which puts the 21 cm HI
            line (1420.406 MHz) at +416 kHz baseband — in band, clear of the DC
            spike. Was 1575.42e6 (GPS L1) from the divider work; that default
            silently cost 7 captures on Aug 6, 2026.
        gain: int, optional
            Tuner gain in dB for each RTL dongle (RTL mode only). Default 40.0.
        serials: str, optional
            Serial number of each RTL dongle, one per antenna (rtl mode only).
            Text labels, not numbers -> leading zeros matter. Length sets the
            number of antennas. Default ['00000101', '00000102'].
        biastee: bool, optional
            If True, enable the bias-T to power an active antenna through the
            coax (rtl mode only). Default True.
        '''
        gr.top_block.__init__(self)

        self.sample_rate = sample_rate 
        self.seconds = seconds 
        self.rx_amp = rx_amp
        self.delay_len = delay_len
        self.sky_seed = sky_seed
        self.drift_ppm = drift_ppm
        self.source = source
        self.center_freq = center_freq
        self.gain = gain
        self.serials = serials
        self.biastee =  biastee
        self.sinks = []
        
        if self.delay_len is None:
            self.delay_len = [0,37]
        if self.drift_ppm is None:
            self.drift_ppm = [0] * len(self.delay_len)
        if len(self.drift_ppm) != len(self.delay_len):
            raise ValueError("drift ppm and delay len dont have same number of inputs")
        # Establish run filenames from timestamp
        self.set_filename()
        # default is None instead of [0,37] because a mutable default is
        # created once at function definition, not per call; every call would
        # share (and could corrupt) the same list. Assigning inside gives each
        # call a fresh one.
        self.rx_seeds = []
        self.iq_files = []
        self.n_samples = self.sample_rate * self.seconds
        self.n_samples = int(self.n_samples)

        if source =='sim':
            self.sky = analog.noise_source_c(analog.GR_GAUSSIAN, 1, self.sky_seed)
            self.num_ANT = len(self.delay_len)
            #sky source is shared across different receivers, but each receiver has its own source
            #we are looking at the same source, so we expect the same signal
            #but each receiver is a different instrument and thus has a different source/seed
        
        elif source == 'rtl':
            if self.serials is None:
                self.serials = ['00000101', '00000102']
            self.num_ANT = len(self.serials)
        else:
            raise ValueError(f"unknown source '{source}' (expected 'sim' or 'rtl')")

        for i in range(self.num_ANT):
            if source == 'rtl':
                front = self.rtl_front_end(i)
            
            else:
                front = self.sim_front_end(i)
            self.itemsize = front.output_signature().sizeof_stream_item(0)
            self.sinks.append(blocks.file_sink(self.itemsize, self.filebase + f".ANT_{i+1}.iq"))
            self.iq_files.append(self.filebase + f".ANT_{i+1}.iq")
            head = blocks.head(self.itemsize, self.n_samples)

            self.connect((front, 0),(head, 0))
            self.connect((head, 0),(self.sinks[i], 0))

        self.date=datetime.date.today()    
        self.start_time = time.time()

    def rtl_front_end(self, i):
        serial = self.serials[i]
        src = soapy.source(f"driver=rtlsdr,serial={serial}", 'sc8', 1, '', 'buffers=64', [''], ['']) 
        #soapy.source(device, data type, nchan, dev_args, stream_args, tune_args, other settings) 
        # for data types: https://github.com/pothosware/SoapyRTLSDR/blob/master/Streaming.cpp; however, cs needs to be sc, and cf needs to be fc when putting it above
        src.set_sample_rate(0, self.sample_rate)
        src.set_frequency(0, self.center_freq)
        src.set_gain(0, 'TUNER', self.gain)
        src.write_setting('biastee', 'true' if self.biastee else 'false')
        return src
        
    def sim_front_end(self, i):
        #receiver noise:
        rx = analog.noise_source_c(analog.GR_GAUSSIAN, self.rx_amp, int(i+1))
        self.rx_seeds.append(i+1)
        #we iterate the third param (the seed) because each receiver has a different source seed
        adder = blocks.add_cc()
        delay = blocks.delay(gr.sizeof_gr_complex, self.delay_len[i])
        drift = 1 + self.drift_ppm[i]*1e-6
        crystal_sampler = filter.mmse_resampler_cc(0, drift)
        #heads are per branch (per receiver) because the delay can create different lengths of the signals, which we don't want
        self.connect((self.sky, 0),(delay, 0))
        self.connect((delay, 0),(adder, 0))
        self.connect((rx, 0),(adder, 1))
        self.connect((adder, 0),(crystal_sampler, 0))
        return crystal_sampler

    def set_filename(self, filebase=None):
        """Set filename.

        Args:
            filebase: Optional base for filename. If not supplied,
                create filename from datetime
        """
        if filebase is None:
            filebase = str(datetime.datetime.now()).replace(' ', '_')
            filebase = filebase.replace(':', '-')
        self.filebase = filebase 
        self.metadata_file = filebase + '.metadata.npz'

    def meta_save(self):
        """Save the metadata."""
        if self.source == 'sim':
            np.savez(self.metadata_file,
                     date = self.date,
                     start_time = self.start_time,
                     end_time = time.time(),
                     sample_rate = self.sample_rate, 
                     seconds = self.seconds,
                     num_ANT = self.num_ANT,
                     rx_amp = self.rx_amp,
                     delay_len = self.delay_len,
                     drift_ppm = self.drift_ppm,
                     sky_seed = self.sky_seed,
                     dtype = ["complex64"],
                     metadata_file = self.metadata_file,
                     rx_seeds = self.rx_seeds,
                     iq_files = self.iq_files,
                     source = self.source,
                     serials = self.serials,
                     center_freq = None,
                     gain = None,
                     biastee = None
                   )
        else:
            np.savez(self.metadata_file,
                     date = self.date,
                     start_time = self.start_time,
                     end_time = time.time(),
                     sample_rate = self.sample_rate, 
                     seconds = self.seconds,
                     num_ANT = self.num_ANT,
                     rx_amp = None,
                     delay_len = None,
                     drift_ppm = None,
                     sky_seed = None,
                     dtype = ['int8'],
                     metadata_file = self.metadata_file,
                     rx_seeds = None,
                     iq_files = self.iq_files,
                     source = self.source,
                     center_freq = self.center_freq,
                     gain = self.gain,
                     serials = self.serials,
                     biastee = self.biastee
                    )
def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True

    if v.lower() in ("no", "false", "f", "n", "0"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")

    
def collectArgs():

    ap = argparse.ArgumentParser(prog="capture.py", description="CHART data interferometry collection utility", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--seconds", default = 1, type = int, help = "Length in seconds of observation")
    ap.add_argument("--source", default = 'sim', type = str, help = "simulated (sim) source or real (rtl) source")

    ap.add_argument("--rx_amp", default = 1, type = float, help = "Amplitude of each receivers noise source")
    ap.add_argument("--delay_len", nargs = '+', default = None, type = int, help = "delay in samples")
    ap.add_argument("--sky_seed", default = 45, type = int, help = "random seed of sky noise")
    ap.add_argument("--drift_ppm", nargs = '+', default = None, type = float, help = "drift of an antennas crystal")
    
    ap.add_argument("--center_freq", default = 1419.99e6, type = float, help = "Center frequency, in Hz")
    ap.add_argument("--gain", default = 40.0, type = float, help = "how concentrated you want your radio energy")
    ap.add_argument("--serials", nargs = '+', default = None, type = str, help = "label of the antennae")
    ap.add_argument("--biastee", default = True, type=str2bool, nargs="?", const=True, help = "Enable Bias-T power")


    return ap.parse_args()

def check(meta_file):
    data = np.load(meta_file, allow_pickle = True)
    iq = data["iq_files"]
    base = os.path.dirname(meta_file)
    iq_file = []
    var = []
    i=0
    for file_name in iq:
        raw = (np.fromfile(os.path.join(base, file_name), dtype = data["dtype"][0]))

        if data["dtype"][0] == 'int8':
            raw = ((raw[0::2] + CS8_DC) + 1j*(raw[1::2] + CS8_DC)) / 128
            raw = raw.astype(np.complex64)
        iq_file.append(raw)
        var.append(np.var(iq_file[i]))
        print(f"{len(iq_file[i])} samples, var {np.var(iq_file[i]):.7f}")
        i+=1
    if max(var)/min(var) > 5:
        print(f"WARNING: var imbalance {max(var)/min(var):.1f}")
    if min(var) == 0 or not np.isfinite(var).all():
        print(f"WARNING: dead or empty file — {[len(x) for x in iq_file]} samples")
        
    return iq_file



if __name__ == "__main__":
    args = collectArgs()
    app = TopBlock(seconds = args.seconds, rx_amp = args.rx_amp, delay_len = args.delay_len, sky_seed = args.sky_seed, drift_ppm = args.drift_ppm, source = args.source, center_freq = args.center_freq, gain =  args.gain, serials = args.serials, biastee =  args.biastee)
    app.run()
    app.meta_save()
    check(app.metadata_file)
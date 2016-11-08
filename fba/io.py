import os
import numpy as np
from neo.io import Spike2IO
from scipy.io import loadmat

from . import features


def get_files_with_ext(path, ext):
    return list(filter(lambda fname: ext in fname, os.listdir(path)))


def get_zipped_paths(base_path):
    spike_paths = [name for name in os.listdir(base_path) if len(name) == 3]
    track_paths = [name for name in os.listdir(base_path) if len(name) > 3]

    zipped_paths = []
    for spike_path in spike_paths:
        for track_path in track_paths:
            if spike_path == track_path.split('-1c')[0].split('_')[1]:
                zipped_paths.append((spike_path, track_path))
                break

    return zipped_paths


def data_path_generator(base_path):
    zipped_paths = get_zipped_paths(base_path)

    for spike_path, track_path in zipped_paths:
        spike_path = os.path.join(base_path, spike_path)
        track_path = os.path.join(base_path, track_path)

        spike_fname = get_files_with_ext(spike_path, 'smr')
        assert(len(spike_fname) == 1)
        spike_fname = os.path.join(spike_path, spike_fname[0])

        track_fname = get_files_with_ext(track_path, 'mat')
        assert(len(track_fname) == 1)
        track_fname = os.path.join(track_path, track_fname[0])

        yield spike_fname, track_fname


def load_track(track_fname):
    track_mat = loadmat(track_fname)
    x = track_mat['x']
    y = track_mat['y']
    theta = track_mat['angle']

    return x, y, theta


def load_spike2io(fname):
    r = Spike2IO(filename=fname)
    seg = r.read_segment(lazy=False, cascade=True)
    return seg


def parse_spiketrains(fname, binsize=100):
    seg = load_spike2io(fname)

    binned_spiketrains = [features.bin_spiketrain(strain, binsize) for strain in seg.spiketrains]
    num_spikes = [int(strain.sum()) for strain in binned_spiketrains]

    assert(num_spikes[0] > np.all(np.array(num_spikes[2:])))
    assert(num_spikes[1] > np.all(np.array(num_spikes[2:])))

    return binned_spiketrains[:2]

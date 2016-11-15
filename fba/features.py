import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d


class FeatureTransformer:
    def __init__(self, x, y, theta):
        self.x = x.astype(np.float64)
        self.y = y.astype(np.float64)
        self.theta = theta.astype(np.float64)

        self.x_egocentric = self._make_egocentric(self.x)
        self.y_egocentric = self._make_egocentric(self.y)
        self.theta = self._transform_angles()
        self.relative_angles = self._get_relative_angles()
        self.direction_angles = self._get_direction_angles()
        self.relative_distances = self._get_relative_distances()

    @staticmethod
    def _make_egocentric(coords):
        return coords - np.repeat(coords[:, 0][:, np.newaxis], 12, axis=1)

    def _transform_angles(self):
        angles = self.theta % 360
        angles = (angles / 360) * 2 * np.pi
        angles[angles > np.pi] -= 2 * np.pi
        return angles

    def _get_relative_angles(self):
        relative_angles = np.zeros((self.theta.shape[0], self.theta.shape[1]-1), dtype=np.float32)

        for time_idx in range(self.theta.shape[0]):
            alpha = self.theta[time_idx, 0]
            v_alpha = np.array((np.cos(alpha), np.sin(alpha)))
            v_alpha /= np.linalg.norm(v_alpha)

            for neighbor_idx in range(1, self.theta.shape[1]):
                beta = self.theta[time_idx, neighbor_idx]

                rel_angle = alpha - beta
                if rel_angle < -np.pi:
                    rel_angle += 2 * np.pi
                elif rel_angle > np.pi:
                    rel_angle -= 2 * np.pi
                relative_angles[time_idx, neighbor_idx-1] = rel_angle

        return relative_angles

    def _get_direction_angles(self):
        direction_angles = np.zeros((self.x.shape[0], self.x.shape[1]-1), dtype=np.float32)

        for time_idx in range(self.x.shape[0]):
            alpha = self.theta[time_idx, 0]
            v_alpha = np.array((np.cos(alpha), np.sin(alpha)))
            v_alpha /= np.linalg.norm(v_alpha)

            for neighbor_idx in range(1, self.x.shape[1]):
                v_beta = np.array((self.x_egocentric[time_idx, neighbor_idx],
                                   self.y_egocentric[time_idx, neighbor_idx]))
                beta_norm = np.linalg.norm(v_beta)

                # TODO: FIXME
                if beta_norm == 0.:
                    direction_angles[time_idx, neighbor_idx-1]
                    continue

                v_beta /= np.linalg.norm(v_beta)

                beta = np.arctan2(v_beta[1], v_beta[0])

                rel_angle = alpha - beta
                if rel_angle < -np.pi:
                    rel_angle += 2 * np.pi
                elif rel_angle > np.pi:
                    rel_angle -= 2 * np.pi
                direction_angles[time_idx, neighbor_idx-1] = rel_angle

        return direction_angles

    def _get_relative_distances(self):
        relative_distances = np.zeros((self.x.shape[0], self.x.shape[1]-1), dtype=np.float32)

        for time_idx in range(self.x.shape[0]):
            v = np.array((self.x_egocentric[time_idx, 0], self.y_egocentric[time_idx, 0]))
            assert(np.abs(np.linalg.norm(v)) < 1e-5)

            for neighbor_idx in range(1, self.x.shape[1]):
                v = np.array((self.x_egocentric[time_idx, neighbor_idx],
                              self.y_egocentric[time_idx, neighbor_idx]))
                relative_distances[time_idx, neighbor_idx-1] = np.linalg.norm(v)

        return relative_distances

    def get_features(self, num_bins=8):
        bin_angles = np.arange(-np.pi, np.pi, 2 * np.pi / num_bins)
        angle_binned = np.digitize(self.direction_angles, bin_angles) - 1

        features_per_bin = 6
        X = np.zeros((self.x.shape[0], num_bins, features_per_bin), dtype=np.float32)

        has_vector_idx = 0
        sin_angle_idx = 1
        cos_angle_idx = 2
        dist_idx = 3
        sin_rel_angle_idx = 4
        cos_rel_angle_idx = 5

        X[:, :, has_vector_idx] = -1

        for time_idx in range(self.x.shape[0]):
            for neighbor_idx in range(1, self.x.shape[1]):
                angle_bin = angle_binned[time_idx, neighbor_idx-1]
                dist = self.relative_distances[time_idx, neighbor_idx-1]
                angle = self.direction_angles[time_idx, neighbor_idx-1]
                rel_angle = self.relative_angles[time_idx, neighbor_idx-1]

                if np.isnan(dist) or not np.isfinite(dist):
                    continue

                # if there's already a feature in the bin, only add the current one if it's closer
                if np.any(X[time_idx, angle_bin, has_vector_idx+1:]):
                    if dist >= X[time_idx, angle_bin, dist_idx]:
                        continue

                X[time_idx, angle_bin, has_vector_idx] = 1
                X[time_idx, angle_bin, sin_angle_idx] = np.sin(angle)
                X[time_idx, angle_bin, cos_angle_idx] = np.cos(angle)
                X[time_idx, angle_bin, dist_idx] = dist
                X[time_idx, angle_bin, sin_rel_angle_idx] = np.sin(rel_angle)
                X[time_idx, angle_bin, cos_rel_angle_idx] = np.cos(rel_angle)

        # standardize distance feature
        # TODO
        #X[:, :, dist_idx] -= np.mean(X[:, :, dist_idx])
        #X[:, :, dist_idx] /= np.std(X[:, :, dist_idx])

        return X


class TargetTransformer:
    def __init__(self, spiketrain):
        self.spiketrain = spiketrain if type(spiketrain) == np.ndarray else np.array(spiketrain)

    def substract_trend(self, trend_sigma=5000):
        self.spiketrain -= gaussian_filter1d(self.spiketrain, trend_sigma)
        return self

    def smooth(self, smooth_sigma=100):
        self.spiketrain = gaussian_filter1d(self.spiketrain, smooth_sigma)
        return self

    def to_derivative(self):
        self.spiketrain = np.concatenate(([0], self.spiketrain[1:] - self.spiketrain[:-1]))
        return self

    def standardize(self):
        self.spiketrain /= np.std(self.spiketrain)
        return self

    def digitize(self, vmin=-5, vmax=5, vstep=.5):
        self.bins = np.arange(vmin, vmax, vstep)
        self.spiketrain = np.digitize(self.spiketrain, self.bins)
        return self

    def get_targets(self):
        return self.spiketrain


def bin_spiketrain(strain, binsize=100):
    df_strain = pd.Series([1 for _ in strain], index=pd.to_timedelta(np.array(strain), unit='s'))
    df_strain = pd.concat((pd.Series([0], pd.to_timedelta([0], unit='s')), df_strain))
    resampler = df_strain.resample('{}ms'.format(binsize), closed='left')
    df_binned = resampler.sum().fillna(0.)
    df_binned.index += pd.to_timedelta(binsize, unit='ms')
    return df_binned


class SequenceSampler:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def sample_random_subsequences(self, seq_length, num_samples):
        X_seq = np.zeros((num_samples,
                          seq_length,
                          self.features.shape[1],
                          self.features.shape[2]))
        Y_seq = np.zeros((num_samples, len(self.targets)))

        random_indices = np.random.choice(self.features.shape[0] - seq_length,
                                          size=num_samples,
                                          replace=False)
        for sample_idx, start_idx in enumerate(random_indices):
            X_seq[sample_idx] = self.features[start_idx:start_idx+seq_length]
            for target_idx in range(len(self.targets)):
                Y_seq[sample_idx, target_idx] = self.targets[target_idx][start_idx+seq_length]

        assert(np.all([t.dtype == np.int for t in self.targets]))
        return X_seq, Y_seq.astype(np.int32)

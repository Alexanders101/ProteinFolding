import tensorflow as tf
import numpy as np
from keras.engine.topology import Layer
from keras.layers.core import Lambda


class Lattice(Layer):
    def __init__(self, protein_length, **kwargs):
        """
        Keras Layer for transforming protein index strings into a 3d lattice.

        This layer will only work when using the Tensorflow Backend.

        Parameters
        ----------
        protein_length : Maximum length of input protein. Referred to as N.

        Layer Parameters
        ----------------
        Layer expects the following input:
        [acids, idx, batch]
        -------------------
        acids : (None, N) - float32
            String of acids
        idx : (None, N, 3) - int64
            String of 3d acid indices
        mask : (None, N) - bool
            String of mask for indicating current acid.
        """
        self.N = protein_length
        super(Lattice, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Lattice, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.map_fn(self._make_lattice, inputs,
                         dtype=tf.float32,
                         parallel_iterations=32,
                         back_prop=False,
                         swap_memory=False,
                         infer_shape=True)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 2 * self.N - 1, 2 * self.N - 1, 2 * self.N - 1

    def _make_lattice(self, data):
        acids, idx, mask = data
        idx = idx + (self.N - 1)
        masked_idx = tf.boolean_mask(idx, mask, axis=0)
        masked_acids = tf.boolean_mask(acids, mask, axis=0)
        return tf.sparse_to_dense(masked_idx, (2 * self.N - 1, 2 * self.N - 1, 2 * self.N - 1), masked_acids,
                                  validate_indices=False)


class LatticeSnake(Layer):
    def __init__(self, protein_length, window_size, **kwargs):
        self.N = protein_length
        self.K = window_size
        super(LatticeSnake, self).__init__(**kwargs)

    def build(self, input_shape):
        self.SIZET = tf.constant(np.repeat(np.int64(self.K), 3), dtype=tf.int64, shape=(3,))
        self.DIFFT = tf.constant(np.repeat(np.int64((self.K - 1) / 2), 3), dtype=tf.int64, shape=(3,))
        super(LatticeSnake, self).build(input_shape)

    def call(self, inputs, **kwargs):
        acids, mask, idx = inputs

        # Preprocess Inputs
        idx = 2 * (idx + (self.N - 1))
        float_mask = tf.cast(mask, tf.float32)
        float_mask = tf.reshape(float_mask, (-1, self.N, 1, 1, 1))

        inter_idx = tf.cast(tf.nn.pool(tf.cast(idx, tf.float32), [2], "AVG", "VALID"), tf.int64)

        inter_values = tf.nn.pool(tf.expand_dims(acids, 2), [2], "AVG", "VALID")
        inter_values = tf.reshape(inter_values, (-1, 9))
        inter_values = 2 * inter_values + 1

        inter_mask = mask[:, 1:]

        data = (acids, mask, idx, inter_values, inter_mask, inter_idx, float_mask)
        output = tf.map_fn(self._extractSnake, data, dtype=tf.float32, parallel_iterations=32,
                           back_prop=False, swap_memory=False, infer_shape=True)
        return tf.expand_dims(output, -1)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        return batch_size, self.N, self.K, self.K, self.K, 1

    def _extractSnake(self, data):
        acids, mask, idx, inter_values, inter_mask, inter_idx, float_mask = data

        masked_idx = tf.boolean_mask(idx, mask, axis=0)
        masked_acids = tf.boolean_mask(acids, mask, axis=0)

        masked_inter_idx = tf.boolean_mask(inter_idx, inter_mask, axis=0)
        masked_inter_values = tf.boolean_mask(inter_values, inter_mask, axis=0)

        sparse_idx = tf.concat((masked_idx, masked_inter_idx), axis=0)
        sparse_values = tf.concat((masked_acids, masked_inter_values), axis=0)

        lattice = tf.SparseTensor(sparse_idx, sparse_values, (4 * self.N - 3, 4 * self.N - 3, 4 * self.N - 3))
        lattice = tf.sparse_reorder(lattice)

        def extractRegion(offset_idx):
            result = tf.sparse_tensor_to_dense(tf.sparse_slice(lattice, offset_idx, self.SIZET),
                                               default_value=0, validate_indices=False)
            result.set_shape((self.K, self.K, self.K))
            return result

        regions = tf.map_fn(extractRegion, idx - self.DIFFT, dtype=tf.float32, parallel_iterations=1,
                            back_prop=False, infer_shape=True, swap_memory=False)
        regions = regions * float_mask
        return regions

class RemoveMask(Lambda):
    def __init__(self):
        super(RemoveMask, self).__init__((lambda x, mask: x))
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return None

class BooleanMask(Layer):
    def __init__(self, **kwargs):
        super(BooleanMask, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        _, mask = inputs
        return mask

    def call(self, inputs, **kwargs):
        data, mask = inputs
        mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        return data * mask

    def compute_output_shape(self, input_shape):
        return input_shape[0]

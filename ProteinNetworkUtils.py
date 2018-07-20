import tensorflow as tf
import numpy as np
from tensorflow import keras

lattice_module = tf.load_op_library('./LatticeSnakeOp/LatticeSnake.so')


class Lattice(keras.layers.Layer):
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


class LatticeSnake(keras.layers.Layer):
    def __init__(self, protein_length, window_size, num_threads=4, **kwargs):
        self.protein_length = protein_length
        self.window_size = window_size
        self.num_threads=num_threads
        super(LatticeSnake, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        acids, mask, idx = inputs

        idx = 2 * (idx + (self.protein_length - 1))
        idx = tf.cast(idx, tf.int32)

        inter_idx = tf.cast(tf.nn.pool(tf.cast(idx, tf.float32), [2], "AVG", "VALID"), tf.int32)

        inter_values = tf.nn.pool(tf.expand_dims(acids, 2), [2], "AVG", "VALID")
        inter_values = tf.squeeze(inter_values, 2)
        inter_values = 2 * inter_values + 1

        combined_idx = tf.concat([idx, inter_idx], axis=1)
        combined_values = tf.concat([acids, inter_values], axis=1)
        combined_mask = tf.concat([mask, mask[:, 1:]], axis=1)

        def extract_snake(data):
            return lattice_module.lattice_snake(data[0], data[1], data[2], self.protein_length, self.window_size)

        lattice = tf.map_fn(extract_snake, (combined_values, combined_mask, combined_idx), dtype=tf.float32,
                            parallel_iterations=self.num_threads, back_prop=False, infer_shape=True)
        return tf.expand_dims(lattice, -1)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        return batch_size, self.protein_length, self.window_size, self.window_size, self.window_size, 1

class LatticeSnake_old(keras.layers.Layer):
    def __init__(self, protein_length, window_size, **kwargs):
        self.N = protein_length
        self.K = window_size
        super(LatticeSnake_old, self).__init__(**kwargs)

    def build(self, input_shape):
        self.SIZET = tf.constant(np.repeat(np.int64(self.K), 3), dtype=tf.int64, shape=(3,))
        self.DIFFT = tf.constant(np.repeat(np.int64((self.K - 1) / 2), 3), dtype=tf.int64, shape=(3,))
        super(LatticeSnake_old, self).build(input_shape)

    def call(self, inputs, **kwargs):
        acids, mask, idx = inputs

        # Preprocess Inputs
        idx = 2 * (idx + (self.N - 1))
        float_mask = tf.cast(mask, tf.float32)
        float_mask = tf.reshape(float_mask, (-1, self.N, 1, 1, 1))

        inter_idx = tf.cast(tf.nn.pool(tf.cast(idx, tf.float32), [2], "AVG", "VALID"), tf.int64)

        inter_values = tf.nn.pool(tf.expand_dims(acids, 2), [2], "AVG", "VALID")
        inter_values = tf.reshape(inter_values, (-1, self.N-1))
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

        lattice_dim = 4 * self.N - 3 + self.N
        lattice = tf.SparseTensor(sparse_idx, sparse_values, (lattice_dim, lattice_dim, lattice_dim))
        lattice = tf.sparse_reorder(lattice)

        def extractRegion(offset_idx):
            result = tf.sparse_tensor_to_dense(tf.sparse_slice(lattice, offset_idx, self.SIZET),
                                               default_value=0, validate_indices=False)
            result.set_shape((self.K, self.K, self.K))
            return result

        offset_idx = idx - self.DIFFT
        regions = tf.stack([extractRegion(offset_idx[i]) for i in range(self.N)])
        # regions = tf.map_fn(extractRegion, idx - self.DIFFT, dtype=tf.float32, parallel_iterations=1,
        #                     back_prop=False, infer_shape=True, swap_memory=False)
        regions = regions * float_mask
        return regions

class RemoveMask(keras.layers.Lambda):
    def __init__(self):
        super(RemoveMask, self).__init__((lambda x, mask: x))
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return None

class BooleanMask(keras.layers.Layer):
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



def eval_energy(state, energy_distance=2):
    idx = state[2:]
    mask1 = state[0]-2
    mask2 = state[1]
    idx = tf.cast(idx, tf.float64)
    mask1 = tf.cast(mask1, tf.bool)
    mask2 = tf.cast(mask2, tf.bool)
    mask = mask1 & mask2
    idx = tf.transpose(idx)
    na = tf.reduce_sum(tf.square(idx), 1)
    # casting as a row and column vectors
    row = tf.reshape(na, [-1, 1])
    col = tf.reshape(na, [1, -1])
    # return pairwise euclidean difference matrix
    result = tf.sqrt(tf.maximum(row - 2 * tf.matmul(idx, idx, False, True) + col, 0.0))
    other = tf.matrix_band_part(result, -1, 0)
    other = tf.matrix_set_diag(other[1:,:-1], np.zeros(other.shape[0]-1))
    result2 = tf.boolean_mask(other, mask[1:], axis=0)
    result3 = tf.boolean_mask(result2, mask[:-1], axis=1)
    final1 = tf.less_equal(result3, tf.constant(energy_distance, tf.float64))
    final2 = tf.greater(result3, tf.constant(0, tf.float64))
    return tf.count_nonzero(final1 & final2)


import tensorflow as tf
from keras.engine.topology import Layer

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
        return input_shape[0], 2 * self.N - 1, 2 * self.N - 1, 2 * self.N - 1

    def _make_lattice(self, data):
        acids, idx, mask = data
        idx = idx + (self.N - 1)
        masked_idx = tf.boolean_mask(idx, mask, axis=0)
        masked_acids = tf.boolean_mask(acids, mask, axis=0)
        return tf.sparse_to_dense(masked_idx, (2 * self.N - 1, 2 * self.N - 1, 2 * self.N - 1), masked_acids,
                                  validate_indices=False)
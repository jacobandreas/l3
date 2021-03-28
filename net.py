import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

INIT_SCALE = 1.43

def _linear(t_in, n_out):
    v_w = tf.get_variable(
            "w",
            shape=(t_in.get_shape()[-1], n_out),
            initializer=tf.uniform_unit_scaling_initializer(
                factor=INIT_SCALE))
    v_b = tf.get_variable(
            "b",
            shape=n_out,
            initializer=tf.constant_initializer(0))
    if len(t_in.get_shape()) == 2:
        return tf.einsum("ij,jk->ik", t_in, v_w) + v_b
    elif len(t_in.get_shape()) == 3:
        return tf.einsum("ijk,kl->ijl", t_in, v_w) + v_b
    else:
        assert False

def _embed(t_in, n_embeddings, n_out):
    v = tf.get_variable(
            "embed", shape=(n_embeddings, n_out),
            initializer=tf.uniform_unit_scaling_initializer())
    t_embed = tf.nn.embedding_lookup(v, t_in)
    return t_embed

def _embed_dict(t_in, emb_dict):
    if isinstance(emb_dict, tf.Variable):
        v = emb_dict
    else:
        assert isinstance(emb_dict, np.array)
        v = tf.get_variable(
                "embed", shape=emb_dict.shape,
                initializer=tf.constant_initializer(emb_dict))

    t_embed = tf.nn.embedding_lookup(v, t_in)
    return t_embed

def _mlp(t_in, widths, activations):
    assert len(t_in.get_shape()) in (2, 3)
    assert len(widths) == len(activations)
    prev_width = t_in.get_shape()[1]
    prev_layer = t_in
    for i_layer, (width, act) in enumerate(zip(widths, activations)):
        with tf.variable_scope(str(i_layer)):
            layer = _linear(prev_layer, width)
            if act is not None:
                layer = act(layer)
        prev_layer = layer
        prev_width = width
    return prev_layer

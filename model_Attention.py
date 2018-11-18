import numpy as np
import tensorflow as tf
import keras.backend as K
np.random.seed(1337)  # for reproducibility
from keras.models import *
from keras.layers import Input, Dense, multiply

input_dim = 32

def get_data(n, input_dim, attention_column=1):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column] = y[:, 0]
    return x, y

def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def build_model(inputs, seq_len):
    inputs = Input(shape=(input_dim,))

    # ATTENTION PART STARTS HERE
    attention_probs = Dense(seq_len, activation='softmax', name='attention_vec')(inputs)
    attention_mul = concatenate([inputs, attention_probs], output_shape=32, name='attention_mul', mode='mul')
    # ATTENTION PART FINISHES HERE

    attention_mul = Dense(64)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


def Model(inputs,
        seq_len,
        num_classes=30,
        num_hidden = 100,
        num_layers = 1,
        batch_size = 1,
        is_training=True,
        scope='model_Attention'):
    with tf.variable_scope(scope, 'model_Attention', [inputs, seq_len]) as sc:
        #N = 10000
        #inputs_1, outputs = get_data(N, num_classes)

        inputs_new = Input(shape=(2,))
        # ATTENTION PART STARTS HERE
        attention_probs = Dense(2, activation='softmax', name='attention_vec')(inputs_new)
        #attention_mul = add([inputs_new, attention_probs], output_shape=32, name='attention_mul', mode='mul')
        attention_mul = multiply([inputs_new, attention_probs])
        # ATTENTION PART FINISHES HERE

        attention_mul = Dense(64)(attention_mul)
        output = Dense(1, activation='sigmoid')(attention_mul)
        #m = Model(input=[inputs], output=output)
        m = Model([inputs_new], output)

        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        #print(m.summary())

        m.fit([inputs], outputs, epochs=10, batch_size=64, validation_split=0.5)

        testing_inputs_1, testing_outputs = get_data(1, seq_len)

        # Attention vector corresponds to the second matrix.
        # The first one is the Inputs output.
        attention_vector = get_activations(m, testing_inputs_1,
                                           print_shape_only=True,
                                           layer_name='attention_vec')[0].flatten()
        print('attention =', attention_vector)

    # plot part.
    '''import matplotlib.pyplot as plt
    import pandas as pd

    pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar',
                                                                   title='Attention Mechanism as '
                                                                         'a function of input'
                                                                         ' dimensions.')
    plt.show()'''

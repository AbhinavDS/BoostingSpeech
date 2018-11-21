import os
import numpy as np

import tensorflow as tf
from tensorflow.python.layers.core import Dense
rnn_size_encoder = 256
rnn_size_decoder = 256
def make_rnn_cell(rnn_size):
    cell = tf.nn.rnn_cell.LSTMCell(rnn_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                         input_keep_prob=0.825,
                                         output_keep_prob=0.895,
                                         state_keep_prob=0.86)
    return cell

def make_attention_cell(dec_cell, rnn_size, enc_output, lengths):
    """Wraps the given cell with Bahdanau Attention.
    """
    print("dec_cell::",dec_cell," enc_output", enc_output)
    print("rnn_size::",rnn_size," lengths", lengths)
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=rnn_size,
                                                               memory=enc_output,
                                                               name='BahdanauAttention')

    return tf.contrib.seq2seq.AttentionWrapper(cell=dec_cell,
                                               attention_mechanism=attention_mechanism,
                                               attention_layer_size=None,
                                               output_attention=False)

def blstm(inputs,
          input_sequence_length,
          n_hidden,
          scope=None,
          initial_state_fw=None,
          initial_state_bw=None):
    
    fw_cell = make_rnn_cell(n_hidden)
    bw_cell = make_rnn_cell(n_hidden)
    print("type:",type(seq_len))
    (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=fw_cell,
        cell_bw=bw_cell,
        inputs=inputs,
        sequence_length=input_sequence_length,
        initial_state_fw=initial_state_fw,
        initial_state_bw=initial_state_bw,
        dtype=tf.float32,
        scope=scope
    )

    return (out_fw, out_bw), (state_fw, state_bw)


def reshape_pyramidal(outputs, sequence_length):
    shape = tf.shape(outputs)
    batch_size, max_time = shape[0], shape[1]
    num_units = outputs.get_shape().as_list()[-1]

    pads = [[0, 0], [0, tf.floormod(max_time, 2)], [0, 0]]
    outputs = tf.pad(outputs, pads)

    concat_outputs = tf.reshape(outputs, (batch_size, -1, num_units * 2))
    return concat_outputs, tf.floordiv(sequence_length, 2) + tf.floormod(sequence_length, 2)

def Model(inputs,
    seq_len,
    input_sequence_length,
    char_ids,
    maximum_iterations,
    num_classes=28,
    num_hidden = 100,
    num_layers = 1,
    batch_size = 1,
    is_training=True,
    num_encoder_layers = 1,
    num_decoder_layers = 1,
    scope='model_attention'):
    with tf.variable_scope(scope, 'model_attention', [inputs, seq_len]) as sc:
        # Encoder
        encoder_outputs, encoder_state = build_encoder(inputs, seq_len, num_encoder_layers)

        # Decoder
        logits, _, _ = build_decoder(encoder_outputs,encoder_state,num_classes, num_decoder_layers, maximum_iterations)            
        return logits

def build_encoder(inputs, seq_len, input_sequence_length, num_encoder_layers):
    
    # Pyramidal bidirectional LSTM(s)

    initial_state_fw = None
    initial_state_bw = None

    global rnn_size_encoder

    for n in range(num_encoder_layers):
        scope = 'pBLSTM' + str(n)
        (out_fw, out_bw), (state_fw, state_bw) = blstm(
            inputs,
            input_sequence_length,
            rnn_size_encoder // 2,
            scope=scope,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw
        )

        inputs = tf.concat([out_fw, out_bw], -1)
        inputs, input_sequence_length = reshape_pyramidal(inputs, input_sequence_length)
        initial_state_fw = state_fw
        initial_state_bw = state_bw


    bi_state_c = tf.concat((initial_state_fw.c, initial_state_fw.c), -1)
    bi_state_h = tf.concat((initial_state_fw.h, initial_state_fw.h), -1)
    bi_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=bi_state_c, h=bi_state_h)
    encoder_state = tuple([bi_lstm_state] * num_encoder_layers)

    return inputs, encoder_state



def build_decoder(encoder_outputs, encoder_state, num_classes, num_decoder_layers,maximum_iterations):

    out_layer = Dense(num_classes+1, name='output_projection')

    # Decoder.
    with tf.variable_scope("decoder") as decoder_scope:

        cell, decoder_initial_state = build_decoder_cell(
            encoder_outputs,
            encoder_state,
            input_sequence_length,
            num_decoder_layers,
            batch_size)

        # Train
        # if mode != 'INFER':
        char_ids = tf.placeholder(tf.int32,
                                       shape=[None, None],
                                       name='ids_target')
        embedding = tf.get_variable('embedding',
                                    shape=[num_classes+1, 2], # embeddings dimension I have given 2
                                    dtype=tf.float32)


        char_embedding_lookup = tf.nn.embedding_lookup(embedding,
                                                char_ids,
                                                name='char_embedding')
        char_embedding = tf.nn.dropout(char_embedding_lookup,
                                            0.986,
                                            name='char_embedding_dropout')

        helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs=char_embedding,
            sequence_length=num_classes,
            embedding=embedding,
            sampling_probability=0.5,
            time_major=False)

        # Decoder
        my_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                     helper,
                                                     decoder_initial_state,
                                                     output_layer=out_layer)

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            output_time_major=False,
            maximum_iterations=maximum_iterations,
            swap_memory=False,
            impute_finished=True,
            scope=decoder_scope
        )

        sample_id = outputs.sample_id
        logits = outputs.rnn_output


        # Inference
        # else:
        #     start_tokens = tf.fill([batch_size], sos_id_2)
        #     end_token = eos_id_2

        #     # Beam search
        #     if beam_width > 0:
        #         my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        #             cell=cell,
        #             embedding=embedding,
        #             start_tokens=start_tokens,
        #             end_token=end_token,
        #             initial_state=decoder_initial_state,
        #             beam_width=beam_width,
        #             output_layer=output_layer,
        #         )

        #     # Greedy
        #     else:
        #         helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding,
        #                                                           start_tokens,
        #                                                           end_token)

        #         my_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
        #                                                      helper,
        #                                                      decoder_initial_state,
        #                                                      output_layer=output_layer)
        #     if inference_targets:
        #         maximum_iterations = maximum_iterations
        #     else:
        #         maximum_iterations = None

        #     # Dynamic decoding
        #     outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
        #         my_decoder,
        #         maximum_iterations=maximum_iterations,
        #         output_time_major=False,
        #         impute_finished=False,
        #         swap_memory=False,
        #         scope=decoder_scope)

        #     if beam_width > 0:
        #         logits = tf.no_op()
        #         sample_id = outputs.predicted_ids
        #     else:
        #         logits = tf.no_op()
        #         sample_id = outputs.sample_id

    return logits, sample_id, final_context_state

def build_decoder_cell(encoder_outputs, encoder_state,
                       input_sequence_length, num_decoder_layers, batch_size):
    """Builds the attention decoder cell. If mode is inference performs tiling
       Passes last encoder state.
    """

    memory = encoder_outputs
    global rnn_size_decoder
    # if mode == 'INFER' and beam_width > 0:
    #     memory = tf.contrib.seq2seq.tile_batch(memory,
    #                                            multiplier=beam_width)
    #     encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state,
    #                                                   multiplier=beam_width)
    #     audio_sequence_lengths = tf.contrib.seq2seq.tile_batch(audio_sequence_lengths,
    #                                                            multiplier=beam_width)
    #     batch_size = batch_size * beam_width

    # else:
    # batch_size = 1

    # if num_layers_decoder is not None:
    # lstm_cell = [tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True) for n in range(num_layers)]

    # else:
    #     lstm_cell = make_rnn_cell(rnn_size_decoder)

    # attention cell
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
                [make_rnn_cell(rnn_size_decoder) for _ in
                 range(num_decoder_layers)])
    cell = make_attention_cell(lstm_cell,
                                    rnn_size_decoder,
                                    memory,
                                    input_sequence_length) 

    decoder_initial_state = cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

    return cell, decoder_initial_state

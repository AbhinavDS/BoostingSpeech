import tensorflow as tf

def Model(inputs,
		seq_len,
		num_classes=28,
		num_hidden = 100,
		num_layers = 1,
		batch_size = 1,
		is_training=True,
		scope='model_bigru'):
	with tf.variable_scope(scope, 'model_bigru', [inputs, seq_len]) as sc:
		# Defining the cell
		# Can be:
		#   tf.nn.rnn_cell.RNNCell
		#   tf.nn.rnn_cell.GRUCell

		fw_cells = cells = [tf.contrib.rnn.GRUCell(num_hidden, activation=tf.nn.relu6) for n in range(num_layers)]
		bw_cells = cells = [tf.contrib.rnn.GRUCell(num_hidden, activation=tf.nn.relu6) for n in range(num_layers)]
        #stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
		stack_fw = tf.contrib.rnn.MultiRNNCell(fw_cells)
		stack_bw = tf.contrib.rnn.MultiRNNCell(bw_cells)
		# The second output is the last state and we will no use that
		outputs, _ = tf.nn.bidirectional_dynamic_rnn(stack_fw,stack_bw,inputs, seq_len, dtype=tf.float32)

		shape = tf.shape(inputs)
		batch_s, max_time_steps = shape[0], shape[1]

		# Reshaping to apply the same weights over the timesteps
		outputs = tf.reshape(outputs, [-1, num_hidden])

		# Truncated normal with mean 0 and stdev=0.1
		# Tip: Try another initialization
		# see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
		W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
		# Zero initialization
		# Tip: Is tf.zeros_initializer the same?
		b = tf.Variable(tf.constant(0., shape=[num_classes]))

		# Doing the affine projection
		logits = tf.matmul(outputs, W) + b

		# Reshaping back to the original shape
		logits = tf.reshape(logits, [batch_s, -1, num_classes])

		# Time major
		logits = tf.transpose(logits, (1, 0, 2))

		return logits

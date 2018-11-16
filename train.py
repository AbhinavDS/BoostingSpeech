import time
import numpy as np
import tensorflow as tf
import model1 
# DATA
data_x1 = np.load("TEDLIUM_release1/test/sph/mfcc.npy")
data_x2 = np.load("TEDLIUM_release1/test/sph/spec.npy")
data_y  = np.load("TEDLIUM_release1/test/stm/label.npy")
data_x1 = np.transpose(data_x1, axes=[0, 2, 1])[0:1]
data_x2 = np.transpose(data_x2, axes=[0, 2, 1])[0:1]
data_y = data_y[0:1]

# Get sequence length
data_y_len = np.zeros((data_y.shape[0],1), dtype=int)
for i in range(len(data_y)):
	for j in range(len(data_y[i])-1, -1, -1):
		if data_y[i][j] != -1:
			data_y_len[i] = j+1
			break

print (data_x1.shape, data_x2.shape, data_y.shape, data_y_len.shape)

# Some configs
num_features = data_x1.shape[-1]
# Accounting the 0th index +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters
num_epochs = 10000
num_hidden = 100
num_layers = 1
batch_size = 1

num_examples = 1
num_batches_per_epoch = int(num_examples / batch_size)

counter = 0
total_len = len(data_y)

def run_ctc():
	graph = tf.Graph()
	global counter, total_len
	with graph.as_default():
		# Has size [batch_size, max_step_size, num_features], but the
		# batch_size and max_step_size can vary along each step
		inputs = tf.placeholder(tf.float32, [None, None, num_features])

		# Here we use sparse_placeholder that will generate a
		# SparseTensor required by ctc_loss op.
		targets = tf.sparse_placeholder(tf.int32)
		
		# 1d array of size [batch_size]
		seq_len = tf.placeholder(tf.int32, [None])


		logits = model1.Model(inputs, seq_len, num_classes=num_classes, num_hidden=num_hidden, num_layers=num_layers)
		

		loss = tf.nn.ctc_loss(targets, logits, seq_len)
		cost = tf.reduce_mean(loss)

		# optimizer = tf.train.AdamOptimizer().minimize(cost)
		# optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(cost)
		optimizer = tf.train.MomentumOptimizer(learning_rate=0.005, momentum=0.9).minimize(cost)

		# Option 2: tf.contrib.ctc.ctc_beam_search_decoder
		# (it's slower but you'll get better results)
		decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

		# Inaccuracy: label error rate
		ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
											  targets))

	def sparse_tuple_from(sequences, dtype=np.int32):
		indices = []
		values = []

		for n, seq in enumerate(sequences):
			indices.extend(zip([n] * len(seq), range(len(seq))))
			values.extend(seq)

		indices = np.asarray(indices, dtype=np.int64)
		values = np.asarray(values, dtype=dtype)
		shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
		return (indices, values, shape)

	def next_training_batch():
		global counter, total_len
		counter += 1
		counter %= total_len
		target = data_y[counter:counter+1]
		indices, values, shape = sparse_tuple_from(target)
		# target = tf.SparseTensor(indices=indices, values=values, shape=shape) 
                target = (indices, values, shape)
                print('target::',target)
		return data_x1[counter:counter+1], target, data_y_len[counter:counter+1], data_y[counter:counter+1]

	def next_testing_batch():
		# for now testing and training on same
		return next_training_batch()

	with tf.Session(graph=graph) as session:
		tf.global_variables_initializer().run()

		for curr_epoch in range(num_epochs):
			train_cost = train_ler = 0
			start = time.time()

			for batch in range(num_batches_per_epoch):
				train_inputs, train_targets, train_seq_len, original = next_training_batch()
				feed = {inputs: train_inputs,
						targets: train_targets,
						seq_len: train_seq_len}
                                print ('FEED::', feed)

				batch_cost, _ = session.run([cost, optimizer], feed)
				train_cost += batch_cost * batch_size
				train_ler += session.run(ler, feed_dict=feed) * batch_size

				# Decoding
				d = session.run(decoded[0], feed_dict=feed)
				str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
				# Replacing blank label to none
				str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
				# Replacing space label to space
				str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

				print('Original: %s' % original)
				print('Decoded: %s' % str_decoded)

			train_cost /= num_examples
			train_ler /= num_examples

			val_inputs, val_targets, val_seq_len, val_original, random_shift = next_testing_batch()
			val_feed = {inputs: val_inputs,
						targets: val_targets,
						seq_len: val_seq_len}

			val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)

			# Decoding
			d = session.run(decoded[0], feed_dict=val_feed)
			str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
			# Replacing blank label to none
			str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
			# Replacing space label to space
			str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

			print('Original val: %s' % val_original)
			print('Decoded val: %s' % str_decoded)

			log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, " \
				  "val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"


			print(log.format(curr_epoch + 1, num_epochs, train_cost, train_ler,
							 val_cost, val_ler, time.time() - start))


if __name__ == '__main__':
	run_ctc()

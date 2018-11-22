import time
import os
import numpy as np
import tensorflow as tf
from preprocessing_yield import data_generator
from ensemble_yield import ensemble_data_generator

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model_paths",  nargs='+', required=True, help="Path of stored logits dir")
args = vars(ap.parse_args())

# select model
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1 


# Some configs
feature = 'spec'
if feature == 'spec':
	num_features = 128
else:
	num_features = 50

# Accounting the 0th index +  space + blank label + eos = 29 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1 + 1 + 1

# Hyper-parameter
num_hidden = 1 #not really used
num_layers = 1 #not really used
batch_size = 128
num_models = len(args["model_paths"])
maxlen_input = 1000

test_data_gen = data_generator(text_dir='TEDLIUM_release1/test/stm', speech_dir='TEDLIUM_release1/test/sph', batch_size=batch_size, feature=feature, num_features=num_features)
ensemble_data_gen = ensemble_data_generator(args["model_paths"], batch_size=batch_size)

def run_ctc():
	graph = tf.Graph()
	with graph.as_default():
		# Has size [num_models, num_features, batch_size, num_classes], but the
		# batch_size can vary along each step
		all_logits = tf.placeholder(tf.float32, [num_models, maxlen_input, None, num_classes])
		
		# Here we use sparse_placeholder that will generate a
		# SparseTensor required by ctc_loss op.
		targets = tf.sparse_placeholder(tf.int32)
		
		# 1d array of size [batch_size]
		seq_len = tf.placeholder(tf.int32, [None])
		logits = tf.reduce_mean(all_logits, 0) # Has size [num_features, batch_size, num_classes]
		
		loss = tf.nn.ctc_loss(targets, logits, seq_len)
		cost = tf.reduce_mean(loss)

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

	def next_testing_batch():
		global test_data_gen
		_, data_y, len_y, epoch_num = next(test_data_gen)
		ensemble_logits = next(ensemble_data_gen)
		target = sparse_tuple_from(data_y)
		return ensemble_logits, target, len_y, data_y, epoch_num

	best_ler = 2.0
	with tf.Session(graph=graph) as session:
		tf.global_variables_initializer().run()
		writer = tf.summary.FileWriter("output", session.graph)
		# Add ops to save and restore all the variables.
		for curr_epoch in range(1):
			print ("Starting Testing")	
			test_cost = 0
			test_ler = 0
			start = time.time()
			num_examples = 0
			epoch_num = curr_epoch
			while(epoch_num<=curr_epoch):
				print ("Total Examples seen: ",num_examples)
				test_inputs, test_targets, test_seq_len, original, epoch_num = next_testing_batch()
				feed = {all_logits: test_inputs,
						targets: test_targets,
						seq_len: test_seq_len}

				
				batch_cost = session.run(cost, feed)
				test_cost += batch_cost * len(original)
				test_ler += (session.run(ler, feed_dict=feed) * len(original))

				# Decoding
				d = session.run(decoded[0], feed_dict=feed)
				str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
				# Replacing blank label to none
				str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
				# Replacing space label to space
				str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

				num_examples += len(original)
				# print('Original: %s' % original)
				# print('Decoded: %s' % str_decoded)
			test_cost /= num_examples
			test_ler /= num_examples

			log = "Epoch {}/{}, test_cost = {:.3f}, test_ler = {:.3f}, time = {:.3f}"
			print ("Total Examples seen: ",num_examples)
				
			print(log.format(curr_epoch + 1, 1, test_cost, test_ler, time.time() - start))
		
		writer.close()

if __name__ == '__main__':
        run_ctc()

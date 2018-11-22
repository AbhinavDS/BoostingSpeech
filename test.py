import time
import os
import numpy as np
import tensorflow as tf
from preprocessing_yield import data_generator
import shutil

import argparse

ap = argparse.ArgumentParser()
list_of_choices = ["LSTM", "GRU", "BILSTM", "BIGRU", "ATTN"]
ap.add_argument("-n", "--cell", required=True, help="name of the cell unit",  choices=list_of_choices)
ap.add_argument("-ckpt", "--checkpoint", required=True, help="path of checkpoint")
ap.add_argument("-logits", "--logits_dir", default="logits/", help="path of logits dir")
# ap.add_argument("-h", "--num-hidden", required=True, help="number of hidden cell unit")
#ap.add_argument("-l", "--num-layers", required=True, help="name of layers")
#ap.add_argument("-e","--num-epochs",required=True, help="number of epochs")
#ap.add_argument("-e","--feature",required=True, help="mfcc or spec")
#ap.add_argument("-e","--lear",required=True, help="learning rate --0.00001")

args = vars(ap.parse_args())

# select model
cell_name = args["cell"]
if cell_name == 'LSTM':
	print ('Using LSTM cells')
	import model_lstm as model
elif cell_name == 'GRU':
	print ('Using GRU cells')
	import model_gru as model
elif cell_name == 'BILSTM':
	print ('Using Bi-LSTM cells')
	import model_bilstm as model
elif cell_name == 'BIGRU':
	print ('Using Bi-GRU cells')
	import model_bigru as model
elif cell_name == 'ATTN':
	print ('using Attention')
	import model_attention as model


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
num_hidden = 100#args["num-hidden"]
num_layers = 1#args["num-layers"]
batch_size = 128

test_data_gen = data_generator(text_dir='TEDLIUM_release1/test/stm', speech_dir='TEDLIUM_release1/test/sph', batch_size=batch_size, feature=feature, num_features=num_features)

def run_ctc():
	graph = tf.Graph()
	with graph.as_default():
		# Has size [batch_size, max_step_size, num_features], but the
		# batch_size and max_step_size can vary along each step
		inputs = tf.placeholder(tf.float32, [None, None, num_features])

		# Here we use sparse_placeholder that will generate a
		# SparseTensor required by ctc_loss op.

		targets = tf.sparse_placeholder(tf.int32)
		
		# 1d array of size [batch_size]
		seq_len = tf.placeholder(tf.int32, [None])
		logits = model.Model(inputs, seq_len, num_classes=num_classes, num_hidden=num_hidden, num_layers=num_layers)
		
		loss = tf.nn.ctc_loss(targets, logits, seq_len)
		cost = tf.reduce_mean(loss)

		# optimizer = tf.train.AdamOptimizer().minimize(cost)
		# optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(cost)
		optimizer = tf.train.MomentumOptimizer(learning_rate=1e-4, momentum=0.9).minimize(cost)
		# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

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
		data_x, data_y, len_y, epoch_num = next(test_data_gen)
		target = sparse_tuple_from(data_y)
		return data_x, target, len_y, data_y, epoch_num

	best_ler = 2.0
	ckpt_path = args["checkpoint"]
	ckpt_name = os.path.basename(ckpt_path).split('.ckpt')[0]
	logits_dir = os.path.join(args["logits_dir"], ckpt_name)
	if os.path.exists(logits_dir):
		shutil.rmtree(logits_dir)
	os.makedirs(logits_dir)

	logits_path = os.path.join(logits_dir, "logits.npy")
	with tf.Session(graph=graph) as session:
		tf.global_variables_initializer().run()
		writer = tf.summary.FileWriter("output", session.graph)
		# Add ops to save and restore all the variables.
		saver = tf.train.Saver()
		saver.restore(session, ckpt_path)
		for curr_epoch in range(1):
			print ("Starting Testing")	
			test_cost = 0
			test_ler = 0
			start = time.time()
			num_examples = 0
			num_i = 0
			epoch_num = curr_epoch
			while(epoch_num<=curr_epoch):
				print ("Total Examples seen: ",num_examples)
				test_inputs, test_targets, test_seq_len, original, epoch_num = next_testing_batch()
				feed = {inputs: test_inputs,
						targets: test_targets,
						seq_len: test_seq_len}

				
				batch_cost, batch_logits = session.run([cost, logits], feed)

				# Save the logits
				logits_path = os.path.join(logits_dir, "logits_%i.npy"%num_i)
				np.save(logits_path, batch_logits)
				num_i += 1

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
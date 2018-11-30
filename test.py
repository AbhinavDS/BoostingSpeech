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
ap.add_argument("-f","--feature", nargs='?', help="mfcc or spec", default="spec", choices=["mfcc", "spec"])
ap.add_argument("-ckpt", "--checkpoint", required=True, help="path of checkpoint")
ap.add_argument("-logits", "--logits_dir", default="logits/", help="path of logits dir")
ap.add_argument("-log", "--log_file", required=True, default="out.log", help="path of log_file")
ap.add_argument("-nh", "--num_hidden",  nargs='?', type=int, default=100, help="number of hidden cell unit")
ap.add_argument("-nl", "--num_layers", nargs='?', type=int, default=1, help="name of layers")
ap.add_argument("-bs", "--batch_size", nargs='?', type=int, default=128, help="batch_size")
ap.add_argument("-maxf", "--max_feature_len", nargs='?', type=int, default=500, help="maximum timesteps for mfcc or spec per data point")
ap.add_argument("-maxs", "--max_seq_len", nargs='?', type=int, default=500, help="maximum timesteps for target")

args = vars(ap.parse_args())

# select model
def log_print(string):
	log_file = args["log_file"]
	f = open(log_file,"a")
	f.write(string+"\n")
	f.close()

args_print = "ARGS::\n"
for key in args.keys():
	args_print += str(key)+"::"+str(args[key])+"\n"
log_print(args_print)

# select model
cell_name = args["cell"]
if cell_name == 'LSTM':
	log_print ('Using LSTM cells')
	import model_lstm as model
elif cell_name == 'GRU':
	log_print ('Using GRU cells')
	import model_gru as model
elif cell_name == 'BILSTM':
	log_print ('Using Bi-LSTM cells')
	import model_bilstm as model
elif cell_name == 'BIGRU':
	log_print ('Using Bi-GRU cells')
	import model_bigru as model
elif cell_name == 'ATTN':
	log_print ('using Attention')
	import model_attention as model


SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1 

# Hyper-parameter
num_hidden = args["num_hidden"]
num_layers = args["num_layers"]
batch_size = args["batch_size"]
feature = args["feature"]
max_feature_len = args["max_feature_len"]
max_seq_len = args["max_seq_len"]
if feature == 'spec':
	num_features = 128
else:
	num_features = 40

learning_rate = 0.1
# Accounting the 0th index +  space + blank label + eos = 29 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1 + 1 + 1

test_data_gen = data_generator(text_dir='TEDLIUM_release1/test/stm', speech_dir='TEDLIUM_release1/test/sph', batch_size=batch_size, feature=feature, num_features=num_features, overfit=False, maxlen_mfcc=max_feature_len, maxlen_spec=max_feature_len, maxlen_seq=max_seq_len)

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

		if cell_name == 'ATTN':
			input_sequence_length = tf.placeholder(tf.int32, [None])
			char_ids = tf.placeholder(tf.int32,
                                       shape=[None, None],
                                       name='ids_target')
			
			logits = model.Model(inputs, seq_len, input_sequence_length, maximum_iterations, char_ids, num_classes=num_classes, num_hidden=num_hidden, num_layers=num_layers)
			logits = tf.transpose(logits, perm=[1, 0, 2])
		else:
			logits = model.Model(inputs, seq_len, num_classes=num_classes, num_hidden=num_hidden, num_layers=num_layers)
		
		loss = tf.nn.ctc_loss(targets, logits, seq_len)
		cost = tf.reduce_mean(loss)

		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
		# optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(cost)
		# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(cost)
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
		data_x, data_y, len_y, epoch_num, len_x, unpadded_data_y = next(test_data_gen)

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
			log_print ("Starting Testing")	
			test_cost = 0
			test_ler = 0
			start = time.time()
			num_examples = 0
			num_i = 0
			epoch_num = curr_epoch
			while(epoch_num<=curr_epoch):
				log_print ("Total Examples seen: %i"%num_examples)
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
				# log_print('Original: %s' % original)
				# log_print('Decoded: %s' % str_decoded)
			
			test_cost /= num_examples
			test_ler /= num_examples

			log = "Epoch {}/{}, test_cost = {:.3f}, test_ler = {:.3f}, time = {:.3f}"
			log_print ("Total Examples seen: %i"%num_examples)
				
			log_print(log.format(curr_epoch + 1, 1, test_cost, test_ler, time.time() - start))
		
		writer.close()

if __name__ == '__main__':
        run_ctc()

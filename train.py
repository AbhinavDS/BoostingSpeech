import time
import os
import numpy as np
import tensorflow as tf
from preprocessing_yield import data_generator

import argparse
import itertools

ap = argparse.ArgumentParser()
list_of_choices = ["LSTM", "GRU", "BILSTM", "BIGRU", "ATTN"]
ap.add_argument("-n", "--cell", required=True, help="name of the cell unit",  choices=list_of_choices)
ap.add_argument("-f","--feature", nargs='?', help="mfcc or spec", default="spec", choices=["mfcc", "spec"])
ap.add_argument("-ckpt", "--checkpoint", required=True, default="", help="path of checkpoint")
ap.add_argument("-s", "--suffix",  required=True, default="", help="extra suffix for model")
ap.add_argument("-log", "--log_file", required=True, default="out.log", help="path of log_file")
ap.add_argument("-dev", "--dev_set",  required=True, default="dev", help="path of dev files")
ap.add_argument("-train", "--train_set", required=True, default="train", help="path of train files")
ap.add_argument("-ne", "--num_epochs", nargs='?', type=int, default=2000, help="number of epochs")
ap.add_argument("-nh", "--num_hidden",  nargs='?', type=int, default=100, help="number of hidden cell unit")
ap.add_argument("-nl", "--num_layers", nargs='?', type=int, default=1, help="name of layers")
ap.add_argument("-bs", "--batch_size", nargs='?', type=int, default=128, help="batch_size")
ap.add_argument("-maxf", "--max_feature_len", nargs='?', type=int, default=500, help="maximum timesteps for mfcc or spec per data point")
ap.add_argument("-maxs", "--max_seq_len", nargs='?', type=int, default=500, help="maximum timesteps for target")
ap.add_argument("-lr", "--learning_rate", nargs='?', type=float, default=1e-4, help="learning rate --0.0001")
ap.add_argument('-o', "--overfit", action='store_true', default=False, dest='overfit', help='Set a switch to true')

args = vars(ap.parse_args())

# if os.path.exists(args["log_file"]):
# 	os.remove(args["log_file"])

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
suffix = args["suffix"]

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
num_epochs = args["num_epochs"]
num_hidden = args["num_hidden"]
num_layers = args["num_layers"]
batch_size = args["batch_size"]
learning_rate = args["learning_rate"]
feature = args["feature"]
overfit = args["overfit"]
max_feature_len = args["max_feature_len"]
max_seq_len = args["max_seq_len"]
if feature == 'spec':
	num_features = 128
else:
	num_features = 40

# Accounting the 0th index +  space + blank label + eos = 29 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1 + 1 + 1


train_data_gen = data_generator(text_dir='TEDLIUM_release1/%s/stm'%args["train_set"], speech_dir='TEDLIUM_release1/%s/sph'%args["train_set"], batch_size=batch_size, feature=feature, num_features=num_features, overfit=overfit, maxlen_mfcc=max_feature_len, maxlen_spec=max_feature_len, maxlen_seq=max_seq_len)
valid_data_gen = data_generator(text_dir='TEDLIUM_release1/%s/stm'%args["dev_set"], speech_dir='TEDLIUM_release1/%s/sph'%args["dev_set"], batch_size=batch_size, feature=feature, num_features=num_features, overfit=overfit, maxlen_mfcc=max_feature_len, maxlen_spec=max_feature_len, maxlen_seq=max_seq_len)
maximum_iterations=1000# max_feature_len
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

	def next_training_batch():
		global train_data_gen
		data_x, data_y, len_y, epoch_num, len_x, unpadded_data_y = next(train_data_gen)
		target = sparse_tuple_from(data_y)
		return data_x, target, len_y, data_y, epoch_num, len_x, unpadded_data_y

	def next_validation_batch():
		global valid_data_gen
		data_x, data_y, len_y, _, len_x, unpadded_data_y = next(valid_data_gen)

		target = sparse_tuple_from(data_y)
		return data_x, target, len_y, data_y, len_x, 0, unpadded_data_y

	ckpt_path = args["checkpoint"]
	last_epoch_path = ckpt_path+".last"
	best_ler = 2.0
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(graph=graph, config=config) as session:
		tf.global_variables_initializer().run()
		writer = tf.summary.FileWriter("output", session.graph)
		# Add ops to save and restore all the variables.
		last_epoch = 0
		if ckpt_path and os.path.exists(ckpt_path+'.meta'):
			log_print ("RESTORING ...")
			saver = tf.train.Saver()
			saver.restore(session, ckpt_path)
			if os.path.exists(last_epoch_path):
				f_last = open(last_epoch_path, 'r')
				last_epoch = int(f_last.readline().strip()) + 1
				best_ler = float(f_last.readline().strip())
				f_last.close()
		for curr_epoch in range(last_epoch, num_epochs):
			log_print ("Starting Epoch %i" % (curr_epoch + 1))
			train_cost = 0
			train_ler = 0
			start = time.time()
			num_examples = 0
			epoch_num = curr_epoch - last_epoch
			while(epoch_num<=curr_epoch - last_epoch):
				log_print ("Total Examples seen: %i"%num_examples)
				train_inputs, train_targets, train_seq_len, original, epoch_num, train_inputs_length, char_map_str = next_training_batch()
				feed = {inputs: train_inputs,
						targets: train_targets,
						seq_len: train_seq_len}

				if cell_name == "ATTN":
					feed[input_sequence_length] = train_inputs_length
					feed[char_ids] = char_map_str
					

				batch_cost, _ = session.run([cost, optimizer], feed)
				train_cost += batch_cost * len(original)
				train_ler += session.run(ler, feed_dict=feed) * len(original)

				# # Decoding
				# d = session.run(decoded[0], feed_dict=feed)
				# str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
				# # Replacing blank label to none
				# str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
				# # Replacing space label to space
				# str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

				num_examples += len(original)
				# log_print('Original: %s' % original)
				# log_print('Decoded: %s' % str_decoded)

				if overfit:
					break
			log_print ("Total Examples seen: %i" % num_examples)
				
			train_cost /= num_examples
			train_ler /= num_examples

			val_inputs, val_targets, val_seq_len, val_original, val_inputs_length, random_shift, char_map_str = next_validation_batch()
			val_feed = {inputs: val_inputs,
						targets: val_targets,
						seq_len: val_seq_len}
			if cell_name == "ATTN":
				val_feed[input_sequence_length] = val_inputs_length
				val_feed[char_ids] = char_map_str

			val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)

			
			# Decoding
			d = session.run(decoded[0], feed_dict=val_feed)
			
			if True:#not curr_epoch:
				val_original = ''.join([chr(x) for x in np.array(val_original[0]) + FIRST_INDEX])
				# Replacing blank label to none
				val_original = val_original.replace(chr(ord('z') + 1), '')
				# Replacing space label to space
				val_original = val_original.replace(chr(ord('a') - 1), ' ')

				log_print('Original val: %s : (%i)' % (val_original, len(val_original)))
			
			str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
			# Replacing blank label to none
			str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
			# Replacing space label to space
			str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
			str_decoded = str_decoded[:str_decoded.find('}')]+'}'
			log_print('Decoded val: %s : (%i)' % (str_decoded, len(str_decoded)))

			log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, " \
				  "val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"

			log_print(log.format(curr_epoch + 1, num_epochs, train_cost, train_ler,
							 val_cost, val_ler, time.time() - start))
			if val_ler < best_ler:
				best_ler = val_ler
				saver = tf.train.Saver()
				save_path = saver.save(session, "models/model_"+suffix+"_"+cell_name+"_best.ckpt")
				log_print("Better model found Model saved in path: %s" % save_path)
			
			# SAVED LAST EPOCH ANYWAY
			saver = tf.train.Saver()
			save_path = saver.save(session, "models/model_"+suffix+"_"+cell_name+"_last.ckpt")
			f_last = open("models/model_"+suffix+"_"+cell_name+"_last.ckpt.last", 'w')
			f_last.write(str(curr_epoch)+'\n')
			f_last.write(str(best_ler))
			f_last.close()
		writer.close()

if __name__ == '__main__':
        run_ctc()

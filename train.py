import time
import numpy as np
import tensorflow as tf
from preprocessing_yield import data_generator

import argparse

ap = argparse.ArgumentParser()
list_of_choices = ["LSTM", "GRU", "BILSTM", "BIGRU", "ATTN"]
ap.add_argument("-n", "--cell", required=True, help="name of the cell unit",  choices=list_of_choices)
#ap.add_argument("-h", "--num-hidden", required=True, help="number of hidden cell unit")
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
num_epochs = 200#00#args["num-epochs"]
num_hidden = 1#args["num-hidden"]
num_layers = 1#args["num-layers"]
batch_size = 1

train_data_gen = data_generator(text_dir='TEDLIUM_release1/test/stm', speech_dir='TEDLIUM_release1/test/sph', batch_size=batch_size, feature=feature, num_features=num_features)
valid_data_gen = data_generator(text_dir='TEDLIUM_release1/test/stm', speech_dir='TEDLIUM_release1/test/sph', batch_size=batch_size, feature=feature, num_features=num_features)

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
		#seq_len=tf.placeholder(tf.int32)
		#seq_len=1
		logits = model.Model(inputs, seq_len, num_classes=num_classes, num_hidden=num_hidden, num_layers=num_layers)
		
		loss = tf.nn.ctc_loss(targets, logits, seq_len)
		cost = tf.reduce_mean(loss)

		# optimizer = tf.train.AdamOptimizer().minimize(cost)
		# optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(cost)
		optimizer = tf.train.MomentumOptimizer(learning_rate=0.0008, momentum=0.9).minimize(cost)
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
		data_x, data_y, len_y, epoch_num = next(train_data_gen)
		target = sparse_tuple_from(data_y)
		return data_x, target, len_y, data_y, epoch_num

	def next_validation_batch():
		global valid_data_gen
		data_x, data_y, len_y, _ = next(valid_data_gen)
		target = sparse_tuple_from(data_y)
		return data_x, target, len_y, data_y, 0

	best_ler = 2.0
	with tf.Session(graph=graph) as session:
		tf.global_variables_initializer().run()
		writer = tf.summary.FileWriter("output", session.graph)
		# Add ops to save and restore all the variables.
		saver = tf.train.Saver()
		
		for curr_epoch in range(num_epochs):	
			train_cost = train_ler = 0
			start = time.time()
			num_examples = 0
			epoch_num = curr_epoch
			while(epoch_num<=curr_epoch):
				train_inputs, train_targets, train_seq_len, original, epoch_num = next_training_batch()
				feed = {inputs: train_inputs,
						targets: train_targets,
						seq_len: train_seq_len}

				
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

				num_examples += len(train_targets)
				# print('Original: %s' % original)
				# print('Decoded: %s' % str_decoded)
				
				# TO OVERFIT UNCOMMENT BELOW LINES
				# break

			train_cost /= num_examples
			train_ler /= num_examples

			val_inputs, val_targets, val_seq_len, val_original, random_shift = next_validation_batch()
			val_feed = {inputs: val_inputs,
						targets: val_targets,
						seq_len: val_seq_len}

			val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)

			
			# Decoding
			d = session.run(decoded[0], feed_dict=val_feed)
			
			if not curr_epoch:
				val_original = ''.join([chr(x) for x in np.array(val_original[0]) + FIRST_INDEX])
				# Replacing blank label to none
				val_original = val_original.replace(chr(ord('z') + 1), '')
				# Replacing space label to space
				val_original = val_original.replace(chr(ord('a') - 1), ' ')

				print('Original val: %s' % val_original)
			
			str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
			# Replacing blank label to none
			str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
			# Replacing space label to space
			str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

			print('Decoded val: %s' % str_decoded)

			log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, " \
				  "val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"

			if val_ler < best_ler:
				best_ler = val_ler
				save_path = saver.save(session, "models/model_"+cell_name+"_"+str(curr_epoch)+".ckpt")
				print("Better model found Model saved in path: %s" % save_path)

			print(log.format(curr_epoch + 1, num_epochs, train_cost, train_ler,
							 val_cost, val_ler, time.time() - start))
		writer.close()

if __name__ == '__main__':
        run_ctc()

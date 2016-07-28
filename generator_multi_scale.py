import scikits.audiolab
import theano
import theano.tensor as T
import numpy
import lasagne
import pickle

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from layers import *

from wav_utils import write_audio_file
from generic_utils import *

import sys 
sys.setrecursionlimit(50000)

def round_to(x, y):
    """round x up to the nearest y"""
    return int(numpy.ceil(x / float(y))) * y

n_files = 1000

GRAD_CLIP = 1.0
BITRATE = 16000
DATA_PATH = "/Tmp/kumarkun/blizzard_small/flac"
OUTPUT_DIR = "/Tmp/kumarkun/generated_blizz_single"
BATCH_SIZE = 1
INPUT_LEN = 16000
SEQ_LEN = INPUT_LEN
DIM = 5120

POOL = False

filter_sizes_all = [3, 5, 11, 21, 41, 81, 161, 321, 641, 1281, 2561, 8001, 15999]
expn = 0

for k in range(2):
	for j in range(len(filter_sizes_all)):
		if k == 0:
			filter_sizes = filter_sizes_all[::-1][:j+1]
		else:
			filter_sizes = filter_sizes_all[:j+1]

		DIM = (DIM//len(filter_sizes))*len(filter_sizes)

		expn += 1

		filters_name = "_".join([str(f) for f in filter_sizes])
		OTHER_INFO = "multiscale_fs_{}_dim_{}_exp_{}".format(filters_name, DIM, expn)
		params = []
		other_params = []

		input_sound_ = T.matrix('input_sound') # shape = (batch_size, input_length)

		output_sound_init = numpy.random.uniform(
		        low=-0.5,
		        high=0.5,
		        size=(BATCH_SIZE, INPUT_LEN)
		    ).astype(numpy.float32)

		output_sound_var = theano.shared(output_sound_init, name="output_sound")

		params.append(output_sound_var)

		output_sound = T.ones_like(input_sound_)*output_sound_var

		input_sound = input_sound_.dimshuffle(0,'x', 1, 'x') # shape = (batch_size, 1, input_length, 1)
		output_sound = output_sound.dimshuffle(0,'x', 1, 'x') # shape = (batch_size, 1, input_length, 1)

		# filter_sizes = [3, 5, 11, 21, 41, 81, 161, 321]
		# filter_sizes = [3, 5, 11, 21]



	# filter_sizes = [3, 5, 11, 21]

		conv_out_input = []
		conv_out_output = []

		for filter_size in filter_sizes:

			current_filter = get_conv_2d_filter(
							(DIM//len(filter_sizes), 1, filter_size,1), 
							param_list = other_params, 
							masktype = None, 
							name = "filter_{}".format(filter_size)
							)

			curr_conv_out_input = T.nnet.conv2d(input_sound, current_filter, border_mode = 'valid') # shape = (batch_size, 128, input_length, 1)
			curr_conv_out_output = T.nnet.conv2d(output_sound, current_filter, border_mode= 'valid') # shape = (batch_size, 128, input_length, 1)

			conv_out_input.append(curr_conv_out_input)
			conv_out_output.append(curr_conv_out_output)

		input_feature_map = T.concatenate(conv_out_input, axis= 1) # (batch_size, 1024, input_length, 1)
		output_feature_map = T.concatenate(conv_out_output, axis= 1) # (batch_size, 1024, input_length, 1)

		input_feature_map = T.nnet.relu(input_feature_map)
		output_feature_map = T.nnet.relu(output_feature_map)

		input_features_reshaped = curr_conv_out_input.dimshuffle(1,0,2,3)
		output_features_reshaped = curr_conv_out_output.dimshuffle(1,0,2,3)

		old_shape = input_features_reshaped.shape

		input_features_reshaped = input_features_reshaped.reshape((old_shape[0], -1))
		output_features_reshaped = output_features_reshaped.reshape((old_shape[0], -1))


		dotted_input_feature_map = T.dot(
				input_features_reshaped, 
				input_features_reshaped.dimshuffle(1,0)
			)
		dotted_output_feature_map = T.dot(
				output_features_reshaped, 
				output_features_reshaped.dimshuffle(1,0)
			)
		# dotted_input_feature_map = dotted_input_feature_map.reshape(
		# 	(old_shape[0], old_shape[1], -1)
		# 	)
		# dotted_output_feature_map = dotted_output_feature_map.reshape(
		# 	(old_shape[0], old_shape[1], -1)
		# 	)

		# if MODE == "TIME_INVARIANT":
		# input_feature_map = T.sum(dotted_input_feature_map, axis = 1)
		# output_feature_map = T.sum(dotted_output_feature_map, axis = 1)
		# else:
		input_feature_map = dotted_input_feature_map
		output_feature_map = dotted_output_feature_map

		input_feature_map = dotted_input_feature_map
		input_feature_norm = T.sqrt(T.sum(input_feature_map*input_feature_map))

		input_feature_map = input_feature_map/ input_feature_norm
		output_feature_map = dotted_output_feature_map/input_feature_norm

		cost = T.sum((input_feature_map - output_feature_map)*(input_feature_map - output_feature_map))

		grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
		grads = [T.clip(g, floatX(-GRAD_CLIP), floatX(GRAD_CLIP)) for g in grads]

		lr = T.scalar('lr')

		updates = lasagne.updates.adam(grads, params, learning_rate=lr)

		updates = lasagne.updates.apply_nesterov_momentum(updates, params=params, momentum=0.80)

		train_fn = theano.function([input_sound_, lr], cost, updates = updates)


		paths = [DATA_PATH+'/p{}.flac'.format(i) for i in xrange(n_files)]

		batch_paths = paths[100:100+BATCH_SIZE]

		# batch_seq_len = len(scikits.audiolab.flacread(batch_paths[0])[0])


		batch = numpy.zeros(
		    (BATCH_SIZE, INPUT_LEN),
		    dtype='float32'
		)

		for i, path in enumerate(batch_paths):
			data, fs, enc = scikits.audiolab.flacread(path)
			data = numpy.float32(data)
			data = data[:INPUT_LEN]
			assert(data.max() - data.min() > 0)
			batch[i, :len(data)] = (data - data.min())/(data.max() - data.min()) - 0.5


		create_folder_if_not_there(OUTPUT_DIR)

		costs = []

		lr_val = numpy.float32(0.001)
		prev_cost_mean = 12345678.
		for i in range(4000):
			cost = train_fn(batch, lr_val)
			costs.append(cost)
			if (i + 1) % 10 == 0:
				curr_cost_mean = numpy.mean(costs[-40:])
				if curr_cost_mean > prev_cost_mean:
					lr_val = numpy.float32(lr_val/2.)
					print "Slashing learning_rate by a factor of 2. New rate is {}".format(lr_val)
				prev_cost_mean = curr_cost_mean

			print " Exp no. {}, iteration {}, cost {}".format(expn, i, cost)
			if (i+1) % 250 == 0:
				output = output_sound_var.get_value()
				print "Saving audio...."
				for j in range(len(output)):
					write_audio_file("{}/iter_{}_{}_{}_{}".format(OUTPUT_DIR, i+1, OTHER_INFO,  j, cost),  BITRATE, output[j])

		pickle.dump(costs, open('{}/costs_{}.pkl'.format(OUTPUT_DIR, OTHER_INFO), 'wb'))







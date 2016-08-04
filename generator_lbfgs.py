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

import scipy.optimize

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

num_layers_filter_sizes = [
				(2, 11),
				(2, 51),
				(2, 501),
				(2, 999),
				(2, 3999),
				(2, 7999),
				(3, 11),
				(3, 99),
				(3, 333),
				(3, 1991),
				(4, 11),
				(4, 251),
				(4, 501),
				(4, 999),
				(4, 999),
				(5, 11),
				(5, 51),
				(5, 99),
				(5, 399),
				(5, 999),
				(6, 5),
				(6, 21),
				(6, 41),
				(7, 5),
				(7, 21),
				(7, 41),
				(8, 3),
				(8, 5),
				(8, 11),
				(8, 15),
				(8, 31),
				(9, 3),
				(9, 7),
				(6, 81),
				(7, 81),
				(8, 81),
				(6, 239)
				]
paths = [DATA_PATH+'/p{}.flac'.format(i) for i in xrange(n_files)]

batch_paths = paths[100:100+BATCH_SIZE]

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



# num_layers_filter_sizes = [
# 					(1, 24001),
# 					(1, 63999),
# 					(1, 48001),
# 					(2, 16001),
# 					(3, 8001),
# 					(4, 8001),
# 					(5, 2001)
# 				]

# num_layers_filter_sizes = [
# 					(1, 7999),
# 					(1, 7997),
# 					(1, 7991),
# 					(1, 7951),
# 					(1, 7001),
# 					(1, 5001),
# 					(1, 3001),
# 					(1, 2001),
# 					(1, 501),
# 					(1)
# 				]

expn = 0
for POOL in (False, True):
	for num_layer, filter_size in num_layers_filter_sizes:

		expn += 1

		OTHER_INFO = "nl_{}_fs_{}_dim_{}_pool{}_exp_{}".format(num_layer, filter_size, DIM, POOL, expn)
		params = []
		other_params = []

		output_sound_init = numpy.array(
			numpy.random.uniform(
		        low=-0.5,
		        high=0.5,
		        size=(INPUT_LEN,)
		    ),
		    order='F',
		    copy=False
		)

		input_sound_var = theano.shared(batch[0], name="input_sound")
		output_sound_ = T.vector('output_sound_')

		input_sound_ = T.ones_like(output_sound_)*input_sound_var

		input_sound = input_sound_.dimshuffle('x','x', 0, 'x') # shape = (batch_size, 1, input_length, 1)
		output_sound = output_sound_.dimshuffle('x','x', 0, 'x') # shape = (batch_size, 1, input_length, 1)

		# filter_sizes = [3, 5, 11, 21, 41, 81, 161, 321]
		# filter_sizes = [3, 5, 11, 21]


		conv_out_input = []
		conv_out_output = []

		curr_conv_out_input = input_sound
		curr_conv_out_output = output_sound

		for i in range(num_layer):
			print "{}".format(1 if i == 0 else DIM)

			current_filter = get_conv_2d_filter(
							(DIM, 1 if i == 0 else DIM, filter_size,1),
							param_list = other_params,
							masktype = None,
							name = "filter_{}".format(filter_size)
							)

			curr_conv_out_input = T.nnet.relu(
							T.nnet.conv2d(
								curr_conv_out_input,
								current_filter,
								border_mode='valid'
							)
						)

			curr_conv_out_output = T.signal.pool.pool_2d(
					T.nnet.relu(
							T.nnet.conv2d(
								curr_conv_out_output,
								current_filter,
								border_mode='valid'
							)
					),
					(2,1),
					ignore_border=True,
					mode = 'average_exc_pad'
				)

			if POOL == True:
				curr_conv_out_input = T.signal.pool.pool_2d(
					curr_conv_out_input,
					(2,1),
					ignore_border=True,
					mode = 'average_exc_pad'
				)

				curr_conv_out_output = T.signal.pool.pool_2d(
					curr_conv_out_output,
					(2,1),
					ignore_border=True,
					mode = 'average_exc_pad'
				)


			# conv_out_input.append(curr_conv_out_input)
			# conv_out_output.append(curr_conv_out_output)

		# input_feature_map = T.concatenate(conv_out_input, axis= 1) # (batch_size, 1024, input_length, 1)
		# output_feature_map = T.concatenate(conv_out_output, axis= 1) # (batch_size, 1024, input_length, 1)

		# input_feature_map = T.nnet.relu(input_feature_map)
		# output_feature_map = T.nnet.relu(output_feature_map)

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
		input_feature_norm = T.sqrt(T.sum(input_feature_map*input_feature_map))

		input_feature_map = input_feature_map/ input_feature_norm
		output_feature_map = dotted_output_feature_map/input_feature_norm

		cost = floatX(1000000)*T.sum((input_feature_map - output_feature_map)*(input_feature_map - output_feature_map))

		grads = T.grad(cost, wrt=output_sound)
		# grads = [T.clip(g, floatX(-GRAD_CLIP), floatX(GRAD_CLIP)) for g in grads]

		# lr = T.scalar('lr')

		# updates = lasagne.updates.adam(grads, params, learning_rate=lr)

		# updates = lasagne.updates.apply_nesterov_momentum(updates, params=params, momentum=0.80)

		func = theano.function([output_sound_], [cost, grads], allow_input_downcast=True)

		def func_x(x):
			c, g = func(x)
			return c, numpy.array(g, dtype='float', order='F', copy=False)

		# batch_seq_len = len(scikits.audiolab.flacread(batch_paths[0])[0])

		create_folder_if_not_there(OUTPUT_DIR)

		costs = []
		print "Training"

		out_x, f, d = scipy.optimize.fmin_l_bfgs_b(
			func_x,
			output_sound_init,
			bounds = [(-0.5, 0.5)]*INPUT_LEN,
			factr= 1e7,
			iprint=10,
			maxiter=2000
		)

		write_audio_file("{}/lbfgs_{}_{}".format(OUTPUT_DIR, OTHER_INFO, f),  BITRATE, out_x)

		pickle.dump(costs, open('{}/costs_{}.pkl'.format(OUTPUT_DIR, OTHER_INFO), 'wb'))







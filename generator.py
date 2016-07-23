import scikits.audiolab
import theano
import theano.tensor as T
import numpy
import lasagne

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from layers import *

from wav_utils import write_audio_file
from models import *


def round_to(x, y):
    """round x up to the nearest y"""
    return int(numpy.ceil(x / float(y))) * y

n_files = 1000

GRAD_CLIP = 1.0
BITRATE = 16000
DATA_PATH = "/Tmp/kumarkun/blizzard_small/flac"
BATCH_SIZE = 2
INPUT_LEN = 128000
SEQ_LEN = INPUT_LEN
MODE = "TIME_INVARIANT"
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
filter_sizes = [3, 5, 11, 21]

conv_out_input = []
conv_out_output = []

for filter_size in filter_sizes:

	current_filter = get_conv_2d_filter(
					(8, 1, filter_size,1), 
					param_list = other_params, 
					masktype = None, 
					name = "filter_{}".format(filter_size)
					)

	curr_conv_out_input = T.nnet.conv2d(input_sound, current_filter, border_mode=(filter_size//2,0)) # shape = (batch_size, 128, input_length, 1)
	curr_conv_out_output = T.nnet.conv2d(output_sound, current_filter, border_mode=(filter_size//2,0)) # shape = (batch_size, 128, input_length, 1)

	conv_out_input.append(curr_conv_out_input)
	conv_out_output.append(curr_conv_out_output)

input_feature_map = T.concatenate(conv_out_input, axis= 1) # (batch_size, 1024, input_length, 1)
output_feature_map = T.concatenate(conv_out_output, axis= 1) # (batch_size, 1024, input_length, 1)

input_features_reshaped = input_feature_map.dimshuffle(0,2,1,3)
output_features_reshaped = output_feature_map.dimshuffle(0,2,1,3)

dotted_input_feature_map = T.batched_dot(
		input_features_reshaped.reshape(
			(input_features_reshaped.shape[0]*input_features_reshaped.shape[1],
			input_features_reshaped.shape[2],
			1)
		), 
		input_features_reshaped.reshape(
			(input_features_reshaped.shape[0]*input_features_reshaped.shape[1],
			1,
			input_features_reshaped.shape[2])
		)
	)
dotted_output_feature_map = T.batched_dot(
		output_features_reshaped.reshape(
			(output_features_reshaped.shape[0]*output_features_reshaped.shape[1],
			output_features_reshaped.shape[2],
			1)
		), 
		output_features_reshaped.reshape(
			(output_features_reshaped.shape[0]*output_features_reshaped.shape[1],
			1,
			output_features_reshaped.shape[2])
		)
	)
dotted_input_feature_map = dotted_input_feature_map.reshape((BATCH_SIZE, INPUT_LEN, -1))
dotted_output_feature_map = dotted_output_feature_map.reshape((BATCH_SIZE, INPUT_LEN, -1))

if MODE == "TIME_INVARIANT":
	input_feature_map = T.sum(dotted_input_feature_map, axis = 1)
	output_feature_map = T.sum(dotted_output_feature_map, axis = 1)
else:
	input_feature_map = dotted_input_feature_map
	output_feature_map = dotted_output_feature_map

cost = T.mean((input_feature_map - output_feature_map)*(input_feature_map - output_feature_map))

grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
grads = [T.clip(g, floatX(-GRAD_CLIP), floatX(GRAD_CLIP)) for g in grads]

updates = lasagne.updates.adam(grads, params)

train_fn = theano.function([input_sound_], cost, updates = updates)


paths = [DATA_PATH+'/p{}.flac'.format(i) for i in xrange(n_files)]

batch_paths = paths[100:100+BATCH_SIZE]

batch_seq_len = len(scikits.audiolab.flacread(batch_paths[0])[0])

print(batch_seq_len)

batch = numpy.zeros(
    (BATCH_SIZE, batch_seq_len),
    dtype='float32'
)

for i, path in enumerate(batch_paths):
    data, fs, enc = scikits.audiolab.flacread(path)
    data = numpy.float32(data)
    assert(data.max() - data.min() > 0)
    batch[i, :len(data)] = (data - data.min())/(data.max() - data.min()) - 0.5

print batch.min()
print batch.max()

# for i in range(BATCH_SIZE):
# 	write_audio_file( "sample{}".format(i), BITRATE, batch[i])

for i in range(10000):
	cost = train_fn(batch)
	print cost
	output = output_sound_var.get_value()
	if i+1 % 500:
		for j in range(len(output)):
			write_audio_file("output_{}_{}".format(j, cost),  BITRATE, output[j])







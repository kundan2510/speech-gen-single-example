"""
RNN Speech Generation Model
Originally written by Ishaan Gulrajani ()
Modified by Kundan Kumar
"""

import numpy
import scikits.audiolab

import random
import time

import itertools
def quantize_data(data_array, levels):
    new_data = data_array.flatten().astype('float32')
    new_data = (new_data - new_data.min()) + 0.5
    assert(new_data.min() >= 0),"Some problem with min operation"
    new_data /= (new_data.max())
    assert(new_data.max() <= 1.0),"Some problem with max operation"
    new_data *= (levels*0.999)
    new_data =  numpy.floor(new_data).astype('int32')
    assert (min(new_data) >= 0 and max(new_data) < levels), "some problem: max = %d, min = %d, level = %d "%(max(new_data),min(new_data),levels)
    return new_data


def feed_data(data_path, n_files, BATCH_SIZE, SEQ_LEN, OVERLAP, Q_LEVELS, Q_ZERO):
    """
    Generator that yields training inputs (subbatch, reset). `subbatch` contains
    quantized audio data; `reset` is a boolean indicating the start of a new
    sequence (i.e. you should reset h0 whenever `reset` is True).

    Feeds subsequences which overlap by a specified amount, so that the model
    can always have target for every input in a given subsequence.

    Loads sequentially-named FLAC files in a directory
    (p0.flac, p1.flac, p2.flac, ..., p[n_files-1].flac)

    Assumes all flac files have the same length.

    data_path: directory containing the flac files
    n_files: how many FLAC files are in the directory
    (see two_tier.py for a description of the constants)

    returns: (subbatch, reset)
    subbatch.shape: (BATCH_SIZE, SEQ_LEN + OVERLAP)
    reset: True or False
    """

    def round_to(x, y):
        """round x up to the nearest y"""
        return int(numpy.ceil(x / float(y))) * y

    paths = [data_path+'/p{}.flac'.format(i) for i in xrange(n_files)]

    for epoch in itertools.count():

        random.seed(123)
        random.shuffle(paths)

        batches = []
        for i in xrange(len(paths) / BATCH_SIZE):
            batches.append(paths[i*BATCH_SIZE:(i+1)*BATCH_SIZE])

        random.shuffle(batches)

        for batch_paths in batches:
            # batch_seq_len = length of longest sequence in the batch, rounded up to
            # the nearest SEQ_LEN.
            batch_seq_len = len(scikits.audiolab.flacread(batch_paths[0])[0])
            batch_seq_len = round_to(batch_seq_len, SEQ_LEN)

            batch = numpy.zeros(
                (BATCH_SIZE, batch_seq_len),
                dtype='int32'
            )

            for i, path in enumerate(batch_paths):
                data, fs, enc = scikits.audiolab.flacread(path)
                batch[i, :len(data)] = quantize_data(data, Q_LEVELS)


            batch = numpy.concatenate([
                numpy.full((BATCH_SIZE, OVERLAP), Q_ZERO, dtype='int32'),
                batch
            ], axis=1)

            for i in xrange(batch.shape[1] / SEQ_LEN):
                reset = numpy.int32(i==0)
                subbatch = batch[:, i*SEQ_LEN : (i+1)*SEQ_LEN + OVERLAP]
                yield (subbatch, reset, epoch)

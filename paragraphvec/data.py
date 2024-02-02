import torchdata.datapipes as dp
import multiprocessing
import os
import re
import signal
from math import ceil
from os.path import join

import numpy as np
import torch
from numpy.random import choice
# from torchtext.data import Field, TabularDataset


from torchtext.vocab import build_vocab_from_iterator
from collections import Counter


def _tokenize_str(str_):
    # keep only alphanumeric and punctations
    str_ = re.sub(r'[^A-Za-z0-9(),.!?\'`]', ' ', str_)
    # remove multiple whitespace characters
    str_ = re.sub(r'\s{2,}', ' ', str_)
    # punctations to tokens
    str_ = re.sub(r'\(', ' ( ', str_)
    str_ = re.sub(r'\)', ' ) ', str_)
    str_ = re.sub(r',', ' , ', str_)
    str_ = re.sub(r'\.', ' . ', str_)
    str_ = re.sub(r'!', ' ! ', str_)
    str_ = re.sub(r'\?', ' ? ', str_)
    # split contractions into multiple tokens
    str_ = re.sub(r'\'s', ' \'s', str_)
    str_ = re.sub(r'\'ve', ' \'ve', str_)
    str_ = re.sub(r'n\'t', ' n\'t', str_)
    str_ = re.sub(r'\'re', ' \'re', str_)
    str_ = re.sub(r'\'d', ' \'d', str_)
    str_ = re.sub(r'\'ll', ' \'ll', str_)
    # lower case
    return str_.strip().lower().split()


def getTokens(data_iter):
    """
    Function to yield tokens from an iterator. Since, our iterator contains
    tuple of sentences (source and target), `place` parameters defines for which
    index to return the tokens for. `place=0` for source and `place=1` for target
    """
    for sentences in data_iter:
        yield _tokenize_str(sentences[0])


class datasetClass():
    def __init__(self, data_pipe):
        self.data_pipe = data_pipe

        self.lines = []
        self.vocab = None
        self.length = 0
        self.counter = Counter()

    def load_dataset(self):
        """Loads contents from a file in the *data* directory into a
        torchtext.data.TabularDataset instance.
        """

        source_vocab = build_vocab_from_iterator(
            getTokens(self.data_pipe),
            min_freq=2,
            specials=['<pad>', '<sos>', '<eos>', '<unk>'],
            special_first=True
        )
        source_vocab.set_default_index(source_vocab['<unk>'])

        self.vocab = source_vocab
        # print(self.vocab)
        # print(f"type of self.vocab: {type(self.vocab)}")

        # Expand the list self.lines with the contents of the file tokenized

        for line in self.data_pipe:
            words = _tokenize_str(line[0])
            self.lines.append(_tokenize_str(line[0]))
            self.counter.update(words)

        self.length = len(self.lines)


class NCEData(object):
    """An infinite, parallel (multiprocess) batch generator for
    noise-contrastive estimation of word vector models.

    Parameters
    ----------
    dataset: torchtext.data.TabularDataset
        Dataset from which examples are generated. A column labeled *text*
        is expected and should be comprised of a list of tokens. Each row
        should represent a single document.

    batch_size: int
        Number of examples per single gradient update.

    context_size: int
        Half the size of a neighbourhood of target words (i.e. how many
        words left and right are regarded as context).

    num_noise_words: int
        Number of noise words to sample from the noise distribution.

    max_size: int
        Maximum number of pre-generated batches.

    num_workers: int
        Number of jobs to run in parallel. If value is set to -1, total number
        of machine CPUs is used.
    """
    # code inspired by parallel generators in https://github.com/fchollet/keras

    def __init__(self, dataset, batch_size, context_size,
                 num_noise_words, max_size, num_workers):
        self.max_size = max_size

        # print(" dataset type is {}".format(type(dataset)))
        # print(" batch_size is {}".format(batch_size))
        # print(" context_size is {}".format(context_size))
        # print(" num_noise_words is {}".format(num_noise_words))
        # print(" max_size is {}".format(max_size))
        # print(" num_workers is {}".format(num_workers))

        self.num_workers = num_workers if num_workers != -1 else os.cpu_count()
        if self.num_workers is None:
            self.num_workers = 1

        # print(f"num of workers: {self.num_workers}")
        # print(f"os.cpu_count() {os.cpu_count()}")

        self._generator = _NCEGenerator(
            dataset,
            batch_size,
            context_size,
            num_noise_words,
            _NCEGeneratorState(context_size))

        self._queue = None
        self._stop_event = None
        self._processes = []

    def __len__(self):
        return len(self._generator)

    def vocabulary_size(self):
        return self._generator.vocabulary_size()

    def start(self):
        """Starts num_worker processes that generate batches of data."""
        self._queue = multiprocessing.Queue(maxsize=self.max_size)
        self._stop_event = multiprocessing.Event()

        for _ in range(self.num_workers):
            process = multiprocessing.Process(target=self._parallel_task)
            process.daemon = True
            self._processes.append(process)
            process.start()

    def _parallel_task(self):
        while not self._stop_event.is_set():
            try:
                batch = self._generator.next()
                # queue blocks a call to put() until a free slot is available
                self._queue.put(batch)
            except KeyboardInterrupt:
                self._stop_event.set()

    def get_generator(self):
        """Returns a generator that yields batches of data."""
        while self._is_running():
            yield self._queue.get()

    def stop(self):
        """Terminates all processes that were created with start()."""
        if self._is_running():
            self._stop_event.set()

        for process in self._processes:
            if process.is_alive():
                os.kill(process.pid, signal.SIGINT)
                process.join()

        if self._queue is not None:
            self._queue.close()

        self._queue = None
        self._stop_event = None
        self._processes = []

    def _is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()


class _NCEGenerator(object):
    """An infinite, process-safe batch generator for noise-contrastive
    estimation of word vector models.

    Parameters
    ----------
    state: paragraphvec.data._NCEGeneratorState
        Initial (indexing) state of the generator.

    For other parameters see the NCEData class.
    """

    def __init__(self, dataset, batch_size, context_size,
                 num_noise_words, state):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_size = context_size
        self.num_noise_words = num_noise_words

        self._vocabulary = self.dataset.vocab
        self._sample_noise = None
        self._init_noise_distribution()
        self._state = state

    def _init_noise_distribution(self):
        # we use a unigram distribution raised to the 3/4rd power,
        # as proposed by T. Mikolov et al. in Distributed Representations
        # of Words and Phrases and their Compositionality
        probs = np.zeros(len(self._vocabulary) - 1)

        for word, freq in self.dataset.counter.items():
            probs[self._word_to_index(word)] = freq

        # print("type of probs is {}".format(type(probs)))
        # print("probs is {}".format(probs))
        # print(f"self.dataset.counter: {self.dataset.counter}")

        probs = np.power(probs, 0.75)
        probs /= np.sum(probs)

        # print("type of probs is {}".format(type(probs)))
        # print("probs is {}".format(probs))

        self._sample_noise = lambda: choice(
            probs.shape[0], self.num_noise_words, p=probs).tolist()

        # print("type of self._sample_noise is {}".format(type(self._sample_noise)))
        # print("type of self.num_noise_words is {}".format(type(self.num_noise_words)))
        # print("type of self._vocabulary is {}".format(type(self._vocabulary)))
        # print("type of self._vocabulary.freqs is {}".format(type(self._vocabulary.freqs)))
        # print("type of self._vocabulary.freqs.items() is {}".format(type(self._vocabulary.freqs.items())))

    def __len__(self):
        num_examples = sum(self._num_examples_in_doc(d) for d in self.dataset.lines)
        return ceil(num_examples / self.batch_size)

    def vocabulary_size(self):
        return len(self._vocabulary) - 1

    def next(self):
        """Updates state for the next process in a process-safe manner
        and generates the current batch."""
        prev_doc_id, prev_in_doc_pos = self._state.update_state(
            self.dataset,
            self.batch_size,
            self.context_size,
            self._num_examples_in_doc)

        # generate the actual batch
        batch = _NCEBatch(self.context_size)

        while len(batch) < self.batch_size:
            if prev_doc_id == self.dataset.length:
                # last document exhausted
                batch.torch_()
                return batch
            if prev_in_doc_pos <= (len(self.dataset.lines[prev_doc_id]) - 1
                                   - self.context_size):
                # more examples in the current document
                self._add_example_to_batch(prev_doc_id, prev_in_doc_pos, batch)
                prev_in_doc_pos += 1
            else:
                # go to the next document
                prev_doc_id += 1
                prev_in_doc_pos = self.context_size

        batch.torch_()
        return batch

    def _num_examples_in_doc(self, doc, in_doc_pos=None):

        if in_doc_pos is not None:
            # number of remaining
            if len(doc) - in_doc_pos >= self.context_size + 1:
                return len(doc) - in_doc_pos - self.context_size
            return 0

        if len(doc) >= 2 * self.context_size + 1:
            # total number
            return len(doc) - 2 * self.context_size
        return 0

    def _add_example_to_batch(self, doc_id, in_doc_pos, batch):
        doc = self.dataset.lines[doc_id]
        batch.doc_ids.append(doc_id)

        # sample from the noise distribution
        current_noise = self._sample_noise()
        current_noise.insert(0, self._word_to_index(doc[in_doc_pos]))
        batch.target_noise_ids.append(current_noise)

        if self.context_size == 0:
            return

        current_context = []
        context_indices = (in_doc_pos + diff for diff in
                           range(-self.context_size, self.context_size + 1)
                           if diff != 0)

        for i in context_indices:
            context_id = self._word_to_index(doc[i])
            current_context.append(context_id)
        batch.context_ids.append(current_context)

    def _word_to_index(self, word):
        return self._vocabulary[word] - 1


class _NCEGeneratorState(object):
    """Batch generator state that is represented with a document id and
    in-document position. It abstracts a process-safe indexing mechanism."""

    def __init__(self, context_size):
        # use raw values because both indices have
        # to manually be locked together
        self._doc_id = multiprocessing.RawValue('i', 0)
        self._in_doc_pos = multiprocessing.RawValue('i', context_size)
        self._lock = multiprocessing.Lock()

    def update_state(self, dataset, batch_size,
                     context_size, num_examples_in_doc):
        """Returns current indices and computes new indices for the
        next process."""
        # Print type and value of every argument
        # print("type of dataset is {}".format(type(dataset)))
        # print("type of batch_size is {}".format(type(batch_size)))
        # print("type of context_size is {}".format(type(context_size)))
        # print("type of num_examples_in_doc is {}".format(type(num_examples_in_doc)))
        # print("dataset is {}".format(dataset))
        # print("batch_size is {}".format(batch_size))
        # print("context_size is {}".format(context_size))
        # print("num_examples_in_doc is {}".format(num_examples_in_doc))
        with self._lock:
            doc_id = self._doc_id.value
            in_doc_pos = self._in_doc_pos.value
            self._advance_indices(
                dataset, batch_size, context_size, num_examples_in_doc)
            return doc_id, in_doc_pos

    def _advance_indices(self, dataset, batch_size,
                         context_size, num_examples_in_doc):
        num_examples = num_examples_in_doc(
            dataset.lines[self._doc_id.value], self._in_doc_pos.value)

        if num_examples > batch_size:
            # more examples in the current document
            self._in_doc_pos.value += batch_size
            return

        if num_examples == batch_size:
            # just enough examples in the current document
            if self._doc_id.value < dataset.length - 1:
                self._doc_id.value += 1
            else:
                self._doc_id.value = 0
            self._in_doc_pos.value = context_size
            return

        while num_examples < batch_size:
            if self._doc_id.value == dataset.length - 1:
                # last document: reset indices
                self._doc_id.value = 0
                self._in_doc_pos.value = context_size
                return

            self._doc_id.value += 1
            num_examples += num_examples_in_doc(
                dataset.lines[self._doc_id.value])

        self._in_doc_pos.value = (len(dataset.lines[self._doc_id.value])
                                  - context_size
                                  - (num_examples - batch_size))


class _NCEBatch(object):
    def __init__(self, context_size):
        self.context_ids = [] if context_size > 0 else None
        self.doc_ids = []
        self.target_noise_ids = []

    def __len__(self):
        return len(self.doc_ids)

    def torch_(self):
        if self.context_ids is not None:
            self.context_ids = torch.LongTensor(self.context_ids)
        self.doc_ids = torch.LongTensor(self.doc_ids)
        self.target_noise_ids = torch.LongTensor(self.target_noise_ids)

    def cuda_(self):
        if self.context_ids is not None:
            self.context_ids = self.context_ids.cuda()
        self.doc_ids = self.doc_ids.cuda()
        self.target_noise_ids = self.target_noise_ids.cuda()

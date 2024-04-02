
import os
import pickle
import time
from sys import float_info, stdout
import gzip
import fire
import torch
import datetime
from torch.optim import Adam
import sys
import csv
from collections import Counter
from torchtext.vocab import Vocab
import csv
import threading
import time
from collections import OrderedDict
sys.path.append(".")
import torchdata.datapipes as dp  # noqa: E402
from paragraphvec.utils import save_training_state  # noqa: E402
from paragraphvec.models import DM, DBOW  # noqa: E402
from paragraphvec.loss import NegativeSampling  # noqa: E402
from paragraphvec.data import NCEData  # noqa: E402
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            value = self.cache.pop(key)
            self.cache[key] = value  # Mark as recently used
            return value

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)  # Pop first item (least recently used)
            self.cache[key] = value  # Insert as most recently used

    def __len__(self):
        return len(self.cache)


class GzFileLoader:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.line_cache = LRUCache(15000)
        self.cache_lock = threading.Lock()
        self.caching_thread = threading.Thread(target=self.background_cache_lines, daemon=True)
        # self.caching_thread.start()
        self.average_line_length = 0
        self.number_of_lines = 0
        self.max_line_length = 0
        self.length = 0
        self.file_line_counts = []

    def __iter__(self):
        for file_idx, file_path in enumerate(self.file_paths):
            try:
                with gzip.open(file_path, 'rt') as f:
                    number_of_lines_in_file = 0
                    # print(f"opened file {file_path}")
                    for line in f:
                        line = line.rstrip()[47:]
                        self.average_line_length = ((self.average_line_length * self.number_of_lines) +
                                                    len(line)) / (self.number_of_lines + 1)
                        if len(line) > self.max_line_length:
                            self.max_line_length = len(line)

                        with self.cache_lock:
                            if len(self.line_cache) < self.line_cache.capacity:
                                self.line_cache.put(self.length, line)

                        self.number_of_lines += 1
                        number_of_lines_in_file += 1
                        yield line

                    self.file_line_counts.append(number_of_lines_in_file)
                    self.length += number_of_lines_in_file

            except Exception as e:
                print(f"Exception: {e}, File Path: {file_path}")
                continue

    def background_cache_lines(self):
        while True:
            with self.cache_lock:
                cache_len = len(self.line_cache)
            if cache_len < self.line_cache.capacity:
                # Preload lines in the background if cache is not full
                self.__call__(cache_len)
            time.sleep(0.01)  # Slow down the thread to avoid performance issues

    def __call__(self, doc_idx):
        # Check if line is in cache
        cached_line = self.line_cache.get(doc_idx)
        if cached_line is not None:
            # print(f"got cached line: {doc_idx}")
            return cached_line

        # print(f"cache miss for doc_idx: {doc_idx}")

        cumulated_lines = 0
        # print(f"getting doc_idx: {doc_idx}")
        for file_idx, lines in enumerate(self.file_line_counts):
            if cumulated_lines + lines > doc_idx:
                # The document is in this file
                try:
                    with gzip.open(self.file_paths[file_idx], 'rt') as f:
                        for current_line, line in enumerate(f):
                            if cumulated_lines + current_line == doc_idx:
                                self.line_cache.put(doc_idx, line)
                                return line.rstrip()[47::]
                except Exception as e:
                    print(f"Exception: {e}, File Path: {self.file_paths[file_idx]}")
                    return None
            cumulated_lines += lines

        # If we haven't returned yet, then the doc_idx is too large
        return None


def start(datapipe_file_name,
          vocab_and_counter_object_file_name,
          num_noise_words,
          vec_dim,
          num_epochs,
          batch_size,
          lr,
          model_ver='dbow',
          context_size=0,
          vec_combine_method='sum',
          save_all=False,
          generate_plot=True,
          max_generated_batches=5,
          num_workers=1,
          cache_objects=False):
    """Trains a new model. The latest checkpoint and the best performing
    model are saved in the *models* directory.

    Parameters
    ----------
    datapipe_file_name: str
        file of the data_pipe object should have a __iter__ yielding sentences 

    vocab_and_counter_object_file_name: Vocab
        Vocab object of type torchtext.vocab.Vocab
        Counter object of type collections.Counter        

    model_ver: str, one of ('dm', 'dbow'), default='dbow'
        Version of the model as proposed by Q. V. Le et al., Distributed
        Representations of Sentences and Documents. 'dbow' stands for
        Distributed Bag Of Words, 'dm' stands for Distributed Memory.

    vec_combine_method: str, one of ('sum', 'concat'), default='sum'
        Method for combining paragraph and word vectors when model_ver='dm'.
        Currently only the 'sum' operation is implemented.

    context_size: int, default=0
        Half the size of a neighbourhood of target words when model_ver='dm'
        (i.e. how many words left and right are regarded as context). When
        model_ver='dm' context_size has to greater than 0, when
        model_ver='dbow' context_size has to be 0.

    num_noise_words: int
        Number of noise words to sample from the noise distribution.

    vec_dim: int
        Dimensionality of vectors to be learned (for paragraphs and words).

    num_epochs: int
        Number of iterations to train the model (i.e. number
        of times every example is seen during training).

    batch_size: int
        Number of examples per single gradient update.

    lr: float
        Learning rate of the Adam optimizer.

    save_all: bool, default=False
        Indicates whether a checkpoint is saved after each epoch.
        If false, only the best performing model is saved.

    generate_plot: bool, default=True
        Indicates whether a diagnostic plot displaying loss value over
        epochs is generated after each epoch.

    max_generated_batches: int, default=5
        Maximum number of pre-generated batches.

    num_workers: int, default=1
        Number of batch generator jobs to run in parallel. If value is set
        to -1 number of machine cores are used.

    cache_objects: bool, default=False
    """

    #########################################################
    # TODO: Make this interface better
    # Load a pickle file for vocab and counter
    with open(f"{vocab_and_counter_object_file_name}.pkl", 'rb') as f:
        vocab_object, counter_object = pickle.load(f)

    # Load a pickle file for data_pipe
    print(f"datapipe_file_name: {datapipe_file_name}")
    # ...

    # Load a CSV file for data_pipe
    files_list = []
    with open(datapipe_file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        csv_reader.__next__()  # Skip the header
        for row in csv_reader:
            files_list.append(row[0])

    datapipe_object = GzFileLoader(files_list)
    #########################################################

    # Dataset checks
    if vocab_object is None:
        raise ValueError("Vocab object is required")
    if not isinstance(vocab_object, Vocab):
        raise ValueError("Invalid type of vocab object")

    if counter_object is None:
        raise ValueError("Counter object is required")
    if not isinstance(counter_object, Counter):
        raise ValueError("Invalid type of counter object")

    if datapipe_object is None:
        raise ValueError("Data pipe object is required")
    if not hasattr(datapipe_object, "__iter__"):
        raise ValueError("Data pipe object must have an __iter__ method")

    # Model checks
    if model_ver not in ('dm', 'dbow'):
        raise ValueError("Invalid version of the model")

    # Create a name for the cache file
    cache_file_name = f"dataset_{vocab_and_counter_object_file_name}_.pickle"

    model_ver_is_dbow = (model_ver == 'dbow')

    if model_ver_is_dbow and context_size != 0:
        raise ValueError("Context size has to be zero when using dbow")
    if not model_ver_is_dbow:
        if vec_combine_method not in ('sum', 'concat'):
            raise ValueError("Invalid method for combining paragraph and word "
                             "vectors when using dm")
        if context_size <= 0:
            raise ValueError("Context size must be positive when using dm")

    nce_data = NCEData(
        vocab_object,
        counter_object,
        datapipe_object,
        batch_size,
        context_size,
        num_noise_words,
        max_generated_batches,
        num_workers)
    nce_data.start()

    # Save dataset to pickle with date and parameters in the name
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    params_str = f"model_ver={model_ver}_vec_dim={vec_dim}_num_epochs={num_epochs}_batch_size={batch_size}"
    pickle_file_name = f"dataset_{date_str}_{params_str}.pickle"
    # with open(pickle_file_name, 'wb') as f:
    #     pickle.dump(dataset, f)

    try:
        _run(datapipe_file_name, datapipe_object, nce_data.get_generator(), len(nce_data),
             nce_data.vocabulary_size(), context_size, num_noise_words, vec_dim,
             num_epochs, batch_size, lr, model_ver, vec_combine_method,
             save_all, generate_plot, model_ver_is_dbow)
    except KeyboardInterrupt:
        nce_data.stop()


def _run(datapipe_file_name,
         datapipe_object,
         data_generator,
         num_batches,
         vocabulary_size,
         context_size,
         num_noise_words,
         vec_dim,
         num_epochs,
         batch_size,
         lr,
         model_ver,
         vec_combine_method,
         save_all,
         generate_plot,
         model_ver_is_dbow):

    if model_ver_is_dbow:
        model = DBOW(vec_dim, num_docs=datapipe_object.length, num_words=vocabulary_size)
    else:
        model = DM(vec_dim, num_docs=datapipe_object.length, num_words=vocabulary_size)

    cost_func = NegativeSampling()
    optimizer = Adam(params=model.parameters(), lr=lr)

    if torch.cuda.is_available():
        print(f"using cuda")
        model.cuda()

    print("Dataset comprised of {:f} documents.".format(datapipe_object.length))
    print("Vocabulary size is {:f}.\n".format(vocabulary_size))
    print("Training started.")

    best_loss = float("inf")
    prev_model_file_path = None

    for epoch_i in range(num_epochs):
        epoch_start_time = time.time()
        loss = []

        for batch_i in range(num_batches):
            batch = next(data_generator)
            if torch.cuda.is_available():
                batch.cuda_()

            if model_ver_is_dbow:
                x = model.forward(batch.doc_ids, batch.target_noise_ids)
            else:
                x = model.forward(
                    batch.context_ids,
                    batch.doc_ids,
                    batch.target_noise_ids)

            x = cost_func.forward(x)

            loss.append(x.item())
            model.zero_grad()
            x.backward()
            optimizer.step()
            _print_progress(epoch_i, batch_i, num_batches)

        # end of epoch
        loss = torch.mean(torch.FloatTensor(loss))
        is_best_loss = loss < best_loss
        best_loss = min(loss, best_loss)

        state = {
            'epoch': epoch_i + 1,
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer_state_dict': optimizer.state_dict()
        }

        prev_model_file_path = save_training_state(
            datapipe_file_name.split('/')[-1],
            model_ver,
            vec_combine_method,
            context_size,
            num_noise_words,
            vec_dim,
            batch_size,
            lr,
            epoch_i,
            loss,
            state,
            save_all,
            generate_plot,
            is_best_loss,
            prev_model_file_path,
            model_ver_is_dbow)

        epoch_total_time = round(time.time() - epoch_start_time)
        print(" ({:f}s) - loss: {:.4f}".format(epoch_total_time, loss))


def _print_progress(epoch_i, batch_i, num_batches):
    progress = round((batch_i + 1) / num_batches * 100)
    if progress == 100:
        print("\rEpoch {:f}".format(epoch_i + 1))
        stdout.write(" - {:f}%".format(progress))
        stdout.flush()


if __name__ == '__main__':
    fire.Fire()

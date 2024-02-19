
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
sys.path.append(".")
import torchdata.datapipes as dp  # noqa: E402
from paragraphvec.utils import save_training_state  # noqa: E402
from paragraphvec.models import DM, DBOW  # noqa: E402
from paragraphvec.loss import NegativeSampling  # noqa: E402
from paragraphvec.data import NCEData, datasetClass  # noqa: E402

FILE_LIST = 'csv_files.csv'
entries = []

with open(FILE_LIST, 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)  # Skip the headers
    for row in reader:
        entries.append(row[0])

# print(entries)

# Custom DataPipe to handle gzip decompression


class GzFileLoader:
    def __init__(self, file_paths):
        self.average_line_length = 0
        self.number_of_lines = 0
        self.max_line_length = 0
        self.file_paths = file_paths

    def __iter__(self):
        file_counter = 0
        for file_path in self.file_paths:

            file_counter += 1
            print(
                f"Processing file {file_counter}/{len(self.file_paths)} average line length: {self.average_line_length} max line length: {self.max_line_length} number of lines: {self.number_of_lines}")
            try:
                with gzip.open(file_path, 'rt') as f:
                    for line in f:
                        rstripped_line = line.rstrip()
                        self.average_line_length = ((self.average_line_length * self.number_of_lines) +
                                                    len(rstripped_line[47::])) / (self.number_of_lines + 1)

                        if len(rstripped_line[47::]) > self.max_line_length:
                            self.max_line_length = len(rstripped_line[47::])
                        self.number_of_lines += 1
                        yield rstripped_line
            except Exception as e:
                print(f"Exception: {e}, File Path: {file_path}")
                continue


data_pipe = GzFileLoader(entries)


def start(data_file_name,
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
          num_workers=1):
    """Trains a new model. The latest checkpoint and the best performing
    model are saved in the *models* directory.

    Parameters
    ----------
    data_file_name: str
        Name of a file in the *data* directory.

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
    """
    if model_ver not in ('dm', 'dbow'):
        raise ValueError("Invalid version of the model")

    model_ver_is_dbow = model_ver == 'dbow'

    if model_ver_is_dbow and context_size != 0:
        raise ValueError("Context size has to be zero when using dbow")
    if not model_ver_is_dbow:
        if vec_combine_method not in ('sum', 'concat'):
            raise ValueError("Invalid method for combining paragraph and word "
                             "vectors when using dm")
        if context_size <= 0:
            raise ValueError("Context size must be positive when using dm")
    dataset = datasetClass(data_pipe)
    dataset.load_dataset()
    nce_data = NCEData(
        dataset,
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
    with open(pickle_file_name, 'wb') as f:
        pickle.dump(dataset, f)

    try:
        _run(data_file_name, dataset, nce_data.get_generator(), len(nce_data),
             nce_data.vocabulary_size(), context_size, num_noise_words, vec_dim,
             num_epochs, batch_size, lr, model_ver, vec_combine_method,
             save_all, generate_plot, model_ver_is_dbow)
    except KeyboardInterrupt:
        nce_data.stop()


def _run(data_file_name,
         dataset,
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
        model = DBOW(vec_dim, num_docs=dataset.length, num_words=vocabulary_size)
    else:
        model = DM(vec_dim, num_docs=dataset.length, num_words=vocabulary_size)

    cost_func = NegativeSampling()
    optimizer = Adam(params=model.parameters(), lr=lr)

    if torch.cuda.is_available():
        print(f"using cuda")
        model.cuda()

    print("Dataset comprised of {:f} documents.".format(dataset.length))
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
            data_file_name,
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

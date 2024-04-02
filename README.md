# Natural_Language_APIs

APIs of different NLP models and methods updated to latest python version.

## About the project

This project started from the work of @inejc at https://github.com/inejc/paragraph-vectors. The original project was written in Python 2.7 and I updated it to Python 3.10. Paragraph-vectors was the starting point of this project, and the plan is to explore different NLP models.

## Running example

In order to be able to train a large amount of text which doesn't fit in RAM, the project use a custom datapipe which yields the data from the disk.

```bash
source .venv/bin/activate
```

```bash
python paragraphvec/train.py start --dataset_file_name '' --vocab_and_counter_object_file_name '' --num_epochs 1000 --batch_size 500000 --num_noise_words 2 --vec_dim 100 --lr 1e-3 --model_ver 'dm' --context_size 4 --num_workers -1 --vec_combine_method 'sum' cache_objects true
```
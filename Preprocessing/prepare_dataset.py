"""
Function for downloading, cleaning and preparing a dataset
"""

"""
1. Data needs start and end tokens to signal the start and end of a sentence
2. Special characters need to be removed
3. A vocabulary with word indexing as the models outputs are numerical not strings
4. Sentence padding
"""

import os
import re
import io
import pathlib
import unicodedata

import tensorflow as tf

from numpy import ndarray
from tf.keras.preprocessing.text import Tokenizer

BATCH_SIZE = 64
NUM_EXAMPLES = 30000


def tokenize(texts: tuple) -> tuple[ndarray, Tokenizer]:
    """
    Tokenize the data - dictionary of each word and its index
    :param texts: Language texts
    :return: Tokenized words and tokenizer object
    """

    # Tokenizer vectorizes a text corpus
    lang_tokenizer = Tokenizer(
        filters='',  # Filtered from texts
        oov_token='<OOV>'  # Out Of Vocab Token - Replaces Tokens not in vocab with str
    )

    # Update internal vocab based on list of texts - create a dict of words and indexes
    lang_tokenizer.fit_on_texts(texts)

    # transforms each text in texts to a sequence of ints - convert each word into an int based on vocab list above
    tensor = lang_tokenizer.texts_to_sequences(texts)

    # Add padding after each sequence to match the length of the longest sequence
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer


def preprocess_sentence(w: str) -> str:
    # Convert all characters to ASCII
    w = ''.join(c for c in unicodedata.normalize('NFD', w.lower().strip()) if unicodedata.category(c) != 'Mn')

    # pad punctuation with whitespace
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def get_dataset(filename: str) -> tuple[ndarray, ndarray, Tokenizer, Tokenizer]:
    # Download and extract zip file - TF has datasets from http://www.manythings.org/anki/
    path_to_zip = tf.keras.utils.get_file(
        filename + '.zip',
        origin='http://storage.googleapis.com/download.tensorflow.org/data/' + filename + '.zip',
        extract=True
    )

    # Get file location
    path_to_file = os.path.dirname(path_to_zip) + "/" + filename[0:3] + ".txt"

    # Read file and split each line
    lines = io.open(path_to_file, encoding='UTF-8').read().strip().split('\n')

    # clean each sentence and split the sentence into input and output language
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:NUM_EXAMPLES]]

    # Split input language and target language
    inp, tar = zip(*word_pairs)

    # Tokenize each language
    input_tensor, input_tokenizer = tokenize(inp)
    target_tensor, target_tokenizer = tokenize(tar)

    return input_tensor, target_tensor, input_tokenizer, target_tokenizer


def main():
    # Datasets from http://www.manythings.org/anki/
    get_dataset('fra-eng')


if __name__ == '__main__':
    main()

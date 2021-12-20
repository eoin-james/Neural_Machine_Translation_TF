from Preprocessing.prepare_dataset import get_dataset


def main():
    # First Get the cleaned data set and tokenizers
    train_dataset, val_dataset, input_lang, target_lang = get_dataset(
        filename=data,
        num_examples=num_examples,
        buffer_size=buffer_size,
        batch_size=batch_size
    )

    vocab_inp_size = len(input_lang.word_index) + 1
    vocab_tar_size = len(target_lang.word_index) + 1
    # max_length_input = example_input_batch.shape[1]
    # max_length_output = example_target_batch.shape[1]

    embedding_dim = 256
    units = 1024
    steps_per_epoch = num_examples // batch_size

    print(vocab_inp_size)


if __name__ == '__main__':
    data = 'fra-eng'
    num_examples = 30_000  # Less = Faster training
    buffer_size = 64
    batch_size = 32_000

    main()

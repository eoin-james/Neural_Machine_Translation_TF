from Preprocessing.prepare_dataset import get_dataset


def main():
    # First Get the cleaned data set
    input_tensor, target_tensor, input_tokenizer, target_tokenizer = get_dataset('fra-eng')


if __name__ == '__main__':
    main()

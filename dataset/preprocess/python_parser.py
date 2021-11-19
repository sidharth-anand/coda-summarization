import argparse
import re
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-dir', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--test-split', type=float, default=0.2)
    parser.add_argument('--validation-split', type=float, default=0.2)

    return parser.parse_args()


def clean_code(declaration_body: str) -> str:
    return declaration_body


def clean_comment(description: str) -> str:
    description = description.replace(' DCNL DCSP', ' ')
    description = description.replace(' DCNL ', ' ')
    description = description.replace(' DCSP ', ' ')

    description = description.lower()

    description = description.replace("this's", 'this is')
    description = description.replace("that's", 'that is')
    description = description.replace("there's", 'there is')

    description = description.replace('\\', '')
    description = description.replace('``', '')
    description = description.replace('`', '')
    description = description.replace('\'', '')

    removes = re.findall("(?<=[(])[^()]+[^()]+(?=[)])", description)
    for r in removes:
        description = description.replace('('+r+')', '')

    urls = re.findall(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', description)
    for url in urls:
        description = description.replace(url, 'URL')

    description = description.split('.')[0]
    description = description.split(',')[0]
    description = description.split(':param')[0]
    description = description.split('@param')[0]
    description = description.split('>>>')[0]

    description = description.strip().strip('\n') + ' .'

    return description


def generate_and_save_pairs(dataset_path: str, save_path: str) -> None:
    with open(save_path + '/all.code', 'w') as code_file:
        with open(save_path + '/all.comment', 'w') as comment_file:
            with open(dataset_path + '/data_ps.declbodies', 'r') as declbodies_file:
                with open(dataset_path + '/data_ps.descriptions', 'r', errors='ignore') as descriptions_file:
                    declbodies_lines = declbodies_file.readlines()
                    descriptions_lines = descriptions_file.readlines()

                    for i in range(len(declbodies_lines)):
                        code = clean_code(declbodies_lines[i])
                        comment = clean_comment(descriptions_lines[i])

                        if not comment.startswith('todo') and len(comment.split()) > 2 and comment[0].isalpha():
                            code_file.write(code)
                            comment_file.write(comment + '\n')


def save_split(train_split: float, test_split: float, validation_split: float, save_path: str, extenstion: str, train, test, validation) -> None:
    idx = 0

    with open(save_path + f'/all.{extenstion}', 'r') as all_file:
        with open(save_path + f"/train_{train_split:.1f}{validation_split:.1f}{test_split:.1f}.{extenstion}", 'w') as train_file:
            with open(save_path + f"/validation_{train_split:.1f}{validation_split:.1f}{test_split:.1f}.{extenstion}", 'w') as validation_file:
                with open(save_path + f"/test_{train_split:.1f}{validation_split:.1f}{test_split:.1f}.{extenstion}", 'w') as test_file:
                    for a_line in all_file.readlines():
                        if idx in train:
                            train_file.write(a_line)
                        elif idx in validation:
                            validation_file.write(a_line)
                        elif idx in test:
                            test_file.write(a_line)
                        idx += 1


def split_dataset(test_split: float, validation_split: float, save_path: str) -> None:
    train_split = 1 - test_split - validation_split

    num = 0
    with open(save_path + "/all.code", 'r') as training_file:
        num = len(training_file.readlines())

    sidx = np.random.permutation(num)
    n_dev = int(np.round(num * validation_split))
    validation, data = (sidx[:n_dev], sidx[n_dev:])
    print('Number of pairs in dev set: %d.' % len(validation))

    pidx = np.random.permutation(len(data))
    n_train = int(np.round(num * train_split))
    train, test = (data[pidx[:n_train]], data[pidx[n_train:]])
    print('Number of pairs in train set: %d.' % len(train))
    print('Number of pairs in test set: %d.' % len(test))

    save_split(train_split, test_split, validation_split,
               save_path, 'code', train, test, validation)
    save_split(train_split, test_split, validation_split,
               save_path, 'comment', train, test, validation)


if __name__ == '__main__':
    options = parse_arguments()

    print("Generating and saving Code/Comment Pairs...")
    generate_and_save_pairs(options.dataset_dir, options.save_dir)

    print('Generating Train/Test/Validation splits...')
    split_dataset(options.test_split,
                  options.validation_split, options.save_dir)

    print('Done')

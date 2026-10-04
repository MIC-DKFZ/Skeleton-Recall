from os.path import dirname, join

EXT_TRAINER_PATH = join(dirname(__file__), 'training', 'nnUNetTrainer')


def print_ext_trainer_path():
    print(EXT_TRAINER_PATH)

#Scripts to create a symbolic link between data
import os

def make_dir(p):
    if not os.path.exists(p):
        os.mkdir(p)

def link(src, dest):
    if not os.path.exists(dest):
        os.symlink(src, dest, target_is_directory=True)

make_dir('fruits-360-small')

classes = [
    'Apple Golden 1',
    'Avocado',
    'Lemon',
    'Mango',
    'Kiwi',
    'Banana',
    'Strawberry',
    'Raspberry'
]


# Where grab the data: https://www.kaggle.com/moltean/fruits
train_path_from = os.path.abspath('/home/roger/Documents/machine_learning_examples/large_files/fruits-360/Training')
valid_path_from = os.path.abspath('/home/roger/Documents/machine_learning_examples/large_files/fruits-360/Test')

train_path_to = os.path.abspath('fruits-360-small/Training')
valid_path_to = os.path.abspath('fruits-360-small/Validation')

make_dir(train_path_to)
make_dir(valid_path_to)

for c in classes:
    link(train_path_from + '/' + c, train_path_to + '/' + c)
    link(valid_path_from + '/' + c, valid_path_to + '/' + c)
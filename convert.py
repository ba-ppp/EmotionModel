import pathlib
from util import *
import pandas as pd

def convert(folder_image: str, train_percent: int):
    current_path = pathlib.Path().resolve()
    input_path = os.path.join(current_path, 'input\\')
    paths = []

    data = {
        'id': [],
        'type': []
    }

    sub_folders = list_folder(folder_image)
    for folder in sub_folders:
        emotion_type = get_folder_name(folder)
        files = list_files(folder)
        for file in files:
            # add to data
            data['type'].append(emotion_type)
            id = get_filename_without_extension(file)
            data['id'].append(id)

            # move files to input
            paths.append(get_filename_path(folder, file))
    train_path, test_path = divide_train_test(paths, train_percent)
    move_files(train_path, input_path + 'train\\')
    move_files(test_path, input_path + 'test\\')

    # convert to csv
    df = pd.DataFrame(data)
    df.to_csv("./input/labels.csv", index=False)

if __name__ == '__main__':
    image_path = './images'
    convert(image_path, 70)
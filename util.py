from os import listdir
from os.path import isfile, join
import os
from typing import List
from pathlib import Path
from shutil import copyfile

def get_folder_name(folder: str) -> str:
    return os.path.basename(folder)

def get_filename_without_extension(file: str) -> str:
    return os.path.splitext(file)[0]

def get_filename_path(folder: str, file: str)-> str:
    return os.path.join(os.path.abspath(folder), file)

def list_folder(main_folder: str) -> List[str]:
    # list all folder in folder images and foldername
    subfolders = [ f.path for f in os.scandir(main_folder) if f.is_dir() ]
    return subfolders

def list_files(folder: str) -> List[str]:
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    return files

def move_files(paths: List[str], destination: str):
    for path in paths:
        copyfile(path, destination + path.split('\\')[-1])

def divide_train_test(paths: List[str], train_percent: int):
    if (train_percent < 50):
        print('Train must greater than test')
        return
    train_count = round(train_percent / 100 * len(paths))
    return [paths[0: train_count], paths[train_count: len(paths)]]
    

    
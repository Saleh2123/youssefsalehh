########################################################################
# import python-library
########################################################################
# default
import os
import sys
import csv
import shutil
import argparse
from pathlib import Path
import glob
import random
import itertools
import re
import pathlib

# additional
import numpy as np
import librosa
import librosa.core
import librosa.feature
import yaml
import urllib.request
import urllib.error
import zipfile
import shutil
import time
import fasteners
import pickle
import pickletools
import time
import datetime
from enum import Enum, auto
########################################################################


########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""


import logging

logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


########################################################################


########################################################################
# version
########################################################################
_versions_ = "1.0.0"
########################################################################


########################################################################
# download dataset parameter
########################################################################
DOWNLOAD_PATH_YAML_DICT = {
    "DCASE2023T2":"datasets/download_path_2023.yaml",
    "legacy":"datasets/download_path_legacy.yaml",
}        
########################################################################


########################################################################
# load parameter.yaml
########################################################################
# def yaml_load():
#     with open("baseline.yaml") as stream:
#         param = yaml.safe_load(stream)
#     return param

########################################################################


########################################################################
# file I/O
########################################################################
# wav file input
ROOT_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../"
os.chdir(ROOT_DIR)

# labeled data path
EVAL_DATA_LIST_PATH = {
    "DCASE2020T2": f"{ROOT_DIR}/datasets/eval_data_list_2020.csv",
    "DCASE2021T2": f"{ROOT_DIR}/datasets/eval_data_list_2021.csv",
    "DCASE2022T2": f"{ROOT_DIR}/datasets/eval_data_list_2022.csv",
    "DCASE2023T2": f"{ROOT_DIR}/datasets/eval_data_list_2023.csv",
}

FILENAME_COL = 0
LABELING_FILENAME_COL = 1
MACHINE_TYPE_COL = 0

CHK_MACHINE_TYPE_LINE = 2
def rename_wav(dataset_parent_dir, dataset_type):
    dataset_dir = str(Path(f"{ROOT_DIR}/{dataset_parent_dir}/raw/").relative_to(ROOT_DIR))
    eval_data_list_path = EVAL_DATA_LIST_PATH[dataset_type]  # Use dataset_type here
    if not eval_data_list_path:
        return None
    
    if os.path.exists(eval_data_list_path):
        with open(eval_data_list_path) as fp:
            eval_data_list = list(csv.reader(fp))
    else:
        print(f"Err: eval_data_list.csv not found: {eval_data_list_path}")
        sys.exit(1)

    count = 0
    print('copy... : test -> test_rename')
    for eval_data in eval_data_list:
        if len(eval_data) < CHK_MACHINE_TYPE_LINE:
            machine_type = eval_data[MACHINE_TYPE_COL]
            default_dir = dataset_dir.lower() + "/" + machine_type + "/test"
            save_dir = dataset_dir.lower() + "/" + machine_type + "/test_rename"
            if not os.path.exists(save_dir):
                Path(save_dir).mkdir(parents=True, exist_ok=True)
            count = 0
            sys.stdout.write('\n')
            sys.stdout.flush()
        else:
            if os.path.exists(default_dir + "/" + eval_data[FILENAME_COL]):
                shutil.copy2(
                    default_dir + "/" + eval_data[FILENAME_COL],
                    save_dir + "/" + eval_data[LABELING_FILENAME_COL])
                count += 1
            sys.stdout.write(f'\r\t{machine_type}: {str(count)} files\tsaved dir: {save_dir}')
            sys.stdout.flush()
    sys.stdout.write('\n')

if __name__ == "_main_":
    parser = argparse.ArgumentParser(
            description='Main function to call training for different AutoEncoders')
    parser.add_argument("--dataset_parent_dir", type=str, default="data",
                        help="saving datasets directory name.")
    parser.add_argument("dataset_type", type=str, choices=["DCASE2020T2", "DCASE2021T2", "DCASE2022T2", "DCASE2023T2"],
                        help="Dataset name to rename.")
    args = parser.parse_args()
    print(args)

    rename_wav(
        dataset_parent_dir=f"{args.dataset_parent_dir}/{args.dataset_type}/eval_data",
        dataset_type=args.dataset_type
    )
    
def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


########################################################################


########################################################################
# feature extractor
########################################################################
def file_to_vectors(file_name,
                    n_mels=64,
                    n_frames=5,
                    n_fft=1024,
                    hop_length=512,
                    power=2.0,
                    fmax=None,
                    fmin=None,
                    win_length=None,):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # generate melspectrogram using librosa
    y, sr = file_load(file_name, mono=True)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                        sr=sr,
                                                        n_fft=n_fft,
                                                        hop_length=hop_length,
                                                        n_mels=n_mels,
                                                        power=power,
                                                        fmax=fmax,
                                                        fmin=fmin,
                                                        win_length=win_length,
                                                        )

    # convert melspectrogram to log mel energies
    log_mel_spectrogram = 20.0 / power * np.log10(np.maximum(mel_spectrogram, sys.float_info.epsilon))

    # calculate total vector size
    n_vectors = len(log_mel_spectrogram[0, :]) - n_frames + 1

    # skip too short clips
    if n_vectors < 1:
        return np.empty((0, dims))

    # generate feature vectors by concatenating multi frames
    vectors = np.zeros((n_vectors, dims))
    for t in range(n_frames):
        vectors[:, n_mels * t : n_mels * (t + 1)] = log_mel_spectrogram[:, t : t + n_vectors].T

    return vectors



########################################################################


########################################################################
# get directory paths according to mode
########################################################################
def select_dirs(param, mode):
    """
    param : dict
        baseline.yaml data

    return :
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode:
        logger.info("load_directory <- development")
        query = os.path.abspath("{base}/*".format(base=param["dev_directory"]))
    else:
        logger.info("load_directory <- evaluation")
        query = os.path.abspath("{base}/*".format(base=param["eval_directory"]))
    dirs = sorted(glob.glob(query))
    dirs = [f for f in dirs if os.path.isdir(f)]
    return dirs


########################################################################


########################################################################
# get machine IDs
########################################################################
def get_section_names(target_dir,
                      dir_name,
                      ext="wav"):
    """
    target_dir : str
        base directory path
    dir_name : str
        sub directory name
    ext : str (default="wav)
        file extension of audio files

    return :
        section_names : list [ str ]
            list of section names extracted from the names of audio files
    """
    # create test files
    query = os.path.abspath("{target_dir}/{dir_name}/*.{ext}".format(target_dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(query))
    # extract section names
    section_names = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('section_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return section_names


########################################################################


########################################################################
# get the list of wave file paths
########################################################################
def file_list_generator(target_dir,
                        section_name,
                        unique_section_names,
                        dir_name,
                        mode,
                        train,
                        prefix_normal="normal",
                        prefix_anomaly="anomaly",
                        prefix_masked="pitched",  # Add prefix_masked argument
                        ext="wav"):
    """
    target_dir : str
        base directory path
    section_name : str
        section name of audio file in <<dir_name>> directory
    dir_name : str
        sub directory name
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    prefix_masked : str (default="masked")  # Add prefix_masked argument
        masked directory name
    ext : str (default="wav")
        file extension of audio files

    return :
        if the mode is "development":
            files : list [ str ]
                audio file list
            labels : list [ boolean ]
                label info. list
                * normal/anomaly/masked = 0/1/2
        if the mode is "evaluation":
            files : list [ str ]
                audio file list
    """
    logger.info("target_dir : {}".format(target_dir + "_" + section_name))
    condition_array = np.eye(len(unique_section_names))

    # development
    if mode:
        query_normal = os.path.abspath("{target_dir}/{dir_name}/{section_name}*{prefix_normal}_*.{ext}".format(target_dir=target_dir,
                                                                                                                  dir_name=dir_name,
                                                                                                                  section_name=section_name,
                                                                                                                  prefix_normal=prefix_normal,
                                                                                                                  ext=ext))
        normal_files = sorted(glob.glob(query_normal))
        if len(normal_files) == 0:
            normal_files = sorted(
                glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                      dir_name=dir_name,
                                                                                      prefix_normal=prefix_normal,
                                                                                      id_name=section_name,
                                                                                      ext=ext)))
        normal_labels = np.zeros(len(normal_files))

        query_anomaly = os.path.abspath("{target_dir}/{dir_name}/{section_name}*{prefix_anomaly}_*.{ext}".format(target_dir=target_dir,
                                                                                                                    dir_name=dir_name,
                                                                                                                    section_name=section_name,
                                                                                                                    prefix_anomaly=prefix_anomaly,
                                                                                                                    ext=ext))
        anomaly_files = sorted(glob.glob(query_anomaly))
        if len(anomaly_files) == 0:
            anomaly_files = sorted(
                glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                      dir_name=dir_name,
                                                                                      prefix_anomaly=prefix_anomaly,
                                                                                      id_name=section_name,
                                                                                      ext=ext)))
        anomaly_labels = np.ones(len(anomaly_files))

        query_masked = os.path.abspath("{target_dir}/{dir_name}/{section_name}*{prefix_masked}_*.{ext}".format(target_dir=target_dir,
                                                                                                                    dir_name=dir_name,
                                                                                                                    section_name=section_name,
                                                                                                                    prefix_masked=prefix_masked,
                                                                                                                    ext=ext))
        masked_files = sorted(glob.glob(query_masked))
        if len(masked_files) == 0:
            masked_files = sorted(
                glob.glob("{dir}/{dir_name}/{prefix_masked}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                      dir_name=dir_name,
                                                                                      prefix_masked=prefix_masked,
                                                                                      id_name=section_name,
                                                                                      ext=ext)))
        masked_labels = np.ones(len(masked_files)) * 2

        files = np.concatenate((normal_files, anomaly_files, masked_files), axis=0)
        labels = np.concatenate((normal_labels, anomaly_labels, masked_labels), axis=0)

        logger.info("#files : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n========================================")

    # evaluation
    else:
        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_.{ext}".format(target_dir=target_dir,
                                                                                           dir_name=dir_name,
                                                                                           section_name=section_name,
                                                                                           ext=ext))
        files = sorted(glob.glob(query))
        if train:
            normal_files = sorted(glob.glob(query))
            labels = np.zeros(len(normal_files))
        else:
            labels = None
        logger.info("#files : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n=========================================")

    condition = []
    for _, file in enumerate(files):
        for j, unique_section_name in enumerate(unique_section_names):
            if unique_section_name in file:
                condition.append(condition_array[j])

    return files, labels, condition


def download_raw_data(
    target_dir,
    dir_name,
    machine_type,
    data_type,
    dataset,
    root
):
    if dataset == "DCASE2023T2":
        download_path_yaml = DOWNLOAD_PATH_YAML_DICT["DCASE2023T2"]
    else:
        download_path_yaml = DOWNLOAD_PATH_YAML_DICT["legacy"]
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    with open(download_path_yaml, "r") as f:
        file_url = yaml.safe_load(f)[dataset]

    lock_file_path = get_lockfile_path(target_dir=target_dir)
    if os.path.exists(f"{target_dir}/{dir_name}") and not os.path.isfile(lock_file_path):
        print(f"{target_dir}/{dir_name} is already downloaded")
        return
    print(f"{target_dir}/{dir_name} is not directory.\nperform dataset download.")

    lock = fasteners.InterProcessReaderWriterLock(lock_file_path)
    try:
        lock.acquire_write_lock()
    except:
        print(f"{target_dir}/{dir_name} is already downloaded")
        return

    if os.path.exists(f"{target_dir}/{dir_name}"):
        print(f"{target_dir}/{dir_name} is already downloaded")
        release_write_lock(
            lock=lock,
            lock_file_path=lock_file_path,
        )
        return

    for i in np.arange(len(file_url[machine_type][data_type])):
        zip_file_basename = os.path.basename(file_url[machine_type][data_type][i])
        zip_file_path = f"{target_dir}/../{zip_file_basename}"
        try:
            if not zipfile.is_zipfile(zip_file_path):
                print(f"Downloading...\n\tURL: {file_url[machine_type][data_type][i]}\r\tDownload: {zip_file_path}")
                urllib.request.urlretrieve(
                    file_url[machine_type][data_type][i],
                    zip_file_path,
                    urllib_progress
                )
        except urllib.error.URLError as e:
            print(e)
            print("retry dataset download")
            download_raw_data(
                target_dir,
                dir_name,
                machine_type,
                data_type,
                dataset,
                root
            )
            return

        with zipfile.ZipFile(zip_file_path, "r") as obj_zip:
            zip_infos = obj_zip.infolist()
            for zip_info in zip_infos:
                if zip_info.is_dir():
                    os.makedirs(f"{target_dir}/../{zip_info.filename}", exist_ok=True)
                elif not os.path.exists(f"{target_dir}/../{zip_info.filename}"):
                    sys.stdout.write(f"\runzip: {target_dir}/../{zip_info.filename}")
                    obj_zip.extract(zip_info, f"{target_dir}/../")                    
            print("\n")
    
    if dataset == "DCASE2021T2":
        test_data_path = f"{target_dir}/test"
        split_data_path_list = [
            f"{target_dir}/source_test",
            f"{target_dir}/target_test",
        ]
        os.makedirs(test_data_path, exist_ok=True)
        for split_data_path in split_data_path_list:
            shutil.copytree(split_data_path, test_data_path, dirs_exist_ok=True)

    if data_type == "eval":
        rename_wav(
            dataset_parent_dir=root,
            dataset_type=dataset,
        )

    release_write_lock(
        lock=lock,
        lock_file_path=lock_file_path
    )
    return

def urllib_progress (block_count, block_size, total_size):
    progress_value = block_count * block_size / total_size * 100
    sys.stdout.write(f"\r{block_count*block_size/(1024*2):.2f}MB / {total_size/(1024*2):.2f}MB ({progress_value:.2f}%%)")

def get_lockfile_path(target_dir):
    return f"{target_dir}/lockfile"

def release_write_lock(lock, lock_file_path):
    print(f"{datetime.datetime.now()}\trelease write lock : {lock_file_path}")
    lock.release_write_lock()
    if os.path.isfile(lock_file_path):
        try:
            os.remove(lock_file_path)
        except OSError:
            print(f"can not remove {lock_file_path}")

def release_read_lock(lock, lock_file_path):
    print(f"{datetime.datetime.now()}\trelease read lock : {lock_file_path}")
    lock.release_read_lock()
    if os.path.isfile(lock_file_path):
        try:
            os.remove(lock_file_path)
        except OSError:
            print(f"can not remove {lock_file_path}")

def is_enabled_pickle(pickle_path):
    opcodes = []
    with open(pickle_path, "rb") as f:
        pickle = f.read()
        output = pickletools.genops(pickle=pickle)
        for opcode in output:
            opcodes.append(opcode[0])
    return ("PROTO","STOP") == (opcodes[0].name, opcodes[-1].name)

########################################################################
# get machine type and section id in yaml
########################################################################
YAML_PATH = {
    "legacy":"datasets/machine_type_legacy.yaml",
    "DCASE2023T2_dev":"datasets/machine_type_2023_dev.yaml",
    "DCASE2023T2_eval":"datasets/machine_type_2023_eval.yaml",
}

def get_machine_type_dict(dataset_name, mode=True):
    if dataset_name in ["DCASE2020T2", "DCASE2021T2", "DCASE2022T2"]:
        yaml_path = YAML_PATH["legacy"]
    elif dataset_name == "DCASE2023T2" and not mode:
        yaml_path = YAML_PATH["DCASE2023T2_eval"]
    else: 
        yaml_path = YAML_PATH["DCASE2023T2_dev"]
    
    with open(yaml_path, "r") as f:
        machine_type_dict = yaml.safe_load(f)
        return machine_type_dict[dataset_name]
import tensorflow as tf
import numpy as np
from waymo_open_dataset.protos import scenario_pb2
import os
import pickle


def get_wod_prediction_challenge_split(
        split: str,
        dataroot: str = '/data/sets/wod_motion',
        ext: str = ''):
    """
    Gets a list of strings of tf_records for each split.
    :param split: One of 'training', 'validation', 'testing',
            'testing_interactive', 'validation_interactive'.
    :param dataroot: Path to the wodMotion dataset.
    :return: List of record file names belonging to the split.
    NOTE: the file name is the absolute path.
    """
    if split not in {'training', 'validation', 'testing',
                     'validation_interactive', 'testing_interactive', 'training_20s'}:
        raise ValueError("split must be one of \
            (training, validation, testing, validation_interactive, \
            testing_interactive)")
    file_list = []
    data_path = os.path.join(dataroot, split)
    if len(ext) == 0:
        # file_list = [os.path.join(dataroot, split, f)
        #            for f in listdir(data_path) if isfile(join(data_path, f))]
        for root, folders, files in os.walk(data_path):
            for f in files:
                file_list.append(os.path.join(root, f))
    else:
        for root, folders, files in os.walk(data_path):
            for f in files:
                e = os.path.splitext(f)[-1].lstrip('.').lower()
                if e == ext:
                    file_list.append(os.path.join(root, f))
    return file_list

def main(src_path: str, dst_path: str, split: str) -> None:
    """
    Dump samples in tfrecord to pickles.
    Args:
        src_path: root path containing the splits: train, validation, testing
        dst_path: root path to save the splits: train, validation, testing
        split: one of [train, validation, testing]
    Return: None
    """
    tfrecord_list = get_wod_prediction_challenge_split(split, src_path)
    pickle_path = os.path.join(dst_path, split)
    if not os.path.isdir(pickle_path):
        os.makedirs(pickle_path, mode=0o777)   
    sdc_token_list = []
    for tf_file in tfrecord_list:
        dataset = tf.data.TFRecordDataset(tf_file, compression_type='')
        for data in dataset:
            proto_string = data.numpy()
            proto = scenario_pb2.Scenario()
            proto.ParseFromString(proto_string)
            sdc_token_list.append(proto.scenario_id + "_" + str(proto.sdc_track_index))
            outfile = os.path.join(pickle_path, proto.scenario_id + ".pkl")
            with open(outfile, 'wb') as handle:
                pickle.dump(proto, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        print("finished split {}".format(tf_file))
    # save sdc token list to a txt file in dst path
    sdc_token_file = os.path.join(dst_path, split + "_token_list.txt")
    with open(sdc_token_file, 'w') as f:
        for item in sdc_token_list:
            f.write("%s\n" % item)

        
if __name__ == '__main__':
    wod_path = 'PATH_TO_WOD_MOTION_DATASET'
    src_path = os.path.join(wod_path, 'scenario/')  
    dst_path = os.path.join(wod_path, 'scenario_pkl_v1_2/')
    splits = ['validation', 'training', 'testing']
    for split in splits:
        print("start split {}".format(split))
        main(src_path, dst_path, split)
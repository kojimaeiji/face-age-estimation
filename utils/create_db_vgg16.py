import numpy as np
import cv2
import argparse
from utils import get_meta
import random
from joblib import Parallel, delayed


def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="path to output database mat file")
    parser.add_argument("--db", type=str, default="wiki",
                        help="dataset; wiki or imdb")
    parser.add_argument("--img_size", type=int, default=224,
                        help="output image size")
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="minimum face_score")
    parser.add_argument("--max_count", type=int, default=None,
                        help="max_count")
    parser.add_argument("--max_num_per_file", type=int, default=32,
                        help="max_num_per_file")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="train_ratio")
    args = parser.parse_args()
    return args


def path_concat(path_list):
    path = ''
    for s in path_list:
        path += s + '/'
    return path


def write_mat(out_imgs, out_genders, out_ages, file_full
              ):
    image = np.array(out_imgs)
    age = np.array(out_ages)
    gender = np.array(out_genders)
    np.savez_compressed(file_full, image=image, age=age, gender=gender)


def get_passed(length, min_score, age, face_score, gender):
    return [i for i in range(length) if 0 <= age[i] <= 100
            and face_score[i] >= min_score and ~np.isnan(gender[i])]


def minibatch_gen(data, size):
    length = len(data)
    iternum = length // size + 1 if length % size != 0 else length // size
    mini_list = []
    for i in range(iternum):
        mini = data[i * size:(i + 1) * size] if (i + 1) * size <= length \
            else data[i * size:]
        mini_list.append(mini)
    return mini_list


def make_meta(idxs, images, ages, genders):
    return [(images[i][0], ages[i], genders[i]) for i in idxs]


def process(mini_type, root_path, img_size, outpath_prefix,
            file_name, i, mini):
    '''
    process mini batch and save to file
    '''
    out_genders = []
    out_ages = []
    out_imgs = []
    if mini_type == 'train':
        filefull = '%s/%s-tr-%s' % (outpath_prefix,
                                    file_name,
                                    i)
    else:
        filefull = '%s/%s-cv-%s' % (outpath_prefix,
                                    file_name,
                                    i)
    for m in mini:
        # gender
        out_genders.append(int(m[2]))
        # age
        out_ages.append(m[1])
        img = cv2.imread(root_path + str(m[0]), 1)
        img = cv2.resize(img, (img_size, img_size))
        out_imgs.append(img)

    # save to file
    write_mat(out_imgs, out_genders, out_ages,
              filefull
              )


def split_meta_indexes(idxs, train_length, max_count=None):
    if max_count is None:
        return idxs[0:train_length], idxs[train_length:]
    else:
        return idxs[0:train_length], idxs[train_length:max_count]


def main():
    args = get_args()
    output_path = args.output
    db = args.db
    max_count = args.max_count
    img_size = args.img_size
    min_score = args.min_score
    max_num_per_file = args.max_num_per_file
    train_ratio = args.train_ratio

    root_path = "data/{}_crop/".format(db)
    mat_path = root_path + "{}.mat".format(db)
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(
        mat_path, db)

    length = len(face_score)

    # make out filepath
    outpath_prefix = output_path.split('/')[:-1]
    outpath_prefix = path_concat(outpath_prefix)
    filename = output_path.split('/')[-1].split('.')[0]

    # filter bad images and get passed indexes
    indexes = get_passed(length, min_score, age, face_score, gender)

    # shuffle indexes
    random.shuffle(indexes)
    effective_length = len(indexes)
    train_length = int(
        max_count * train_ratio) if max_count is not None else int(effective_length * train_ratio)
    print('train_length=%s' % train_length)

    # spilit meta data
    train_idxs, val_idxs = split_meta_indexes(
        indexes, train_length, max_count=max_count)

    # get meta data from filtered indexes
    train_metas = make_meta(train_idxs, full_path, age, gender)
    val_metas = make_meta(val_idxs, full_path, age, gender)

    # split to mini batchs
    mini_list = minibatch_gen(train_metas, max_num_per_file)

    # process and save train data
    Parallel(n_jobs=-1, verbose=5)([delayed(process)('train', root_path,
                                                     img_size,
                                                     outpath_prefix,
                                                     filename, i, mini) for i, mini in enumerate(mini_list)])

    # process and save validation data
    mini_list = minibatch_gen(val_metas, max_num_per_file)
    Parallel(n_jobs=-1, verbose=5)([delayed(process)('val', root_path,
                                                     img_size,
                                                     outpath_prefix,
                                                     filename, i, mini) for i, mini in enumerate(mini_list)])


if __name__ == '__main__':
    main()

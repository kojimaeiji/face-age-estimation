import numpy as np
import cv2
import scipy.io
import argparse
from tqdm import tqdm
from utils import get_meta
import random


def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#    parser.add_argument("--input", "-i", type=str, required=True,
#                        help="path to input database mat file")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="path to output database mat file")
    parser.add_argument("--db", type=str, default="wiki",
                        help="dataset; wiki or imdb")
    parser.add_argument("--img_size", type=int, default=32,
                        help="output image size")
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="minimum face_score")
    parser.add_argument("--max_count", type=int, default=None,
                        help="max_count")
    parser.add_argument("--max_num_per_file", type=int, default=64,
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


def write_mat(out_imgs, out_genders, out_ages, db, img_size,
              min_score, total_count, train_length, outpath_prefix,
              filename, file_count, ext
              ):
    output = {"image": np.array(out_imgs),
              "gender": np.array(out_genders),
              "age": np.array(out_ages),
              "db": db,
              "img_size": img_size,
              "min_score": min_score}
    if total_count <= train_length:
        scipy.io.savemat('%s/%s-tr-%s.%s' % (outpath_prefix,
                                             filename,
                                             file_count, ext),
                         output)
    else:
        scipy.io.savemat('%s/%s-cv-%s.%s' % (outpath_prefix,
                                             filename,
                                             file_count, ext),
                         output)


def get_passed(length, min_score, age, face_score, gender):
    return [i for i in range(length) if 0<=age[i]<=100 and face_score[i]>=min_score and ~np.isnan(gender[i])]

def main():
    args = get_args()
    output_path = args.output
    db = args.db
#    mat_path = args.input
    max_count = args.max_count
    img_size = args.img_size
    min_score = args.min_score
    max_num_per_file = args.max_num_per_file
    train_ratio = args.train_ratio

    root_path = "data/{}_crop/".format(db)
    mat_path = root_path + "{}.mat".format(db)
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(
        mat_path, db)

    out_genders = []
    out_ages = []
    out_imgs = []

    length = len(face_score)
#    max_num_per_file = num_per_file
    file_count = 0
    outpath_prefix = output_path.split('/')[:-1]
    outpath_prefix = path_concat(outpath_prefix)
    filename = output_path.split('/')[-1].split('.')[0]
    ext = output_path.split('/')[-1].split('.')[1]
    total_count = 0
    indexes = get_passed(length, min_score, age, face_score, gender)
    random.shuffle(indexes)
    effective_length = len(indexes)
    train_length = int(
        max_count * train_ratio) if max_count is not None else int(effective_length * train_ratio)
    print('train_length=%s' % train_length)
    for i in tqdm(indexes):
        #print('total_count=%s' % total_count)
#        if face_score[i] < min_score:
#            print('face score bad')
#            continue

#         if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
#             continue

#        if ~(0 <= age[i] <= 100):
#            print('age bad')
#            continue

#        if np.isnan(gender[i]):
#            print('gender bad')
#            continue

        out_genders.append(int(gender[i]))
        out_ages.append(age[i])
        img = cv2.imread(root_path + str(full_path[i][0]), 1)
        img = cv2.resize(img, (img_size, img_size))
        img = img[...,::-1]
        img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)

        out_imgs.append(img)
        total_count += 1
        if max_count is not None and total_count >= max_count:
            break

        if (len(out_imgs) % max_num_per_file == 0 and len(out_imgs) > 0) or \
                total_count == train_length:
            write_mat(out_imgs, out_genders, out_ages, db, img_size,
                      min_score, total_count, train_length, outpath_prefix,
                      filename, file_count, ext
                      )
            file_count += 1
            if total_count == train_length:
                print('train range end')
                file_count = 0
            out_imgs = []
            out_genders = []
            out_ages = []
            output = {}
    write_mat(out_imgs, out_genders, out_ages, db, img_size,
                    min_score, total_count, train_length, outpath_prefix,
                      filename, file_count, ext
                )


if __name__ == '__main__':
    main()

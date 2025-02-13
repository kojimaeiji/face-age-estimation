import numpy as np
import cv2
import scipy.io
import argparse
from tqdm import tqdm
from utils import get_meta


def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    output_path = args.output
    db = args.db
    max_count = args.max_count
    img_size = args.img_size
    min_score = args.min_score

    root_path = "data/{}_crop/".format(db)
    mat_path = root_path + "{}.mat".format(db)
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(
        mat_path, db)

    out_genders = []
    out_ages = []
    out_imgs = []

    length = len(face_score)
    for i in tqdm(range(length)):
        if face_score[i] < min_score:
            continue

        #if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
        #    continue

        if ~(0 <= age[i] <= 100):
            continue

        if np.isnan(gender[i]):
            continue

        out_genders.append(int(gender[i]))
        out_ages.append(age[i])
        img = cv2.imread(root_path + str(full_path[i][0]), 1)
        img = cv2.resize(img, (img_size, img_size))
        img = img[...,::-1]
        img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
        #img = img
        out_imgs.append(img)
        if max_count is not None and len(out_imgs) >= max_count:
            break
            

    output = {"image": np.array(out_imgs), "gender": np.array(out_genders), "age": np.array(out_ages),
              "db": db, "img_size": img_size, "min_score": min_score}
    scipy.io.savemat(output_path, output)


if __name__ == '__main__':
    main()

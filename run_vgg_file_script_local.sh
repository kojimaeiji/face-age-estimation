DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=face_age_rec_$DATE
#export GCS_JOB_DIR=/Users/saboten/mljob
export GCS_JOB_DIR=/home/jiman/mljob/$JOB_NAME
echo $GCS_JOB_DIR
#rm -rf /Users/saboten/mljob/*
#rm -rf /home/jiman/mljob/*

gcloud ml-engine local train \
  --job-dir $GCS_JOB_DIR \
--module-name trainer.task \
--package-path trainer/ \
  -- \
-tr "/home/jiman/data/imdb_face_vgg_all/imdb_224_all-tr*" \
-cv "/home/jiman/data/imdb_face_vgg_all/imdb_224_all-cv*" \
--learning-rate 0.00005 \
--num-epochs 10 \
--dropout 0.0 \
--lam 0.0 \
--model-file /home/jiman/mljob/face_age_rec_20180210_125146/checkpoint.03.hdf5
#--trainable -5

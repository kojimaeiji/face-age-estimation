DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=face_age_vgg_face_$DATE
#export GCS_JOB_DIR=/Users/saboten/mljob
export GCS_JOB_DIR=/home/jiman/mljob/$JOB_NAME
echo $GCS_JOB_DIR
#export PYTHONPATH="/home/jiman/workspace/face-age-estimation/trainer_vgg_face/keras_vgg_face:$PYTHONPATH"
#rm -rf /Users/saboten/mljob/*
#rm -rf /home/jiman/mljob/*

gcloud ml-engine local train \
  --job-dir $GCS_JOB_DIR \
--module-name trainer_vgg_face.task \
--package-path trainer_vgg_face \
  -- \
-tr "/home/jiman/data/imdb_face_vgg_all/imdb_224_all-tr*" \
-cv "/home/jiman/data/imdb_face_vgg_all/imdb_224_all-cv*" \
--learning-rate 0.001 \
--num-epochs 10 \
--dropout 0.5 \
--lam 0.0 \
#--model-file /home/jiman/mljob/face_age_vgg_face_20180212_200225/checkpoint.01.hdf5
#--trainable -5

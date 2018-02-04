DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=face_age_rec_$DATE
#export GCS_JOB_DIR=/Users/saboten/mljob
export GCS_JOB_DIR=/home/jiman/mljob
echo $GCS_JOB_DIR
#rm -rf /Users/saboten/mljob/*
rm -rf /home/jiman/mljob/*

gcloud ml-engine local train \
  --job-dir $GCS_JOB_DIR \
--module-name trainer.task \
--package-path trainer/ \
  -- \
-tr "/home/jiman/data/wiki_face_vgg_small/wiki_3_224_64-tr*" \
-cv "/home/jiman/data/wiki_face_vgg_small/wiki_3_224_64-cv*" \
--learning-rate 0.0001 \
--num-epochs 20 \
--dropout 0.0 \
--lam 0.0
#--trainable -5

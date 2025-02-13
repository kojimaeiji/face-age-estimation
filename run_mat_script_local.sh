DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=face_age_vgg16_$DATE
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
-tr "/home/jiman/data/wiki_process_10000.mat" \
--learning-rate 0.0001 \
--num-epochs 50 \
--dropout 0.5 \
--lam 0.0


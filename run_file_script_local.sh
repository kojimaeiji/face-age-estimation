DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=face_age_rec_$DATE
#export GCS_JOB_DIR=/Users/saboten/mljob
export GCS_JOB_DIR=/home/jiman/mljob
echo $GCS_JOB_DIR
#rm -rf /Users/saboten/mljob/*
rm -rf /home/jiman/mljob/*

gcloud ml-engine local train \
  --job-dir $GCS_JOB_DIR \
--module-name trainer_face_recog.task \
--package-path trainer_face_recog/ \
  -- \
-tr "/home/jiman/data/wiki_face_rec/wiki_3_96_all-tr*" \
-cv "/home/jiman/data/wiki_face_rec/wiki_3_96_all-cv*" \
--learning-rate 0.1 \
--num-epochs 20 \
--dropout 0.0 \
--lam 0.0 \
--trainable -5

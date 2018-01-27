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
-tr "gs://kceproject-1113-ml/wiki_face_rec/*" \
-cv "gs://kceproject-1113-ml/wiki_face_rec/*" \
--learning-rate 0.001 \
--num-epochs 50 \
--dropout 0.5 \
--lam 0.0


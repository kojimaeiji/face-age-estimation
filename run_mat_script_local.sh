DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=face_age_vgg16_$DATE
#export GCS_JOB_DIR=/Users/saboten/mljob
export GCS_JOB_DIR=/home/jiman/mljob
echo $GCS_JOB_DIR
#rm -rf /Users/saboten/mljob
rm -rf /home/jiman/mljob/*

gcloud ml-engine local train \
  --job-dir $GCS_JOB_DIR \
--module-name trainer.task \
--package-path trainer/ \
  -- \
 -tr "gs://kceproject-1113-ml/face-age-estimation/wiki_process_60_8092-tr*" \
 -cv "gs://kceproject-1113-ml/face-age-estimation/wiki_process_60_8092-cv*" \
--learning-rate 0.0001 \
--num-epochs 20 \
--dropout 0.0 \
--lam 0.0


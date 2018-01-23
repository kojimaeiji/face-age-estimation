DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=face_age_vgg16_$DATE
export GCS_JOB_DIR=/Users/saboten/mljob
echo $GCS_JOB_DIR
rm -rf /Users/saboten/mljob

gcloud ml-engine local train \
  --job-dir $GCS_JOB_DIR \
--module-name trainer.task \
--package-path trainer/ \
  -- \
 -tr "/Users/saboten/data/wiki_process_60_128*" \
 -cv "/Users/saboten/data/wiki_process_60_128*" \
--learning-rate 0.0001 \
--num-epochs 2 \
--dropout 0.0 \
--lam 0.0


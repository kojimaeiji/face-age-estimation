DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=face_age_vgg16_$DATE
export GCS_JOB_DIR=gs://kceproject-1113-ml/ml-job/$JOB_NAME
echo $GCS_JOB_DIR

gcloud ml-engine jobs submit training $JOB_NAME \
  --stream-logs \
  --runtime-version 1.4 \
  --job-dir $GCS_JOB_DIR \
  --module-name trainer.task \
  --package-path trainer/ \
  --region us-central1 \
  --scale-tier basic-gpu \
  -- \
  -tr "gs://kceproject-1113-ml/face-age-estimation/wiki_process_60_8092-tr*" \
  -cv "gs://kceproject-1113-ml/face-age-estimation/wiki_process_60_8092-cv*" \
  --lam 0.0 \
  --dropout 0.0 \
  --num-epochs 20 \
  --learning-rate 0.0001

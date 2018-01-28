DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=face_age_$DATE
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
  -tr "gs://kceproject-1113-ml/ordinal-face/wiki_processed_all.mat" \
  --lam 0.0 \
  --dropout 0.5 \
  --num-epochs 50 \
  --learning-rate 0.001 \
  --trainable -5

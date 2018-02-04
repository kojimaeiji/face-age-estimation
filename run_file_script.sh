DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=face_age_recog_$DATE
export GCS_JOB_DIR=gs://kceproject-1113-ml/ml-job/$JOB_NAME
echo $GCS_JOB_DIR

gcloud ml-engine jobs submit training $JOB_NAME \
  --stream-logs \
  --runtime-version 1.4 \
  --job-dir $GCS_JOB_DIR \
  --module-name trainer_face_recog.task \
  --package-path trainer_face_recog/ \
  --region us-central1 \
  --config trainer_mat/cloudml-gpu.yaml \
  -- \
  -tr "gs://kceproject-1113-ml/wiki_face_rec/*" \
  -cv "gs://kceproject-1113-ml/wiki_face_rec/*" \
  --lam 0.0 \
  --dropout 0.0 \
  --num-epochs 20 \
  --learning-rate 1.0 \
  --trainable -5

DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=face_age_vgg_$DATE
export GCS_JOB_DIR=gs://kceproject-1113-ml/ml-job/$JOB_NAME
echo $GCS_JOB_DIR

gcloud ml-engine jobs submit training $JOB_NAME \
  --stream-logs \
  --runtime-version 1.4 \
  --job-dir $GCS_JOB_DIR \
  --module-name trainer.task \
  --package-path trainer/ \
  --region us-central1 \
  --config trainer/cloudml-gpu.yaml \
  -- \
  -tr "gs://kceproject-1113-ml/imdb_face_vgg_all/*" \
  -cv "gs://kceproject-1113-ml/imdb_face_vgg_all/*" \
  --lam 0.0 \
  --dropout 0.0 \
  --num-epochs 40 \
  --learning-rate 0.0001
#  --trainable -5

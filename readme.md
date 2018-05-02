```
gcloud ml-engine jobs submit training ec_predict_$(date -u +%y%m%d_%H%M%S) --region=us-central1 --module-name=trainer.task --job-dir=gs://ml_prediction_budickda/model_trained_$(date -u +%y%m%d_%H%M%S) --package-path=${PWD}/trainer --staging-bucket=gs://ml_prediction_budickda --scale-tier=BASIC --runtime-version=1.6 -- --train_data_paths="gs://asl_exploration1/data/secondrun/train.csv" --eval_data_paths="gs://asl_exploration1/data/secondrun/eval.csv" --output_dir="gs://ml_prediction_budickda/model_trained_$(date -u +%y%m%d_%H%M%S)" --model="rnn" --train_steps=10000 --sequence_length=50
```

```
rm -rf output && python -m trainer.task --train_data_paths=./data/firstrun/train.csv --eval_data_paths=./data/firstrun/eval.csv --job-dir=./tmp --model=rnn --train_steps=10 --sequence_length=8 --output_dir=./output
```

```
gcloud ml-engine jobs submit training ec_predict_hypertuning_$(date -u +%y%m%d_%H%M%S) --config=hyperparam.yaml --region=us-central1 --module-name=trainer.task --job-dir=gs://ml_prediction_budickda/model_trained_$(date -u +%y%m%d_%H%M%S) --package-path=${PWD}/trainer --staging-bucket=gs://ml_prediction_budickda --scale-tier=BASIC --runtime-version=1.6 -- --train_data_paths="gs://asl_exploration1/data/firstrun/train.csv" --eval_data_paths="gs://asl_exploration1/data/firstrun/eval.csv" --output_dir="gs://ml_prediction_budickda/model_trained_$(date -u +%y%m%d_%H%M%S)" --model="rnn" --sequence_length=8
```
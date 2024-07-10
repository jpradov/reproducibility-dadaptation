#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <experiment_type>"
  exit 1
fi

experiment_type=$1

case $experiment_type in
  "logistic-regression")
    echo "Running logistic regression experiment..."
    python regression_experiments/reproduce_log_reg.py --filepath "data/regression_experiments/raw/libsvm/" --filename iris.scale --savename 'iris'
    ;;
  "image-classification")
    echo "Running image classification experiment..."
    bash vision_experiments/run_cifar10.sh
    ;;
  "nlp")
    echo "Running NLP experiment..."
    cd nlp_experiments
    bash adam_iwslt14_train.sh 1
    ;;
  *)
    echo "Invalid experiment type. Supported types: logistic-regression, image-classification, nlp"
    exit 1
    ;;
esac


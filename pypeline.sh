#!/bin/bash

echo "-> data_creation.py"
python3 data_creation.py

echo "-> model_preprocessing.py"
python3 model_preprocessing.py

echo "-> model_preparation.py"
python3 model_preparation.py

echo "-> model_testing.py"
python3 model_testing.py
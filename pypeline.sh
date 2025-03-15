#!/bin/bash

echo "-> data_creation.py"
python data_creation.py

echo "-> model_preprocessing.py"
python model_preprocessing.py

echo "-> model_preparation.py"
python model_preparation.py

echo "-> model_testing.py"
python model_testing.py
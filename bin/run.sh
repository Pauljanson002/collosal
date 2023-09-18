#! /bin/bash
python main.py lr=0.001 epochs=10000 method=dds optimizer=sgd  wandb=online workspace=flamingo_test_dds lambda_guidance=1 lambda_rgb=0 text="peacock in water"
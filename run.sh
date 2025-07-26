#!/bin/bash
pip install -r requirments.txt
python MNIST/main.py --config MNIST/config.json
python Adult/main.py --config Adult/config.json

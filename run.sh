#!/bin/bash
pip install -r requirements.txt
python MNIST/main.py --config MNIST/config.json
python Adult/main.py --config Adult/config.json

#!/bin/bash
# modify test json
python modify_submit_jsonl.py "$1" 
# Inference
python Inference.py "$2"  

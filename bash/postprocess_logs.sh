#!/bin/bash

cat tmp/experiment_logs/test1log.txt | tr -d "'"  > tmp/experiment_logs/temp.txt
cat tmp/experiment_logs/temp.txt | tr -d "["  > tmp/experiment_logs/temp2.txt
cat tmp/experiment_logs/temp2.txt | tr -d "]"  > tmp/experiment_logs/log.txt
rm tmp/experiment_logs/temp.txt
rm tmp/experiment_logs/temp2.txt



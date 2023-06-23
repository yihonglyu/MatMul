#!/bin/bash

set -e

for ((i=1; i<=100; i++))
do
  taskset -c 0-30:2 python pt.py
  #taskset -c 1-31:2 python pt.py
  #python pt.py
  #taskset -c 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31 python ort.py
  #taskset -c 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 python ort.py
  #taskset -c 0-30:2 python ort.py
  #taskset -c 1-31:2 python ort.py
  #python ort.py
  #python ort_wo_iobinding.py
done

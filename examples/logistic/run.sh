#!/bin/bash

for ID in {1..20}
do
    python3 main.py synth $ID
    python3 main.py ds1 $ID
    python3 main.py phish $ID
done

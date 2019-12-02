#!/bin/bash

for i in {0..15}
do
	sbatch -p 1080ti-long --gres=gpu:2 --mem=200000 parser_${i}.sh
done

for i in {16..30}
do
        sbatch -p titanx-long --gres=gpu:2 --mem=200000 parser_${i}.sh
done

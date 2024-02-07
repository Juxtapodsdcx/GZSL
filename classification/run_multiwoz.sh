#!/bin/bash
for seed in {0..9}; do
  echo "--------------------------------------------------------------------------------------------------------------Running code with seed $seed"
  python train.py dataset=multiwoz experiment.name=/result/multiwoz experiment.seed=$seed model.prompt_type=hardprompt experiment.cuda=0 >> multiwoz.log
  python evaluate.py dataset=multiwoz experiment.name=/result/multiwoz experiment.seed=$seed model.prompt_type=hardprompt experiment.cuda=0 >> multiwoz.log
  echo "--------------------------------------------------------------------------------------------------------------Code execution with seed $seed completed"
done
#!/bin/bash
for seed in {0..9}; do
  echo "--------------------------------------------------------------------------------------------------------------Running code with seed $seed"
  python train.py dataset=bank experiment.name=/result/bank experiment.seed=$seed model.prompt_type=hardprompt experiment.cuda=0 >> bank.log
  python evaluate.py dataset=bank experiment.name=/result/bank experiment.seed=$seed model.prompt_type=hardprompt experiment.cuda=0 >> bank.log
  echo "--------------------------------------------------------------------------------------------------------------Code execution with seed $seed completed"
done
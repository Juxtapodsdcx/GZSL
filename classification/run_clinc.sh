#!/bin/bash
for seed in {0..9}; do
  echo "--------------------------------------------------------------------------------------------------------------Running code with seed $seed"
  python train.py dataset=clinc experiment.name=/result/clinc experiment.seed=$seed model.prompt_type=hardprompt experiment.cuda=0 >> clinc.log
  python evaluate.py dataset=clinc experiment.name=/result/clinc experiment.seed=$seed model.prompt_type=hardprompt experiment.cuda=0 >> clinc.log
  echo "--------------------------------------------------------------------------------------------------------------Code execution with seed $seed completed"
done

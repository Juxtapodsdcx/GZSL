#!/bin/bash
for seed in {0..9}; do
  echo "--------------------------------------------------------------------------------------------------------------Running code with seed $seed"
  python train.py dataset=atis experiment.name=/result/atis experiment.seed=$seed model.prompt_type=hardprompt experiment.cuda=0 >> atis.log
  python evaluate.py dataset=atis experiment.name=/result/atis experiment.seed=$seed model.prompt_type=hardprompt experiment.cuda=0 >> atis.log
  echo "--------------------------------------------------------------------------------------------------------------Code execution with seed $seed completed"
done
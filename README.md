# LoDen: Making Every Client in Federated Learning a Defender
Against the Poisoning Membership Inference Attacks

These files provides the core functionality in the experiment setting

1.  aggregator.py \- The aggregator collects and calculate the gradients from participants
2.  constants.py \- All hyper-parameters
3.  data\_reader.py \- The module loading data from data set files and distribute them to participants
4.  models.py \- The participants, global model, and local attackers

* * *

These files are the runnable experiment files:

1.  Blackbox LoDen defense file: loden_lable_defence.py
2.  Whitebox LoDen defense file: loden_vector_defence.py
3.  MIA baseline file: loden_MIA.py
* * *


This directory contains some example results:

1.  The indistinguishbility views for adversary with/without LoDen : example_indistinguishible
* * *


To run experiment with certain dataset, you can find the coresponding constant file in baseline_constants directory.

Quick note for set up:

The dataset_purchase.tgz need to be extracted as 'dataset_purchase' before running

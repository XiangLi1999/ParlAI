# Diverse Responses and Evaluation  

**Deep Learning for Automated Discourse: Assignment 4**  

**Alexandra DeLucia, Lisa Li, Aaron Mueller**  
---

## Guide to Repository
### Installation
```bash
git clone -b amueller_bidi git@github.com:XiangLi1999/ParlAI.git
cd ParlAI
python setup.py develop
```

- changes
- description of what we did (created new Agent, modified _generate, made _post_prob and rerank methods)

## Trained Models
- model parameters
- perplexities for forward, backward, and finetuned (all around 20~25), 

## Mechanical Turk Evaluation
Run the Mechanical Turk evaluation with the following command:

```bash
./mturk_evaluation.sh
```
This script is setup to work on the CLSP grid. You may have to sign into your Heroku account (you can create one [here](https://heroku.com/)). More detailed instructions for all the needed accounts are available [here](https://parl.ai/docs/tutorial_mturk.html#running-a-task).

### Results
- results from SANDBOX evals
- future work
- improvements
- comments

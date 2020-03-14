# Diverse Responses and Evaluation  

**Deep Learning for Automated Discourse: Assignment 4**  

**Alexandra DeLucia, Lisa Li, Aaron Mueller**  

## Guide to Repository
Project instructions on the course website [here](https://dialog-systems-class.github.io/assignment4.html).

To implement the data preprocessing from [DialoGPT](https://arxiv.org/pdf/1911.00536.pdf) and the Maximum Mutual Information (MMI) training objective from [A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/abs/1510.03055) we did the following:

1. Created `filter.py` to filter the OpenSubtitles dataset.
2. Trained two TransformerGenerator models, one "forward" model (i.e. P(T|S)) and one "backward" model (i.e. P(S|T)) on the filtered OpenSubtitles dataset. 
3. Finetuned the forward model on the DailyDialog dataset.
4. Implemented a new agent, `TransformerGeneratorMMIAgent`, for the MMI-bidi objective decoding. More details in later sections.
5. Created a new Mechanical Turk task to evaluate our trained TransformerGeneratorMMIAgent.

### Implementation of MMI-bidi Decoding  
We created a new ParlAI agent, `TransformerGeneratorMMIAgent` in `parlai/agents/transformer/transformer.py`. This agent has two `TransformerGenerator` models, one "forward" model and one "backward" model. 

- description of what we did (created new Agent, modified _generate, made _post_prob and rerank methods)


### Installation
```bash
git clone -b amueller_bidi git@github.com:XiangLi1999/ParlAI.git
cd ParlAI
python setup.py develop
```

## Trained Models
- model parameters
- perplexities for forward, backward, and finetuned (all around 20~25)

The training scripts for the forward, backward, and finetuned-forward models are `train_forward.sh`, `train_backward.sh`, and `train_finetune.sh`, respectively. 

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

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

To do this, we had to load two models at once. Thus, we created a new agent, `TransformerGeneratorMMIAgent`, which not only has a `self.model` attribute, but also a `self.model_backwards` attribute. We created methods to deal with this new model. As we are not training using this backward model, we only needed to modify the `_generate` method to include reranking by the backward model. This entailed creating a new method, `_post_prob`, which takes the generated output from the forward model and the source sentence and calculates the posterior probability using the backward model. We do this for all n-best hypotheses, reranking them according to the backward model's posteriors before generating the final best hypothesis.


### Installation
```bash
git clone -b amueller_bidi git@github.com:XiangLi1999/ParlAI.git
cd ParlAI
python setup.py develop
```

To obtain our filtered dataset, you must first download the OpenSubtitles2018 dataset, either from the website or using ParlAI's downloader by specifying the `-t opensubtitles` task. Then, run `filter.py` on the text files to filter according to DialoGPT's filtering methods (as well as getting rid of non-question-answer pairs). Then, you should be able to run our training code.

## Trained Models
The training scripts for the forward, backward, and finetuned-forward models are `train_forward.sh`, `train_backward.sh`, and `train_finetune.sh`, respectively. Run these from the base directory. You may have to modify the paths to work for your particular configuration.

Note that for `train_forward.sh` and `train_backward.sh`, we use the same hyperparameters. For `train_finetune.sh`, we halve the max training time and reduce the learning rate by a factor of 10. We finetune on DailyDialog. 

The training and validation perplexities for the forward, backward, and finetuned models all jumped around 20 to 23 once they converged, depending on the specific iteration they ended on.

## Mechanical Turk Evaluation  
Run the Mechanical Turk evaluation with the following command:

```bash
./mturk_evaluation.sh
```
This script is setup to work on the CLSP grid. You may have to sign into your Heroku account (you can create one [here](https://heroku.com/)). More detailed instructions for all the needed accounts are available [here](https://parl.ai/docs/tutorial_mturk.html#running-a-task).

To create our Mechanical Turk task we modified the `parlai/mturk/tasks/model_evaluator` task. We made the following changes:
* changed the model task to `#DailyDialog`
* loaded our own agent (GeneratorMMI model trained on filtered OpenSubtitles and finetuned on DailyDialog)
* increased the session length so the Turker rates multiple responses in a session
* altered the Heroku server start-up to push to a pre-created app

### Qualitative Evaluation
TODO

### Comments and Future Work

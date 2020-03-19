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

To obtain our filtered dataset, you must first download the OpenSubtitles2018 dataset, either from the website or using ParlAI's downloader by specifying the `-t opensubtitles` task. Then, run `filter.py` on the text files to filter according to DialoGPT's filtering methods (as well as getting rid of non-question-answer pairs). Then, you should be able to run our training code, detailed below.

If you want to use the Gradescope submission, then just clone ParlAI as usual, copy these files to that directory---overwriting ParlAI's---and then these instructions should still apply.

## Trained Models
The training scripts for the forward, backward, and finetuned-forward models are `train_forward.sh`, `train_backward.sh`, and `train_finetune.sh`, respectively. Run these from the base directory. You may have to modify the paths to work for your particular machine; these are made to work on the CLSP grid.

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
We used our MTurk evaluation setup to compare the performance of the `forward_finetune` model with and without the MMI-bidi objective during inference. We first report our average human-given numbers for each model, then we discuss some trends that may escape quantification.

Our MMI-bidi model scored an average of 6.1 over the course of 10 session, each session containing 5 messages. The model with the unmodified objective scored 5.6 in comparison. This numeric difference is not very large, but we find that the performance varies greatly between models. We kept track of interestingness, adequacy, and fluency, and we found that the MMI-bidi model was interesting far more often (it often seemed to have its own personality), but also adequate slightly less often. Both were essentially always fluent, unless you count `_unk_` as non-fluent; regardless, fluency is comparable across the objectives.

Nonetheless, both models struggle after about 5 messages in one session. They enter a loop where, no matter what you say, they repeat the same response.The unmodified objective tends toward "I don't know" or "What are you doing?" In contrast, the MMI-bidi model always has a different response depending on what the input was, but after 5 messages, it will keep repeating that response. We found everything from "I need to clean my room", "I'm going to kill you", and "Yes, I like that" were repeated. Since both objectives lead to this behavior, this suggests some sort of issue independent of the objective; perhaps the architecture or hyperparameters are to blame here.

Another issue we noticed is that the model tends to output a large number of `_unk_`s. While we were typically able to infer what was meant given the context, this is not ideal behavior when deploying a dialogue system in general. We believe that this behavior is due to the domain mismatch between OpenSubtitles and DailyDialog: we build a dictionary using the former, which we use while fine-tuning on the latter. This results in many `_unk_` tokens in the target-side during fine-tuning, which the model learns to replicate.

### Comments and Future Work
Our qualitative evaluations suggest that the MMI-bidi generator is indeed prone to giving more interesting and situational responses, but it can also give more irrelevant responses as well. Meanwhile, the unmodified generator always tends to give safe and uninteresting answers like "I don't know" or "What?" Otherwise, these models have the same issues---most notably, repeating themselves after a certain number of messages within one conversation.

One potential fix to this would be re-initializing the hidden state after every response (or every nth response, where n is a hyperparameter), such that we treat all inputs as independent of each other. While this is not intuitively ideal, it may result in better performance for longer conversations and better question answering.

To fix the `_unk_` problem, we could (1) use a larger dictionary and/or (2) build the dictionary on the combination of the most frequent tokens in the training and fine-tuning sets. In other words, rather than building a dictionary from the training set alone, we share vocabularies and interpolate the frequencies of certain tokens in both, using this shared frequency count to obtain our vocabulary.

A bigger problem is to fix the (seemingly inherent) irrelevance issue of diversity-promoting objectives. One could impose additional constraints on the output to match the semantics of the input, for example, and this could be done using some sort of distance measure on the word and/or sentence embeddings. Alternatively, one could use the personality-promoting features of the [model proposed here](https://www.ijcai.org/Proceedings/2018/0595.pdf). 

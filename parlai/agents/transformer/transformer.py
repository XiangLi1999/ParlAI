#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Transformer Agents.
"""
from parlai.core.agents import Agent
from parlai.utils.torch import padded_3d
from parlai.core.torch_classifier_agent import TorchClassifierAgent
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.utils.distributed import is_distributed, sync_parameters

# Special imports for TransformerGeneratorMMIAgent
from copy import deepcopy
import torch.nn.functional as F
from parlai.core.agents import create_agent, create_agent_from_opt_file

from .modules import (
    TransformerMemNetModel,
    TransformerGeneratorModel,
    TransformerLinearWrapper,
)

import torch


def add_common_cmdline_args(argparser):
    """
    Add common command line args.
    """
    argparser.add_argument(
        '-esz',
        '--embedding-size',
        type=int,
        default=300,
        help='Size of all embedding layers',
    )
    argparser.add_argument('-nl', '--n-layers', type=int, default=2)
    argparser.add_argument(
        '-hid',
        '--ffn-size',
        type=int,
        default=300,
        help='Hidden size of the FFN layers',
    )
    argparser.add_argument(
        '--dropout', type=float, default=0.0, help='Dropout used in Vaswani 2017.'
    )
    argparser.add_argument(
        '--attention-dropout',
        type=float,
        default=0.0,
        help='Dropout used after attention softmax.',
    )
    argparser.add_argument(
        '--relu-dropout',
        type=float,
        default=0.0,
        help='Dropout used after ReLU. From tensor2tensor.',
    )
    argparser.add_argument(
        '--n-heads', type=int, default=2, help='Number of multihead attention heads'
    )
    argparser.add_argument('--learn-positional-embeddings', type='bool', default=False)
    argparser.add_argument('--embeddings-scale', type='bool', default=True)
    argparser.add_argument(
        '--n-positions',
        type=int,
        default=None,
        hidden=True,
        help='Number of positional embeddings to learn. Defaults '
        'to truncate or 1024 if not provided.',
    )
    argparser.add_argument(
        '--n-segments',
        type=int,
        default=0,
        help='The number of segments that support the model. '
        'If zero no segment and no langs_embedding.',
    )
    argparser.add_argument(
        '--variant',
        choices={'aiayn', 'xlm', 'prelayernorm'},
        default='aiayn',
        help='Chooses locations of layer norms, etc. prelayernorm '
        'is used to match some fairseq models',
        recommended='xlm',
    )
    argparser.add_argument(
        '--activation',
        choices={'relu', 'gelu'},
        default='relu',
        help='Nonlinear activation to use. AIAYN uses relu, but '
        'more recent papers prefer gelu.',
        recommended='gelu',
    )
    argparser.add_argument(
        '--output-scaling',
        type=float,
        default=1.0,
        help='scale the output of every transformer by this quantity.',
    )
    argparser.add_argument(
        '--share-word-embeddings',
        type='bool',
        default=True,
        help='Share word embeddings table for candidate and context'
        'in the memory network',
    )
    argparser.add_argument(
        '-nel',
        '--n-encoder-layers',
        type=int,
        default=-1,
        help='This will overide the n-layers for asymmetrical transformers',
    )
    argparser.add_argument(
        '-ndl',
        '--n-decoder-layers',
        type=int,
        default=-1,
        help='This will overide the n-layers for asymmetrical transformers',
    )


class Transformer(Agent):
    """
    Placeholder Transformer Agent.

    Placeholder class, which just throws an error telling the user to specify whether
    they want the ranker or the generator.
    """

    def __init__(self, opt, shared=None):
        raise RuntimeError(
            "`--model transformer` is not a valid choice. Please select either "
            "`--model transformer/ranker` or `--model transformer/generator"
        )


class TransformerRankerAgent(TorchRankerAgent):
    """
    Transformer Ranker Agent.

    Implementation of a TorchRankerAgent, where the model is a Transformer
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        super(TransformerRankerAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Transformer Arguments')
        add_common_cmdline_args(agent)
        # memory and knowledge arguments
        agent.add_argument(
            '--use-memories',
            type='bool',
            default=False,
            help='use memories: must implement the function '
            '`_vectorize_memories` to use this',
        )
        agent.add_argument(
            '--wrap-memory-encoder',
            type='bool',
            default=False,
            help='wrap memory encoder with MLP',
        )
        agent.add_argument(
            '--memory-attention',
            type=str,
            default='sqrt',
            choices=['cosine', 'dot', 'sqrt'],
            help='similarity for basic attention mechanism '
            'when using transformer to encode memories',
        )
        # model specific arguments
        agent.add_argument('--normalize-sent-emb', type='bool', default=False)
        agent.add_argument('--share-encoders', type='bool', default=True)
        argparser.add_argument(
            '--share-word-embeddings',
            type='bool',
            default=True,
            help='Share word embeddings table for candidate and context'
            'in the memory network',
        )
        agent.add_argument(
            '--learn-embeddings', type='bool', default=True, help='learn embeddings'
        )
        agent.add_argument(
            '--data-parallel',
            type='bool',
            default=False,
            help='use model in data parallel, requires ' 'multiple gpus',
        )
        agent.add_argument(
            '--reduction-type',
            type=str,
            default='mean',
            choices=['first', 'max', 'mean'],
            help='Type of reduction at the end of transformer',
        )

        argparser.set_defaults(learningrate=0.0001, optimizer='adamax', truncate=1024)
        cls.dictionary_class().add_cmdline_args(argparser)

        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.data_parallel = opt.get('data_parallel') and self.use_cuda
        if self.data_parallel:
            from parlai.utils.distributed import is_distributed

            if is_distributed():
                raise ValueError('Cannot combine --data-parallel and distributed mode')
            self.model = torch.nn.DataParallel(self.model)

    def _score(self, output, cands):
        if cands.dim() == 2:
            return torch.matmul(output, cands.t())
        elif cands.dim() == 3:
            return torch.bmm(output.unsqueeze(1), cands.transpose(1, 2)).squeeze(1)
        else:
            raise RuntimeError(
                'Unexpected candidate dimensions {}' ''.format(cands.dim())
            )

    def build_model(self, states=None):
        """
        Build and return model.
        """
        model = TransformerMemNetModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(model.embeddings.weight, self.opt['embedding_type'])
        return model

    def batchify(self, obs_batch, sort=False):
        """
        Override so that we can add memories to the Batch object.
        """
        batch = super().batchify(obs_batch, sort)
        if self.opt['use_memories']:
            valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if self.is_valid(ex)]
            valid_inds, exs = zip(*valid_obs)
            mems = None
            if any('memory_vecs' in ex for ex in exs):
                mems = [ex.get('memory_vecs', None) for ex in exs]
            batch.memory_vecs = mems
        return batch

    def _vectorize_memories(self, obs):
        # TODO: move this to Torch Ranker Agent
        raise NotImplementedError(
            'Abstract class: user must implement this function to use memories'
        )

    def vectorize(self, *args, **kwargs):
        """
        Override to include vectorization of memories.
        """
        kwargs['add_start'] = False
        kwargs['add_end'] = False
        obs = super().vectorize(*args, **kwargs)
        if self.opt['use_memories']:
            obs = self._vectorize_memories(obs)
        return obs

    def encode_candidates(self, padded_cands):
        """
        Encode candidates.
        """
        _, cands = self.model(xs=None, mems=None, cands=padded_cands)

        return cands

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        """
        Score candidates.
        """
        # convoluted check that not all memories are empty
        if (
            self.opt['use_memories']
            and batch.memory_vecs is not None
            and sum(len(m) for m in batch.memory_vecs)
        ):
            mems = padded_3d(
                batch.memory_vecs, use_cuda=self.use_cuda, pad_idx=self.NULL_IDX
            )
        else:
            mems = None

        if cand_encs is not None:
            # we pre-encoded the candidates, do not re-encode here
            cand_vecs = None

        context_h, cands_h = self.model(xs=batch.text_vec, mems=mems, cands=cand_vecs)

        if cand_encs is not None:
            cands_h = cand_encs
        scores = self._score(context_h, cands_h)

        return scores


class TransformerGeneratorAgent(TorchGeneratorAgent):
    """
    TransformerGeneratorAgent.

    Implementation of TorchGeneratorAgent, where the model is a Transformer
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        agent = argparser.add_argument_group('Transformer Arguments')
        add_common_cmdline_args(agent)
        cls.dictionary_class().add_cmdline_args(argparser)

        super(TransformerGeneratorAgent, cls).add_cmdline_args(argparser)
        return agent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        model = TransformerGeneratorModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model


class TransformerGeneratorMMIAgent(TorchGeneratorAgent):
    """
    Modified by Alexandra DeLucia and Aaron Mueller.
    """
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.model_backwards = None
        self.max_output_length = None

        self.model_backwards = self.build_model_backwards()
        if self.model_backwards is None:
            raise AttributeError(
                'build_model() and build_criterion() need to return the model or criterion'
            )
        if self.use_cuda:
            self.model_backwards.cuda()

        sync_parameters(self.model_backwards)
        print("Total backwards parameters: {}".format(self._total_parameters()))
        print("Trainable backwards parameters:  {}".format(self._trainable_parameters()))

        if self.fp16:
            self.model_backwards = self.model_backwards.half()

        if shared is None and is_distributed():
            self.model_backwards = torch.nn.parallel.DistributedDataParallel(
                self.model_backwards, device_ids=[self.opt['gpu']], broadcast_buffers=False
            )


    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        agent = argparser.add_argument_group('Transformer Generate MMI Arguments')
        agent.add_argument("--model-file-backward", help="Location of backwards model")
        add_common_cmdline_args(agent)
        cls.dictionary_class().add_cmdline_args(argparser)

        super(TransformerGeneratorMMIAgent, cls).add_cmdline_args(argparser)
        return agent

    def build_model_backwards(self, states=None):
        # Load backwards model
        # Mess with opt['override']?
        print("Loading backwards model...")
        backwards_opt = deepcopy(self.opt)
        if self.opt.get('model_file_backward'):
            backwards_opt['model_file'] = self.opt['model_file_backward']
            backwards_opt['override']['model_file'] = self.opt['model_file_backward']
            backwards_opt['override']['no_cuda'] = self.opt['no_cuda']
        else:
            raise ValueError('model_file_backward option must have a value.')
        model_backwards = TransformerGeneratorModel(backwards_opt, self.dict)
        return model_backwards

    def build_model(self, states=None):
        """
        TODO: modify to give access to forward and backward models
        Build and return model.
        """
        # Load forward model
        print("Loading forward model...")
        model = TransformerGeneratorModel(self.opt, self.dict)
        
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model

    def _compute_posterior(self, inputs, targets):
        """

        inputs: S in P(S|T). Probability of the input given the output. n_best_beam_preds_scores from _generate()
        targets: T in P(S|T). Output generated by the forward model.
        lengths:
        """
        batch_size = len(inputs)
        beam_size = len(inputs[0])
        # targets = batch.text_vec
        # print(f"self.model_backwards: {self.model_backwards}")
        # print(f"self._encoder_input(targets): {self._encoder_input(targets)}")
        # print(f"Input: {inputs}")
        model_backwards = self.model_backwards
        if isinstance(model_backwards, torch.nn.parallel.DistributedDataParallel):
            model_backwards = self.model_backwards.module

        full_lst = []
        for i in range(batch_size):
            beam_lst = []
            for j in range(beam_size):
                encoder_states = model_backwards.encoder(inputs[i][j][0].unsqueeze(0))

                # teacher forcing
                ys = self._encoder_input(targets)[0]

                bsz = ys.size(0)
                seqlen = ys.size(1)
                inputs_ = ys.narrow(1, 0, seqlen - 1)
                inputs_ = torch.cat([model_backwards.START.detach().expand(bsz, 1), \
                        inputs_], 1)
                latent, _ = model_backwards.decoder(inputs_, encoder_states)
                logits = model_backwards.output(latent)
                _, preds = logits.max(dim=2)

                score_view = logits.view(-1, logits.size(-1))
                loss = self.criterion(score_view, ys.view(-1))
                loss = loss.view(logits.shape[:-1]).sum(dim=1)

                beam_lst.append(loss)
            full_lst.append(beam_lst)

        return full_lst


    def rerank(self, inputs, score_back, lambda_):
        batch_size = len(inputs)
        beam_size = len(inputs[0])
        batch_lst = []

        for i in range(batch_size):
            final_lst = []
            for j in range(beam_size):
                text, score_fore = inputs[i][j]

                temp_score = (lambda_ * (-score_fore) + (1-lambda_) * score_back[i][j]).item()

                final_lst.append((text, temp_score))
            final_lst.sort(key=lambda x: x[1])
            batch_lst.append(final_lst)

        return batch_lst


    def _generate(self, batch, beam_size, max_ts):
        """
        Overwritten from parent to implement decoding with MMI-bidi objective.

        Generate an output with beam search.

        Depending on the options, this may perform greedy/topk/nucleus generation.

        :param Batch batch:
            Batch structure with input and labels
        :param int beam_size:
            Size of each beam during the search
        :param int max_ts:
            the maximum length of the decoded sequence

        :return:
            tuple (beam_pred_scores, n_best_pred_scores, beams)

            - beam_preds_scores: list of (prediction, score) pairs for each sample in
              Batch
            - n_best_preds_scores: list of n_best list of tuples (prediction, score)
              for each sample from Batch
            - beams :list of Beam instances defined in Beam class, can be used for any
              following postprocessing, e.g. dot logging.
        """
        print("In TransformerGeneratorMMIAgent self._generate()")
        print(f"\tInput to method:\n\t\tBatch:{batch}\n\t\tBeam size: {beam_size}\n\t\tMax ts:{max_ts}")
        model = self.model
        model_backwards = self.model_backwards
        self.max_output_length = max_ts
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        if isinstance(model_backwards, torch.nn.parallel.DistributedDataParallel):
            model_backwards = self.model_backwards.module
        encoder_states = model.encoder(*self._encoder_input(batch))  # Initialization?
        if batch.text_vec is not None:
            dev = batch.text_vec.device
        else:
            dev = batch.label_vec.device

        bsz = (
            len(batch.text_lengths)
            if batch.text_lengths is not None
            else len(batch.image)
        )
        if batch.text_vec is not None:
            batchsize = batch.text_vec.size(0)
            beams = [
                self._treesearch_factory(dev).set_context(
                    self._get_context(batch, batch_idx)
                )
                for batch_idx in range(batchsize)
            ]
        else:
            beams = [self._treesearch_factory(dev) for _ in range(bsz)]

        # repeat encoder outputs and decoder inputs
        decoder_input = (
            torch.LongTensor([self.START_IDX]).expand(bsz * beam_size, 1).to(dev)
        )

        inds = torch.arange(bsz).to(dev).unsqueeze(1).repeat(1, beam_size).view(-1)
        encoder_states = model.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        for _ts in range(max_ts):
            if all((b.is_done() for b in beams)):
                # exit early if possible
                break

            score, incr_state = model.decoder(decoder_input, encoder_states, incr_state)
            # only need the final hidden state to make the word prediction
            score = score[:, -1:, :]
            score = model.output(score)
            # score contains softmax scores for bsz * beam_size samples
            score = score.view(bsz, beam_size, -1)
            score = F.log_softmax(score, dim=-1)
            for i, b in enumerate(beams):
                if not b.is_done():
                    b.advance(score[i])
            incr_state_inds = torch.cat(
                [
                    beam_size * i + b.get_backtrack_from_current_step()
                    for i, b in enumerate(beams)
                ]
            )
            incr_state = model.reorder_decoder_incremental_state(
                incr_state, incr_state_inds
            )
            decoder_input = torch.index_select(decoder_input, 0, incr_state_inds)
            selection = torch.cat(
                [b.get_output_from_current_step() for b in beams]
            ).unsqueeze(-1)
            decoder_input = torch.cat([decoder_input, selection], dim=-1)

        # get all finalized candidates for each sample (and validate them)
        n_best_beam_preds_scores = [b.get_rescored_finished() for b in beams]
        # print(f"n_best_beam_preds_scores: {n_best_beam_preds_scores}")

        score_back = self._compute_posterior(n_best_beam_preds_scores, batch)
        final_lst = self.rerank(n_best_beam_preds_scores, score_back, 0.7)

        beam_preds_scores = [n_best_list[0] for n_best_list in final_lst]
        return beam_preds_scores, beams


class TransformerClassifierAgent(TorchClassifierAgent):
    """
    Classifier based on Transformer.
    """

    @staticmethod
    def add_cmdline_args(parser):
        TransformerRankerAgent.add_cmdline_args(parser)  # add transformer args
        TorchClassifierAgent.add_cmdline_args(parser)
        parser.add_argument(
            '--load-from-pretrained-ranker',
            type='bool',
            default=False,
            help='load model from base transformer ranking model '
            '(used for pretraining)',
        )
        parser.set_params(reduction_type='first')

    def build_model(self):
        num_classes = len(self.class_list)
        self.base_model = TransformerMemNetModel(self.opt, self.dict)
        return TransformerLinearWrapper(self.base_model.context_encoder, num_classes)

    def vectorize(self, *args, **kwargs):
        """
        Add the start and end token to the text.
        """
        kwargs['add_start'] = True
        kwargs['add_end'] = True
        obs = super().vectorize(*args, **kwargs)
        return obs

    def _set_text_vec(self, *args, **kwargs):
        """
        Add the start and end token to the text.
        """
        obs = super()._set_text_vec(*args, **kwargs)

        if 'text_vec' in obs and 'added_start_end' not in obs:
            obs.force_set(
                'text_vec', self._add_start_end_tokens(obs['text_vec'], True, True)
            )
            obs['added_start_end'] = True

        return obs

    def score(self, batch):
        return self.model(batch.text_vec)

    def load_state_dict(self, state_dict):
        """
        Load the state dict into model.

        This is easily overridable to facilitate transfer of state dicts.
        """
        if self.is_finetune and self.opt['load_from_pretrained_ranker']:
            self.base_model.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(state_dict)

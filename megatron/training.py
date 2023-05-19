# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain utilities."""

from datetime import datetime
import math
import sys
import time
import json
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()

import nvtx
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import get_current_global_batch_size
from megatron import get_num_microbatches
from megatron import is_last_rank
from megatron import update_num_microbatches
from megatron import mpu
from megatron import print_rank_0
from megatron import print_rank_last
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.model import Float16Module
from megatron.optimizer import get_megatron_optimizer
from megatron.initialize import initialize_megatron
from megatron.initialize import write_args_to_tensorboard
from megatron.learning_rates import AnnealingLR
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import get_parameters_in_billions
from megatron.utils import unwrap_model
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.utils import calc_params_l2_norm
from megatron.schedules import forward_backward_no_pipelining
from megatron.schedules import forward_backward_pipelining_without_interleaving
from megatron.schedules import forward_backward_pipelining_with_interleaving
from megatron.utils import report_memory, throughput_calculator, checkpoint_throughput_calculator
from deepspeed.accelerator import get_accelerator
import deepspeed
from deepspeed.compression.compress import init_compression, redundancy_clean


from megatron.model.transformer import  ParallelTransformerLayer
from deepspeed.runtime.data_pipeline.data_routing.helper import convert_to_random_ltd

def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))

@nvtx.annotate("Pretrain", color="white")
def pretrain(train_valid_test_dataset_provider,
             model_provider,
             forward_step_func,
             extra_args_provider=None,
             args_defaults={},
             data_post_process=None):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = get_accelerator().FloatTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')

    args = get_args()
    timers = get_timers()

    if args.deepspeed:
        args.deepspeed_configuration = json.load(
            open(args.deepspeed_config, 'r', encoding='utf-8'))
        if "curriculum_learning" in args.deepspeed_configuration and \
            "enabled" in args.deepspeed_configuration["curriculum_learning"]:
            args.curriculum_learning_legacy = args.deepspeed_configuration[ \
                "curriculum_learning"]["enabled"]
        if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
            from deepspeed.runtime.data_pipeline.curriculum_scheduler \
                import CurriculumScheduler
            args.curriculum_scheduler = CurriculumScheduler( \
                args.deepspeed_configuration["curriculum_learning"])
        if "compression_training" in args.deepspeed_configuration:
            args.compression_training = True

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(
        model_provider, teacher=False, data_post_process=data_post_process,
        build_train_valid_test_datasets_provider=train_valid_test_dataset_provider)
    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')

    # Data stuff.
    with nvtx.annotate("Build all data iterators", color="orange"):
        timers('train/valid/test-data-iterators-setup').start()
        if args.virtual_pipeline_model_parallel_size is not None:
            all_data_iterators = [
                build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
                for _ in range(len(model))
            ]
            train_data_iterator = [data_iterators[0] for data_iterators in all_data_iterators]
            valid_data_iterator = [data_iterators[1] for data_iterators in all_data_iterators]
            test_data_iterator = [data_iterators[2] for data_iterators in all_data_iterators]
        else:
            train_data_iterator, valid_data_iterator, test_data_iterator \
                = build_train_valid_test_data_iterators(
                    train_valid_test_dataset_provider)
        if args.data_efficiency_curriculum_learning:
            if args.deepspeed_dataloader is not None:
                # We use args to pass the deepspeed_dataloader because adding
                # output to setup_model_and_optimizer will break the API for other
                # cases. We clear args.deepspeed_dataloader after updating
                # train_data_iterator because args will be saved in checkpoint and
                # attempting to save the whole deepspeed_dataloader will lead to
                # "AttributeError: Can't pickle local object...".
                train_data_iterator = iter(args.deepspeed_dataloader)
                args.deepspeed_dataloader = None
            else:
                train_data_iterator = None
        timers('train/valid/test-data-iterators-setup').stop()
        print_datetime('after dataloaders are built')

    # args.teacher_model is used as global variable to pass the teacher model
    # for knowledge distillation. Users do not need to set it in the command
    # line to use kd, but users do need to provide teacher model configurations
    # like args.num_layers_teacher as described in setup_teacher_model()
    args.teacher_model = None
    if args.mos or args.kd: # Set up teacher model
        args.teacher_model = setup_teacher_model(args, model_provider)

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup', 'train/valid/test-data-iterators-setup'])

    print('Model Params (B): {:.1f}'.format(get_parameters_in_billions(model)), flush=True)
    print_rank_0('training ...')

    iteration = 0
    if args.do_train and args.train_iters > 0:
        RANK_TENSOR_MODEL_PARALLEL_IDX = mpu.get_tensor_model_parallel_rank() # _TENSOR_MODEL_PARALLEL_GROUP
        RANK_PIPELINE_MODEL_PARALLEL_IDX = mpu.get_pipeline_model_parallel_rank() # _PIPELINE_MODEL_PARALLEL_GROUP
        RANK_DATA_PARALLEL_IDX = mpu.get_data_parallel_rank() # _DATA_PARALLEL_GROUP
        nvtx_train_marker = "Train [T_{},P_{},D_{}]".format(RANK_TENSOR_MODEL_PARALLEL_IDX, 
                                    RANK_PIPELINE_MODEL_PARALLEL_IDX, RANK_DATA_PARALLEL_IDX)
        with nvtx.annotate(nvtx_train_marker, color="purple"):
            iteration = train(forward_step_func,
                            model, optimizer, lr_scheduler,
                            train_data_iterator, valid_data_iterator)
    print_datetime('after training is done')

    if args.do_valid:
        prefix = 'the end of training for val data'
        evaluate_and_print_results(prefix, forward_step_func,
                                valid_data_iterator, model,
                                iteration, False)
    
    # Clean the model and do evaluation again
    if args.compression_training:
        model = [redundancy_clean(model[0], args.deepspeed_config, mpu)]
        if args.do_valid:
            prefix = 'the end of training and after model cleaning for val data'
            evaluate_and_print_results(prefix, forward_step_func,
                                    valid_data_iterator, model,
                                    iteration, False)


    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler)

    if args.do_test:
        # Run on test data.
        with nvtx.annotate("Test part", color="orange"):
            prefix = 'the end of training for test data'
            evaluate_and_print_results(prefix, forward_step_func,
                                    test_data_iterator, model,
                                    0, True, test=True)

def update_train_iters(args):

    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        iterations += (args.train_samples - consumed_samples) // \
                      args.global_batch_size
        args.train_iters = iterations

    print_rank_0('setting training iterations to {}'.format(args.train_iters))


def setup_teacher_model(args, model_provider):        
    
    print_rank_0('***>>>>> Student model checkpoint iteration:{}'.format(args.iteration))
    iteration_stuent = args.iteration
    num_layers_student = args.num_layers
    num_experts_student = args.num_experts
    hidden_size_student = args.hidden_size
    num_attention_heads_student = args.num_attention_heads
    load_student = args.load

    print_rank_0('***>>>>> Setting up the teacher model')

    args.num_layers = args.num_layers_teacher
    args.num_experts = args.num_experts_teacher
    args.hidden_size = args.hidden_size_teacher
    args.num_attention_heads = args.num_attention_heads_teacher
    args.load = args.load_teacher
    teacher_model, _, _ = load_model_weights_only(model_provider)
    print_rank_0('***>>>>> Teacher model:{}'.format(teacher_model))

    args.num_layers = num_layers_student
    args.num_experts = num_experts_student
    args.hidden_size = hidden_size_student
    args.num_attention_heads = num_attention_heads_student
    args.load = load_student
    args.iteration = iteration_stuent

    return teacher_model

def get_model(model_provider_func):
    """Build the model."""
    args = get_args()

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
       args.virtual_pipeline_model_parallel_size is not None:
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        model = model_provider_func(
            pre_process=pre_process,
            post_process=post_process
        )


    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            mpu.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            sum([sum([p.ds_numel if hasattr(p,'ds_id') else p.nelement() for p in model_module.parameters()])
                 for model_module in model])), flush=True)

    if args.deepspeed:
        return model

    # GPU allocation.
    for model_module in model:
        model_module.to(get_accelerator().current_device_name())
 

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    if args.DDP_impl == 'torch':
        i = get_accelerator().current_device()
        model = [torchDDP(model_module, device_ids=[i], output_device=i,
                          process_group=mpu.get_data_parallel_group())
                 for model_module in model]
        return model

    if args.DDP_impl == 'local':
        model = [LocalDDP(model_module,
                          args.accumulate_allreduce_grads_in_fp32,
                          args.use_contiguous_buffers_in_ddp)
                 for model_module in model]
        return model

    raise NotImplementedError('Unknown DDP implementation specified: {}. '
                              'Exiting.'.format(args.DDP_impl))


def get_learning_rate_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Iteration-based training.
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        decay_steps = args.lr_decay_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_iters * args.global_batch_size
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        decay_steps = args.lr_decay_samples
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_samples
    else:
        raise Exception(
            'either train-iters or train-samples should be provided.')

    lr_scheduler = AnnealingLR(
        optimizer,
        max_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        decay_style=args.lr_decay_style,
        use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=args.override_lr_scheduler)

    return lr_scheduler

def load_model_weights_only(model_provider_func):
    """Setup model and optimizer."""
    args = get_args()
    print_rank_0('***>>>>> Args:{}'.format(args))

    model = get_model(model_provider_func)

    optimizer = None
    lr_scheduler = None

    if args.deepspeed:
        with open(args.deepspeed_config, 'r') as fd:
            ds_config = json.load(fd)

        # When loading just the model weights, ZeRO can be disabled.
        if 'zero_optimization' in ds_config:
            del ds_config['zero_optimization']

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model[0],
            config=ds_config
        )

        assert not isinstance(model, deepspeed.PipelineEngine), \
            'Weight loading only mode is not supported in pipeline parallelism yet.'

        model = [model]

    print_datetime('before load checkpoint')
    if args.load is not None:
        iteration = load_checkpoint(model, optimizer, lr_scheduler, strict=True, load_only_weights=True)

    print_datetime('after load checkpoint weights')

    return model, optimizer, lr_scheduler

@nvtx.annotate("setup_model_and_optimizer", color="white")
def setup_model_and_optimizer(model_provider_func, teacher=False,
    data_post_process=None, build_train_valid_test_datasets_provider=None):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func)

    # initialize the compression here
    student_global_steps = 0
    if args.kd or args.mos:
        model, _, _, _ = deepspeed.initialize(
                model=model[0],
                args=args,
                mpu=mpu if args.no_pipeline_parallel else None
            )
        model = [model]
        if args.load is not None:
            args.iteration = load_checkpoint(model, None, None, strict=False)
        else:
            args.iteration = 0
        student_global_steps = model[0].global_steps
        print_rank_0('***>>>>> Student model, global step:{}'.format(student_global_steps))


    if args.compression_training:
        model, _, _, _ = deepspeed.initialize(
            model=model[0],
            args=args,
            mpu=mpu if args.no_pipeline_parallel else None
        )
        model = [model]
        model = [init_compression(model[0].module, args.deepspeed_config, mpu)]
    

    unwrapped_model = unwrap_model(model,
                                   (torchDDP, LocalDDP, Float16Module))

    if args.inference:
        optimizer = None
        lr_scheduler = None
    else:
        if teacher:
          optimizer = None
        else:
          optimizer = get_megatron_optimizer(unwrapped_model)
        lr_scheduler = get_learning_rate_scheduler(optimizer)


    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        pp = mpu.get_pipeline_model_parallel_world_size()
        if args.data_efficiency_curriculum_learning and build_train_valid_test_datasets_provider is not None:
            train_ds = None
            # Only need to build dataset on tp rank 0 since Megatron has the
            # broadcast_data() function that broadcast data from tp rank 0.
            if mpu.get_tensor_model_parallel_rank() == 0:
                # Number of train/valid/test samples.
                if args.train_samples:
                    train_samples = args.train_samples
                else:
                    train_samples = args.train_iters * args.global_batch_size
                # eval_iters and test_iters here are not actually used, only for
                # satisfying the input of build_train_valid_test_datasets_provider.
                # We only need to build the training data here. And we follow
                # baseline's logic to build eval/test dataset later in
                # build_train_valid_test_data_iterators.
                eval_iters = (args.train_iters // args.eval_interval + 1) * \
                            args.eval_iters
                test_iters = args.eval_iters
                train_val_test_num_samples = [train_samples,
                                            eval_iters * args.global_batch_size,
                                            test_iters * args.global_batch_size]
                # Build the datasets.
                train_ds, _, _ = build_train_valid_test_datasets_provider(
                    train_val_test_num_samples)
            model, optimizer, args.deepspeed_dataloader, lr_scheduler = deepspeed.initialize(
                model=model[0],
                optimizer=optimizer,
                args=args,
                lr_scheduler=lr_scheduler,
                training_data=train_ds,
                mpu=mpu if args.no_pipeline_parallel else None
            )
            model.set_data_post_process_func(data_post_process)
        else:
            model, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=model[0],
                optimizer=optimizer,
                args=args,
                lr_scheduler=lr_scheduler,
                mpu=mpu if args.no_pipeline_parallel else None
            )
        if isinstance(model, deepspeed.PipelineEngine):
            # hack to get batch_fn from pretrain_gpt.py
            model.set_batch_fn(model.module._megatron_batch_fn)

            assert model.grid.get_pipe_parallel_rank() == mpu.get_pipeline_model_parallel_rank()
            assert model.grid.get_slice_parallel_rank() == mpu.get_tensor_model_parallel_rank()
            assert model.grid.get_data_parallel_rank() == mpu.get_data_parallel_rank()
        model = [model]

    # Compression has its own checkpoint loading path (e.g, loading both teacher and student models). So if compression is enabled, we skip the following checkpoint loading.
    no_post_init_checkpoint_loading = args.kd or args.mos
    if not no_post_init_checkpoint_loading:
        if args.load is not None:
            timers = get_timers()
            # Extra barrier is added to make sure all ranks report the
            # max time.
            torch.distributed.barrier()
            timers('load-checkpoint').start()
            args.iteration = load_checkpoint(model, optimizer, lr_scheduler)
            torch.distributed.barrier()
            timers('load-checkpoint').stop()
            timers.log(['load-checkpoint'])
        else:
            args.iteration = 0
    else:
        model[0].global_steps = student_global_steps

    # We only support local DDP with multiple micro-batches.
    if len(model) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
        assert args.DDP_impl == 'local'

    # get model without FP16 and/or TorchDDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    # random-LTD requires converting transformer layers
    if args.random_ltd:
        model[0] = convert_to_random_ltd(model[0], ParallelTransformerLayer)

    return model, optimizer, lr_scheduler

@nvtx.annotate("Train_step", color="green")
def train_step(forward_step_func, data_iterator,
               model, optimizer, lr_scheduler):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    if args.deepspeed and args.ds_pipeline_enabled:
        skipped_iter = 0
        num_zeros_in_grad = 0
        assert isinstance(model[0], deepspeed.PipelineEngine)
        loss = model[0].train_batch(data_iter=data_iterator)
        grad_norm = model[0].get_global_grad_norm()
        return {'lm loss' : loss}, skipped_iter, grad_norm, num_zeros_in_grad

    # Set grad to zero.
    if not args.deepspeed:
        if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_ddp:
            for partition in model:
                partition.zero_grad_buffer()
        else:
            optimizer.zero_grad()

    if mpu.get_pipeline_model_parallel_world_size() > 1:
        if args.virtual_pipeline_model_parallel_size is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
            assert get_num_microbatches() % args.pipeline_model_parallel_size == 0, \
                'number of microbatches is not divisible by pipeline-parallel ' \
                'size when using interleaved schedule'
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    if args.mos or args.kd:
        # args.teacher_forward is used as global variable to enable kd loss
        # calculation in forward pass. Users do not need to set it in the
        # command line to use kd.
        args.teacher_forward = True
    losses_reduced = forward_backward_func(
        forward_step_func, data_iterator, model,
        optimizer, timers, forward_only=False)
    if args.mos or args.kd:
        args.teacher_forward = False

    # All-reduce if needed.
    if not args.deepspeed and args.DDP_impl == 'local':
        with nvtx.annotate("backward-params-all-reduce", color="yellow"):
            timers('backward-params-all-reduce').start()
            for model_module in model:
                model_module.allreduce_gradients()
            timers('backward-params-all-reduce').stop()

    # All-reduce word_embeddings' grad across first and last stages to ensure
    # that word_embeddings parameters stay in sync.
    # This should only run for models that support pipelined model parallelism
    # (BERT and GPT-2).
    with nvtx.annotate("backward-embedding-all-reduce", color="yellow"):
        timers('backward-embedding-all-reduce').start()
        if not args.deepspeed:
            if (mpu.is_pipeline_first_stage(ignore_virtual=True) or
                mpu.is_pipeline_last_stage(ignore_virtual=True)) and \
                    mpu.get_pipeline_model_parallel_world_size() > 1:
                if mpu.is_pipeline_first_stage(ignore_virtual=True):
                    unwrapped_model = model[0]
                elif mpu.is_pipeline_last_stage(ignore_virtual=True):
                    unwrapped_model = model[-1]
                unwrapped_model = unwrap_model(
                    unwrapped_model, (torchDDP, LocalDDP, Float16Module))

                if unwrapped_model.share_word_embeddings:
                    word_embeddings_weight = unwrapped_model.word_embeddings_weight()
                    if args.DDP_impl == 'local':
                        grad = word_embeddings_weight.main_grad
                    else:
                        grad = word_embeddings_weight.grad
                    torch.distributed.all_reduce(grad, group=mpu.get_embedding_group())
        timers('backward-embedding-all-reduce').stop()

    # Update parameters.
    with nvtx.annotate("optimizer", color="green"):
        timers('optimizer').start()
        if args.deepspeed:
            increment = get_num_microbatches() * \
                        args.micro_batch_size * \
                        args.data_parallel_size
            model[0].step(lr_kwargs={'increment': increment})
            update_successful = model[0].was_step_applied()
        else:
            update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        timers('optimizer').stop()

    # Update learning rate.
    if args.deepspeed:
        skipped_iter = 0
        grad_norm = None
        num_zeros_in_grad = None
        
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    else:
        if update_successful:
            increment = get_num_microbatches() * \
                        args.micro_batch_size * \
                        args.data_parallel_size
            lr_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in losses_reduced[0]:
                losses_reduced_for_key = [x[key] for x in losses_reduced]
                loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad,
                 model=None, optimizer=None):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, get_accelerator().FloatTensor([0.0])) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = []

    def add_to_logging(name):
        if name in timers.timers:
            timers_to_log.append(name)
    add_to_logging('forward-compute')
    add_to_logging('forward-recv')
    add_to_logging('forward-send')
    add_to_logging('forward-backward-send-forward-backward-recv')
    add_to_logging('backward-compute')
    add_to_logging('backward-recv')
    add_to_logging('backward-send')
    add_to_logging('backward-send-forward-recv')
    add_to_logging('backward-send-backward-recv')
    add_to_logging('backward-params-all-reduce')
    add_to_logging('backward-embedding-all-reduce')
    add_to_logging('optimizer-copy-to-main-grad')
    add_to_logging('optimizer-unscale-and-check-inf')
    add_to_logging('optimizer-clip-main-grad')
    add_to_logging('optimizer-copy-main-to-model-params')
    add_to_logging('optimizer')
    add_to_logging('batch-generator')
    add_to_logging('save-checkpoint')

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \
        get_num_microbatches()
    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    if writer and (iteration % args.tensorboard_log_interval == 0) and \
       is_last_rank():
        writer.add_scalar('steps-vs-samples/y=steps,x=samples', iteration, args.consumed_train_samples)
        writer.add_scalar('steps-vs-samples/y=samples,x=steps', args.consumed_train_samples, iteration)
        writer.add_scalar('steps-vs-tokens/y=steps,x=tokens', iteration, args.consumed_train_tokens)
        writer.add_scalar('steps-vs-tokens/y=tokens,x=steps', args.consumed_train_tokens, iteration)
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar('learning-rate/learning-rate', learning_rate, iteration)
            writer.add_scalar('learning-rate/learning-rate vs samples', learning_rate,
                              args.consumed_train_samples)
            writer.add_scalar('learning-rate/learning-rate vs tokens', learning_rate,
                              args.consumed_train_tokens)
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar('batch-size/batch-size', batch_size, iteration)
            writer.add_scalar('batch-size/batch-size vs samples', batch_size,
                              args.consumed_train_samples)
        for key in loss_dict:
            writer.add_scalar(f"lm-loss-training/{key}", loss_dict[key], iteration)
            writer.add_scalar(f"lm-loss-training/{key}" + ' vs samples', loss_dict[key],
                              args.consumed_train_samples)
            writer.add_scalar(f"lm-loss-training/{key}" + ' vs tokens', loss_dict[key],
                              args.consumed_train_tokens)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale/loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale/loss-scale vs samples', loss_scale,
                              args.consumed_train_samples)
            writer.add_scalar('loss-scale/loss-scale vs tokens', loss_scale,
                              args.consumed_train_tokens)
        if grad_norm is not None:
            writer.add_scalar('grad-norm/grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm/grad-norm vs samples', grad_norm,
                              args.consumed_train_samples)
            writer.add_scalar('grad-norm/grad-norm vs tokens', grad_norm,
                              args.consumed_train_tokens)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros/num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros/num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples)
            writer.add_scalar('num-zeros/num-zeros vs tokens', num_zeros_in_grad,
                              args.consumed_train_tokens)
        if params_norm is not None:
            writer.add_scalar('params-norm/params-norm', params_norm, iteration)
            writer.add_scalar('params-norm/params-norm vs samples', params_norm,
                              args.consumed_train_samples)
            writer.add_scalar('params-norm/params-norm vs tokens', params_norm,
                              args.consumed_train_tokens)
        if hasattr(args, 'actual_seq_length'):
            writer.add_scalar('seqlen/actual_seq_length', args.actual_seq_length,
                              iteration)
            writer.add_scalar('seqlen/actual_seq_length vs samples', args.actual_seq_length,
                              args.consumed_train_samples)
            writer.add_scalar('seqlen/actual_seq_length vs tokens', args.actual_seq_length,
                              args.consumed_train_tokens)
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            writer.add_scalar('seqlen/curriculum_seqlen', args.curriculum_seqlen,
                              iteration)
            writer.add_scalar('seqlen/curriculum_seqlen vs samples', args.curriculum_seqlen,
                              args.consumed_train_samples)
            writer.add_scalar('seqlen/curriculum_seqlen vs tokens', args.curriculum_seqlen,
                              args.consumed_train_tokens)
        if args.random_ltd:
            writer.add_scalar('seqlen/random_ltd_reserved_length', args.random_ltd_reserved_length,
                              iteration)
            writer.add_scalar('seqlen/random_ltd_reserved_length vs samples', args.random_ltd_reserved_length,
                              args.consumed_train_samples)
            writer.add_scalar('seqlen/random_ltd_reserved_length vs tokens', args.random_ltd_reserved_length,
                              args.consumed_train_tokens)
        if args.log_timers_to_tensorboard:
            timers.write(timers_to_log, writer, iteration,
                         normalizer=total_iterations)

    if iteration % args.tensorboard_log_interval == 0:
        # This logging write various optimizer states to tensorboard. This
        # feature may consume extra GPU memory thus is set at false by default.
        if args.log_optimizer_states_to_tensorboard and optimizer is not None:
            opt_stats = [0.0] * 8
            opt_stats_2 = [0.0] * 4
            for _, group in enumerate(optimizer.param_groups):
                for _, param in enumerate(group['params']):
                    opt_stats[0] += (torch.norm(optimizer.state[param]['exp_avg_sq']).item())**2
                    opt_stats[1] += (torch.norm(optimizer.state[param]['exp_avg_sq'].sqrt()).item())**2
                    opt_stats[2] += (torch.norm(optimizer.state[param]['exp_avg']).item())**2
                    opt_stats[3] += (torch.norm(param).item())**2
                    opt_stats[4] += torch.norm(optimizer.state[param]['exp_avg_sq'],p=1).item()
                    opt_stats[5] += torch.norm(optimizer.state[param]['exp_avg_sq'].sqrt(),p=1).item()
                    opt_stats[6] += torch.norm(optimizer.state[param]['exp_avg'],p=1).item()
                    opt_stats[7] += torch.norm(param,p=1).item()
                    opt_stats_2[0] = max(opt_stats_2[0], abs(optimizer.state[param]['exp_avg_sq'].max().item()), abs(optimizer.state[param]['exp_avg_sq'].min().item()))
                    opt_stats_2[1] = max(opt_stats_2[1], optimizer.state[param]['exp_avg_sq'].sqrt().abs_().max().item())
                    opt_stats_2[2] = max(opt_stats_2[2], abs(optimizer.state[param]['exp_avg'].max().item()), abs(optimizer.state[param]['exp_avg'].min().item()))
                    opt_stats_2[3] = max(opt_stats_2[3], abs(param.max().item()), abs(param.min().item()))
            # print('step {} rank {} before sync opt_stats {}, {}'.format(iteration, torch.distributed.get_rank(), opt_stats_2, opt_stats))
            if args.zero_stage > 0:
                # ZeRO partiions optimizer states
                opt_stats = get_accelerator().FloatTensor(opt_stats)
                torch.distributed.all_reduce(opt_stats, group=mpu.get_data_parallel_group())
                opt_stats_2 = get_accelerator().FloatTensor(opt_stats_2)
                torch.distributed.all_reduce(opt_stats_2, op=torch.distributed.ReduceOp.MAX,
                    group=mpu.get_data_parallel_group())

            if args.tensor_model_parallel_size > 1:
                opt_stats = get_accelerator().FloatTensor(opt_stats)
                torch.distributed.all_reduce(opt_stats, group=mpu.get_tensor_model_parallel_group())
                opt_stats_2 = get_accelerator().FloatTensor(opt_stats_2)
                torch.distributed.all_reduce(opt_stats_2, op=torch.distributed.ReduceOp.MAX,
                    group=mpu.get_tensor_model_parallel_group())

            if args.pipeline_model_parallel_size > 1:
                opt_stats = get_accelerator().FloatTensor(opt_stats)
                torch.distributed.all_reduce(opt_stats, group=mpu.get_pipeline_model_parallel_group())
                opt_stats_2 = get_accelerator().FloatTensor(opt_stats_2)
                torch.distributed.all_reduce(opt_stats_2, op=torch.distributed.ReduceOp.MAX,
                    group=mpu.get_pipeline_model_parallel_group())

            # print('step {} rank {} after sync opt_stats {}, {}'.format(iteration, torch.distributed.get_rank(), opt_stats_2, opt_stats))
            if writer and is_last_rank():
                writer.add_scalar('optimizer/variance_l2 vs tokens', opt_stats[0]**0.5, args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_sqrt_l2 vs tokens', opt_stats[1]**0.5, args.consumed_train_tokens)
                writer.add_scalar('optimizer/momentum_l2 vs tokens', opt_stats[2]**0.5, args.consumed_train_tokens)
                writer.add_scalar('optimizer/weight_l2 vs tokens', opt_stats[3]**0.5, args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_l1 vs tokens', opt_stats[4], args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_sqrt_l1 vs tokens', opt_stats[5], args.consumed_train_tokens)
                writer.add_scalar('optimizer/momentum_l1 vs tokens', opt_stats[6], args.consumed_train_tokens)
                writer.add_scalar('optimizer/weight_l1 vs tokens', opt_stats[7], args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_abs_max vs tokens', opt_stats_2[0], args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_sqrt_abs_max vs tokens', opt_stats_2[1], args.consumed_train_tokens)
                writer.add_scalar('optimizer/momentum_abs_max vs tokens', opt_stats_2[2], args.consumed_train_tokens)
                writer.add_scalar('optimizer/weight_abs_max vs tokens', opt_stats_2[3], args.consumed_train_tokens)

                writer.add_scalar('optimizer/variance_l2', opt_stats[0]**0.5, iteration)
                writer.add_scalar('optimizer/variance_sqrt_l2', opt_stats[1]**0.5, iteration)
                writer.add_scalar('optimizer/momentum_l2', opt_stats[2]**0.5, iteration)
                writer.add_scalar('optimizer/weight_l2', opt_stats[3]**0.5, iteration)
                writer.add_scalar('optimizer/variance_l1', opt_stats[4], iteration)
                writer.add_scalar('optimizer/variance_sqrt_l1', opt_stats[5], iteration)
                writer.add_scalar('optimizer/momentum_l1', opt_stats[6], iteration)
                writer.add_scalar('optimizer/weight_l1', opt_stats[7], iteration)
                writer.add_scalar('optimizer/variance_abs_max', opt_stats_2[0], iteration)
                writer.add_scalar('optimizer/variance_sqrt_abs_max', opt_stats_2[1], iteration)
                writer.add_scalar('optimizer/momentum_abs_max', opt_stats_2[2], iteration)
                writer.add_scalar('optimizer/weight_abs_max', opt_stats_2[3], iteration)

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed()
        elapsed_time_per_iteration = elapsed_time / total_iterations
        seq_len = args.seq_length
        if hasattr(args, 'actual_seq_length'):
            seq_len = args.actual_seq_length
        hidden_size = args.hidden_size
        num_layers = args.num_layers
        vocab_size = args.padded_vocab_size

        samples_per_sec, tflops, approx_parameters_in_billions = throughput_calculator(model, args, elapsed_time, total_iterations)

        # Compute throughput.
        samples_per_sec_per_replica = samples_per_sec / args.data_parallel_size
        tokens_per_sec = samples_per_sec * seq_len
        tokens_per_sec_per_replica = tokens_per_sec / args.data_parallel_size

        # only the last rank process has a non-None _GLOBAL_TENSORBOARD_WRITER
        if writer and is_last_rank():
            if args.log_timers_to_tensorboard:
                writer.add_scalar('iteration-time/iteration-time',
                                  elapsed_time_per_iteration, iteration)
                writer.add_scalar('iteration-time/iteration-time vs samples',
                                  elapsed_time_per_iteration, args.consumed_train_samples)
                writer.add_scalar('iteration-time/iteration-time vs tokens',
                                  elapsed_time_per_iteration, args.consumed_train_tokens)
        log_string = ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        log_string += ' consumed tokens: {:12d} |'.format(
            args.consumed_train_tokens)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        log_string += ' global batch size: {:5d} |'.format(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = get_accelerator().FloatTensor([0.0])
        log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            log_string += ' curriculum seqlen: {:5d} |'.format(args.curriculum_seqlen)
        if args.random_ltd:
            log_string += ' random ltd reserved length: {:5d} |'.format(args.random_ltd_reserved_length)
        log_string += ' actual seqlen: {:5d} |'.format(seq_len)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        log_string += ' samples per second: {:.3f} |'.format(samples_per_sec)
        log_string += ' TFLOPs: {:.2f} |'.format(tflops)
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag and learning_rate > 0.:
            # Report memory after optimizer state has been initialized.
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)


    return report_memory_flag

@nvtx.annotate("Save checkpoint and time", color="brown")
def save_checkpoint_and_time(iteration, model, optimizer, lr_scheduler):
    timers = get_timers()
    # Extra barrier is added to make sure
    # all ranks report the max time.
    torch.distributed.barrier()
    timers('save-checkpoint').start()
    save_checkpoint(iteration, model, optimizer, lr_scheduler)
    torch.distributed.barrier()
    timers('save-checkpoint').stop()
    checkpoint_throughput_calculator(model, timers('save-checkpoint').elapsed(reset=False))
    timers.log(['save-checkpoint'])


# @nvtx.annotate("Train", color="purple")
def train(forward_step_func, model, optimizer, lr_scheduler,
          train_data_iterator, valid_data_iterator):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    # Write args to tensorboard
    write_args_to_tensorboard()

    if args.random_ltd:
        # random-ltd requires different randomness on each rank
        import random
        random.seed(args.seed + torch.distributed.get_rank())

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration

    timers('interval-time').start()
    print_datetime('before the start of training step')
    report_memory_flag = True
    if args.random_ltd:
        assert model[0].random_ltd_enabled()
        args.random_ltd_layer_num = model[0].random_ltd_scheduler.get_random_ltd_layer_num()
    
    while iteration < args.train_iters and (args.train_tokens is None or \
        args.consumed_train_tokens < args.train_tokens):
        # with nvtx.annotate("Training", color="green"):
        update_num_microbatches(args.consumed_train_samples)
        if args.deepspeed:
            # inform deepspeed of any batch size changes
            global_batch_size = mpu.get_data_parallel_world_size() * \
                                args.micro_batch_size * \
                                get_num_microbatches()
            model[0].set_train_batch_size(global_batch_size)

        if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
            args.curriculum_seqlen = args.curriculum_scheduler.update_difficulty( \
                    args.iteration + 1)
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
            train_step(forward_step_func,
                    train_data_iterator,
                    model,
                    optimizer,
                    lr_scheduler)
        iteration += 1
        args.iteration = iteration
        new_samples = mpu.get_data_parallel_world_size() * \
                                    args.micro_batch_size * \
                                    get_num_microbatches()
        args.consumed_train_samples += new_samples
        # This actual_seq_length is used for actual consumed tokens calculation, flops calculation, and logging.
        args.actual_seq_length = args.seq_length
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            args.actual_seq_length = args.curriculum_seqlen
        if args.random_ltd:
            args.random_ltd_reserved_length = model[0].random_ltd_scheduler.get_current_seq()
            if args.random_ltd_reserved_length < args.actual_seq_length:
                args.actual_seq_length = (args.actual_seq_length * (args.num_layers - args.random_ltd_layer_num) + args.random_ltd_reserved_length * args.random_ltd_layer_num) // args.num_layers
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            if hasattr(args, 'data_efficiency_curriculum_learning_numel'):
                act_mbsz = args.data_efficiency_curriculum_learning_numel / args.curriculum_seqlen
                act_token = act_mbsz * args.actual_seq_length
                args.consumed_train_tokens += mpu.get_data_parallel_world_size() * \
                        get_num_microbatches() * act_token
            else:
                args.consumed_train_tokens += new_samples * args.actual_seq_length
        else:
            args.consumed_train_tokens += new_samples * args.actual_seq_length
        
        # Logging.
        if args.deepspeed:
            if hasattr(model[0].optimizer, 'cur_scale'):
                loss_scale = model[0].optimizer.cur_scale
            else:
                loss_scale = None
        else:
            loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                        optimizer.param_groups[0]['lr'],
                                        iteration, loss_scale,
                                        report_memory_flag, skipped_iter,
                                        grad_norm, params_norm, num_zeros_in_grad,
                                        model, optimizer)

        # Autoresume
        if args.adlr_autoresume and \
        (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                            lr_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \
        args.do_valid:
            prefix = 'iteration {}'.format(iteration)
            evaluate_and_print_results(prefix, forward_step_func,
                                    valid_data_iterator, model,
                                    iteration, False)

        # Checkpointing
        saved_checkpoint = False
        if args.save and args.save_interval and \
        iteration % args.save_interval == 0:
            with nvtx.annotate("Checkpointing", color="yellow"):
                save_checkpoint_and_time(iteration, model, optimizer,
                                        lr_scheduler)
                saved_checkpoint = True

        # Exiting based on duration
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = get_accelerator().IntTensor(
                [train_time > args.exit_duration_in_mins])
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(iteration, model, optimizer,
                                            lr_scheduler)
                print_datetime('exiting program after {} minutes'.format(train_time))
                sys.exit()

        # Exiting based on iterations
        if args.exit_interval and iteration % args.exit_interval == 0:
            if not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                        lr_scheduler)
            torch.distributed.barrier()
            print_datetime('exiting program at iteration {}'.format(iteration))
            sys.exit()


    return iteration

@nvtx.annotate("Evaluate compute", color="red")
def evaluate(forward_step_func, data_iterator, model, verbose=False):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
        # When curriculum learning is used with pipeline parallelism, we need
        # this logic to ensure that the eval data is not truncated. If there
        # is a seqlen change due to that, we need to call
        # reset_activation_shape() to reset some buffers in deepspeed pipeline
        # engine.
        if args.curriculum_seqlen < args.seq_length:
            args.curriculum_seqlen = args.seq_length
            model[0].reset_activation_shape()

    total_loss_dict = {}

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration,
                                                            args.eval_iters))

            if mpu.get_pipeline_model_parallel_world_size() > 1:
                if args.virtual_pipeline_model_parallel_size is not None:
                    forward_backward_func = forward_backward_pipelining_with_interleaving
                else:
                    forward_backward_func = forward_backward_pipelining_without_interleaving
            else:
                forward_backward_func = forward_backward_no_pipelining
            
            if args.deepspeed and args.ds_pipeline_enabled:
                # DeepSpeed uses eval_batch() and already aggregates losses.
                assert isinstance(model, list) and len(model) == 1
                loss = model[0].eval_batch(data_iterator)
                loss_dicts = [{'lm loss' : loss}] * get_num_microbatches()
            else:
                loss_dicts = forward_backward_func(
                    forward_step_func, data_iterator, model, optimizer=None,
                    timers=None, forward_only=True)
            
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        if 'moe' not in key:
                            total_loss_dict[key] = total_loss_dict.get(
                                key, get_accelerator().FloatTensor([0.0])) + loss_dict[key]

            args.consumed_valid_samples += mpu.get_data_parallel_world_size() \
                                           * args.micro_batch_size \
                                           * get_num_microbatches()
    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * get_num_microbatches()

    if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
        # roll back to actual curriculum seqlen at the end of eval.
        args.curriculum_seqlen = args.curriculum_scheduler.update_difficulty( \
            args.iteration + 1)
        if args.curriculum_seqlen < args.seq_length:
            model[0].reset_activation_shape()

    return total_loss_dict


def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, verbose=False, test=False):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    writer = get_tensorboard_writer()

    total_loss_dict = evaluate(forward_step_func, data_iterator, model, verbose)
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer and is_last_rank():
            data_type = 'test' if test else 'validation'
            writer.add_scalar(f'lm-loss-validation/{key} {data_type}',
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar(f'lm-loss-validation/{key} {data_type} vs samples',
                              total_loss_dict[key].item(),
                              args.consumed_train_samples)
            writer.add_scalar(f'lm-loss-validation/{key} {data_type} vs tokens',
                              total_loss_dict[key].item(),
                              args.consumed_train_tokens)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar(f'lm-loss-validation/{key} {data_type} ppl', ppl,
                                  iteration)
                writer.add_scalar(f'lm-loss-validation/{key} {data_type} ppl vs samples',
                                  ppl, args.consumed_train_samples)
                writer.add_scalar(f'lm-loss-validation/{key} {data_type} ppl vs tokens',
                                  ppl, args.consumed_train_tokens)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x

def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """XXX"""
    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
            args.eval_iters * args.global_batch_size

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_tensor_model_parallel_rank() == 0:

        # Number of train/valid/test samples.
        if args.train_samples:
            train_samples = args.train_samples
        else:
            train_samples = args.train_iters * args.global_batch_size
        eval_iters = (args.train_iters // args.eval_interval + 1) * \
                     args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [train_samples,
                                      eval_iters * args.global_batch_size,
                                      test_iters * args.global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        # Build the datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
            train_val_test_num_samples)

        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples)
        valid_dataloader = build_pretraining_data_loader(
            valid_ds, args.consumed_valid_samples)
        test_dataloader = build_pretraining_data_loader(test_ds, 0)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = get_accelerator().LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = get_accelerator().LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(flags,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()


    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic']

    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader) if dl_type == 'single' \
                              else iter(cyclic_iter(train_dataloader))
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader) if dl_type == 'single' \
                              else iter(cyclic_iter(valid_dataloader))
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader) if dl_type == 'single' \
                             else iter(cyclic_iter(test_dataloader))
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator

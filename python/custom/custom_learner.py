# 
"""
Custom Learner class that extends the tensorflow learner class to have access to the experience sample used to train the agent

@author: roshan94
"""
import os
import gin

from typing import Any, Optional, Tuple

import tensorflow as tf

from absl import logging
from tf_agents.agents import tf_agent
from tf_agents.typing import types
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.train import interval_trigger

TRAIN_DIR = 'train'
POLICY_SAVED_MODEL_DIR = 'policies'
COLLECT_POLICY_SAVED_MODEL_DIR = 'collect_policy'
GREEDY_POLICY_SAVED_MODEL_DIR = 'greedy_policy'
RAW_POLICY_SAVED_MODEL_DIR = 'policy'
POLICY_CHECKPOINT_DIR = 'checkpoints'
REPLAY_BUFFER_CHECKPOINT_DIR = 'replay_buffer_checkpoints'

ExperienceAndSampleInfo = Tuple[types.NestedTensor, Tuple[Any, ...]]

@gin.configurable
class CustomLearner(tf.Module):

    def __init__(
      self,
      root_dir,
      train_step,
      agent,
      experience_dataset_fn=None,
      after_train_strategy_step_fn=None,
      triggers=None,
      checkpoint_interval=100000,
      summary_interval=1000,
      max_checkpoints_to_keep=3,
      use_kwargs_in_agent_train=False,
      strategy=None,
      run_optimizer_variable_init=True,
      use_reverb_v2=False,
      direct_sampling=False,
      experience_dataset_options=None,
      strategy_run_options=None,
      summary_root_dir=None,
    ):
        if checkpoint_interval < 0:
            logging.warning(
                'Warning: checkpointing the training process is manually disabled.'
                'This means training progress will NOT be automatically restored '
                'if the job gets preempted.'
            )

        self._train_dir = os.path.join(root_dir, TRAIN_DIR)
        summary_root_dir = (
            root_dir if summary_root_dir is None else summary_root_dir
        )
        self._summary_dir = os.path.join(summary_root_dir, TRAIN_DIR)
        self._use_reverb_v2 = use_reverb_v2
        self._direct_sampling = direct_sampling
        if summary_interval:
            self.train_summary_writer = tf.compat.v2.summary.create_file_writer(
                self._summary_dir, flush_millis=10000)
        else:
            self.train_summary_writer = tf.summary.create_noop_writer()

        self.train_step = train_step
        self._agent = agent
        self.use_kwargs_in_agent_train = use_kwargs_in_agent_train
        self.strategy = strategy or tf.distribute.get_strategy()

        dataset = None
        if experience_dataset_fn:
            with self.strategy.scope():
                dataset = self.strategy.distribute_datasets_from_function(
                    lambda _: experience_dataset_fn(),
                    options=experience_dataset_options,
                )
                self._experience_iterator = iter(dataset)

        self.after_train_strategy_step_fn = after_train_strategy_step_fn
        self.triggers = triggers or []

        # Prevent autograph from going into the agent.
        self._agent.train = tf.autograph.experimental.do_not_convert(agent.train)

        self._strategy_run_options = strategy_run_options

        checkpoint_dir = os.path.join(self._train_dir, POLICY_CHECKPOINT_DIR)
        with self.strategy.scope():
            agent.initialize()

            if run_optimizer_variable_init:
                # Force a concrete function creation inside of the strategy scope to
                # ensure that all variables, including optimizer slot variables, are
                # created. This has to happen before the checkpointer is created.
                if dataset is not None:
                    if use_reverb_v2:
                        batched_specs = dataset.element_spec.data
                    else:
                        ## Changed from two outputs expected previously which assumed (experience, sample_info) = next(iterator)
                        ## PyUniformReplayBuffer dataset does not output sample_info 
                        batched_specs = dataset.element_spec 
                else:
                    batched_specs = tensor_spec.add_outer_dims_nest(
                        self._agent.training_data_spec,
                        (None, self._agent.train_sequence_length),
                    )
                if self.use_kwargs_in_agent_train:
                    batched_specs = dict(experience=batched_specs)

                @common.function
                def _create_variables(specs):
                    # TODO(b/170516529): Each replica has to be in the same graph.
                    # This can be ensured by placing the `strategy.run(...)` call inside
                    # the `tf.function`.
                    if self.use_kwargs_in_agent_train:
                        return self.strategy.run(
                            self._agent.train,
                            kwargs=specs,
                            options=self._strategy_run_options,
                        )
                    return self.strategy.run(
                        self._agent.train,
                        args=(specs,),
                        options=self._strategy_run_options,
                    )

                _create_variables.get_concrete_function(batched_specs)
            else:
                # TODO(b/186052656) Update clients.
                logging.warning('run_optimizer_variable_init = False is Deprecated')

            self._checkpointer = common.Checkpointer(
                checkpoint_dir,
                max_to_keep=max_checkpoints_to_keep,
                agent=self._agent,
                train_step=self.train_step,
            )
            self._checkpointer.initialize_or_restore()  # pytype: disable=attribute-error

        for trigger in self.triggers:
            if hasattr(trigger, 'set_start'):
                trigger.set_start(self.train_step.numpy())

        self.triggers.append(self._get_checkpoint_trigger(checkpoint_interval))
        self.summary_interval = tf.constant(summary_interval, dtype=tf.int64)
    
        self.current_experience_sample = None  ## Extra line to have access to current training experience sample for saving purposes
    
    def run(self, iterations=1, iterator=None, parallel_iterations=10):
        assert iterations >= 1, (
            'Iterations must be greater or equal to 1, was %d' % iterations)

        def _summary_record_if():
            if self.summary_interval:
                return tf.math.equal(self.train_step % tf.constant(self.summary_interval), 0)
            else:
                return tf.constant(False)

        with self.train_summary_writer.as_default(), common.soft_device_placement(), tf.compat.v2.summary.record_if(
        _summary_record_if), self.strategy.scope():
            iterator = iterator or self._experience_iterator
            loss_info = self._train(tf.constant(iterations), iterator, parallel_iterations)

            train_step_val = self.train_step.numpy()
            for trigger in self.triggers:
                trigger(train_step_val)

            return loss_info
        
    #@common.function(autograph=True) # Do not uncomment otherwise current_experience_sample is stored as a symbolic tensor 
    def _train(self, iterations, iterator, parallel_iterations):
        # Call run explicitly once to get loss info shape for autograph. Because the
        # for loop below will get converted to a `tf.while_loop` by autograph we
        # need the shape of loss info to be well defined.
        loss_info = self.single_train_step(iterator)

        for _ in tf.range(iterations - 1):
            tf.autograph.experimental.set_loop_options(parallel_iterations=parallel_iterations)
            loss_info = self.single_train_step(iterator)

        def _reduce_loss(loss):
            rank = None
            if isinstance(loss, tf.distribute.DistributedValues):
                # If loss is distributed get the rank from the first replica.
                rank = loss.values[0].shape.rank
            elif tf.is_tensor(loss):
                rank = loss.shape.rank
            axis = None
            if rank:
                axis = tuple(range(0, rank))
            return self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=axis)

        # We assume all data can be reduced in the loss_info. This means no
        # string dtypes are currently allowed as LossInfo Fields.
        reduced_loss_info = tf.nest.map_structure(_reduce_loss, loss_info)
        return reduced_loss_info
    
    def single_train_step(self, iterator):
        sample = next(iterator)
        self.current_experience_sample = sample ## Extra line to have access to current training experience sample for saving purposes

        if self._direct_sampling:
            experience, sample_info = sample, None
        elif self._use_reverb_v2:
            experience, sample_info = sample.data, sample.info
        else:
            experience, sample_info = sample

        if self.use_kwargs_in_agent_train:
            loss_info = self.strategy.run(
                self._agent.train,
                kwargs=experience,
                options=self._strategy_run_options)
        else:
            loss_info = self.strategy.run(
                self._agent.train,
                args=(experience,),
                options=self._strategy_run_options)

        if self.after_train_strategy_step_fn:
            if self.use_kwargs_in_agent_train:
                self.strategy.run(
                    self.after_train_strategy_step_fn,
                    kwargs=dict(experience=(experience, sample_info), loss_info=loss_info),
                    options=self._strategy_run_options)
            else:
                self.strategy.run(
                    self.after_train_strategy_step_fn,
                    args=((experience, sample_info), loss_info),
                    options=self._strategy_run_options,
                )

        return loss_info
    
    ## Methods that call corresponding superclass methods
    @property
    def train_step_numpy(self):
        return self.train_step.numpy()
    
    def _get_checkpoint_trigger(self, checkpoint_interval):
        if checkpoint_interval <= 0:
            return lambda _, force_trigger=False: None

        save_fn = lambda: self._checkpointer.save(self.train_step)
        return interval_trigger.IntervalTrigger(
            checkpoint_interval, save_fn, start=self.train_step.numpy()
        )


    # Superclass method can be called here since the agent's loss function is called and gradients are not updated
    def loss(self, experience_and_sample_info: Optional[ExperienceAndSampleInfo] = None,
    reduce_op: tf.distribute.ReduceOp = tf.distribute.ReduceOp.SUM) -> tf_agent.LossInfo:
        def _summary_record_if():
            return tf.math.equal(
                self.train_step % tf.constant(self.summary_interval), 0)

        with self.train_summary_writer.as_default(), common.soft_device_placement(), tf.compat.v2.summary.record_if(
            _summary_record_if
        ), self.strategy.scope():
            if experience_and_sample_info is None:
                sample = next(self._experience_iterator)
                if self._direct_sampling:
                    experience_and_sample_info = (sample, None)
                elif self._use_reverb_v2:
                    experience_and_sample_info = (sample.data, sample.info)
                else:
                    experience_and_sample_info = sample

        loss_info = self._loss(experience_and_sample_info, reduce_op)

        return loss_info
    
    # Use tf.config.experimental_run_functions_eagerly(True) if you want to
    # disable use of tf.function.
    @common.function(autograph=True)
    def _loss(
        self,
        experience_and_sample_info: ExperienceAndSampleInfo,
        reduce_op: tf.distribute.ReduceOp,
    ) -> tf_agent.LossInfo:
        (experience, sample_info) = experience_and_sample_info

        if self.use_kwargs_in_agent_train:
            loss_info = self.strategy.run(self._agent.loss, kwargs=experience)
        else:
            loss_info = self.strategy.run(self._agent.loss, args=(experience,))

        if self.after_train_strategy_step_fn:
            if self.use_kwargs_in_agent_train:
                self.strategy.run(
                    self.after_train_strategy_step_fn,
                    kwargs=dict(
                        experience=(experience, sample_info), loss_info=loss_info
                    ),
                    options=self._strategy_run_options,
                )
            else:
                self.strategy.run(
                    self.after_train_strategy_step_fn,
                    args=((experience, sample_info), loss_info),
                    options=self._strategy_run_options,
                )

        def _reduce_loss(loss):
            rank = None
            if isinstance(loss, tf.distribute.DistributedValues):
                rank = loss.values[0].shape.rank
            elif tf.is_tensor(loss):
                rank = loss.shape.rank
            axis = None
            if rank:
                axis = tuple(range(0, rank))
            return self.strategy.reduce(reduce_op, loss, axis=axis)

        # We assume all data can be reduced in the loss_info. This means no
        # string dtypes are currently allowed as LossInfo Fields.
        reduced_loss_info = tf.nest.map_structure(_reduce_loss, loss_info)
        return reduced_loss_info
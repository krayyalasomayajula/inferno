from ...utils.train_utils import Frequency
from ...utils.exceptions import assert_, FrequencyValueError
from .base import Callback
from ...utils import torch_utils as thu
from ...utils.logging_utils import ScalarLoggingMixin


class ExtraMetric(ScalarLoggingMixin, Callback):
    """Evaluates and logs an extra metric"""
    def __init__(self, metric, frequency, name=None):
        super(ExtraMetric, self).__init__()
        # Privates
        self._eval_every = None
        assert callable(metric)
        self._metric = metric
        self._name = type(metric).__name__ if name is None else name

        # Publics
        self.eval_every = frequency

    @property
    def eval_every(self):
        return self._eval_every

    @eval_every.setter
    def eval_every(self, value):
        self._eval_every = Frequency.build_from(value)
        assert_(self._eval_every.is_consistent,
                "Frequency is not consistent.",
                FrequencyValueError)

    @property
    def eval_now(self):
        return self.eval_every.match(iteration_count=self.trainer.iteration_count,
                                     epoch_count=self.trainer.epoch_count,
                                     persistent=True, match_zero=True)

    def eval(self, prediction, target):
        error = self._metric(thu.unwrap(prediction, to_cpu=False),
                             thu.unwrap(target, to_cpu=False))
        self.save_scalar(self._name, error, self.trainer)

    def end_of_training_iteration(self, **_):
        eval_now = self.eval_now
        if eval_now:
            self.eval(self.trainer.get_state('training_prediction'),
                      self.trainer.get_state('training_target'))

    def end_of_validation_iteration(self, **_):
        self.eval(self.trainer.get_state('validation_prediction'),
                  self.trainer.get_state('validation_target'))
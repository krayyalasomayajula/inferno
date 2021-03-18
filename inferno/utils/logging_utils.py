import torch
from .train_utils import AverageMeter


class ScalarLoggingMixin:
    def __init__(self):
        super(ScalarLoggingMixin, self).__init__()
        self.validation_averages = dict()
        self.current_validation_iteration = None
        self.registered_states = set()

    def save_scalar(self, name, value, trainer=None):
        if trainer is None:
            assert hasattr(self, 'trainer')
            trainer = self.trainer
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().clone()
        # add prefix ('training' or 'validation') to name
        name = self.add_prefix(name, trainer)
        if name not in self.registered_states:
            self.observe_state(name, trainer)
        if trainer.model.training:  # training
            trainer.update_state(name, value)
        else:  # validation
            # check if it is a new validation run
            if self.current_validation_iteration != trainer._last_validated_at_iteration:
                self.current_validation_iteration = trainer._last_validated_at_iteration
                self.validation_averages = dict()

            # check if average meter for name has already been initialized this run
            if name not in self.validation_averages:
                self.validation_averages[name] = AverageMeter()

            self.validation_averages[name].update(value)
            trainer.update_state(name, self.validation_averages[name].avg)

    @staticmethod
    def observe_state(name, trainer):
        assert hasattr(trainer, 'logger')
        logger = trainer.logger
        assert hasattr(logger, 'observe_state')
        time = 'training' if trainer.model.training else 'validation'
        logger.observe_state(name, time)

    @staticmethod
    def add_prefix(name, trainer):
        if trainer.model.training:
            return 'training_' + name
        else:
            return 'validation_' + name
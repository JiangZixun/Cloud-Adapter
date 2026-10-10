"""Resolve user-facing epochs into internal optimizer-update counters."""

import math


def resolve_epoch_schedule(config, train_samples):
    if 'max_epochs' not in config:
        return
    for name, value in [('max_epochs', config['max_epochs']),
                        ('val_interval_epochs', config.get('val_interval_epochs', 1)),
                        ('train_samples', train_samples)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if config.get('grad_accum_steps', 1) != 1:
        raise ValueError('Epoch training currently requires grad_accum_steps=1')
    batch = config['batch_size'] * config.get('world_size', 1)
    if batch < 1:
        raise ValueError('Global physical batch size must be positive')
    steps = math.ceil(train_samples / batch)
    config.update(training_unit='epoch', steps_per_epoch=steps,
                  train_samples=train_samples, epoch_samples=steps * batch,
                  epoch_padding_samples=steps * batch - train_samples,
                  max_iters=config['max_epochs'] * steps,
                  val_interval=config.get('val_interval_epochs', 1) * steps)


def epoch_context(iteration, config):
    if 'max_epochs' not in config:
        return {}
    steps = config['steps_per_epoch']
    return dict(epoch=iteration / steps, epoch_index=math.ceil(iteration / steps),
                step_in_epoch=(iteration - 1) % steps + 1 if iteration else 0,
                steps_per_epoch=steps)


def validate_resume_epochs(saved, current):
    if ('max_epochs' in saved) != ('max_epochs' in current):
        raise ValueError('Checkpoint training unit differs; start a new run')
    if 'max_epochs' in current:
        for name in ('steps_per_epoch', 'train_samples', 'epoch_padding_samples', 'val_interval_epochs'):
            if saved.get(name, 1 if name == 'val_interval_epochs' else None) != current.get(name, 1 if name == 'val_interval_epochs' else None):
                raise ValueError(f'Checkpoint configuration mismatch: {name}')

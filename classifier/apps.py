from django.apps import AppConfig


class ClassifierConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'classifier'

    def ready(self):
        """Schedule cleanup of training runs interrupted by a server shutdown."""
        from django.db.models.signals import post_migrate
        post_migrate.connect(_cleanup_stale_runs, sender=self)


def _cleanup_stale_runs(sender, **kwargs):
    """Mark any in-progress training runs as failed after a server restart."""
    import logging
    logger = logging.getLogger(__name__)

    try:
        from classifier.models import TrainingRun
        stale_statuses = ['pending', 'running', 'training', 'evaluating']
        stale = TrainingRun.objects.filter(status__in=stale_statuses)
        count = stale.update(
            status='failed',
            error_message='Server was shut down while training was in progress.',
        )
        if count:
            logger.warning(
                "Marked %d stale training run(s) as failed (server restart cleanup).",
                count,
            )
    except Exception:
        pass

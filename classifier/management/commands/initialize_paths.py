"""
Management command to initialize WasteX storage paths during installation.

Usage:
    python manage.py initialize_paths \
        --media-root "D:/WasteX/media" \
        --datasets-root "D:/WasteX/datasets" \
        --models-root "D:/WasteX/models"
"""

import os
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db.utils import IntegrityError

from classifier.models import AppSettings


class Command(BaseCommand):
    """Initialize storage paths during WasteX installation."""

    help = 'Initialize WasteX storage paths during installation'

    def add_arguments(self, parser):
        """Add command-line arguments."""
        parser.add_argument(
            '--media-root',
            type=str,
            required=True,
            help='Path to media folder for uploaded images'
        )
        parser.add_argument(
            '--datasets-root',
            type=str,
            required=True,
            help='Path to datasets folder for training data'
        )
        parser.add_argument(
            '--models-root',
            type=str,
            required=True,
            help='Path to models folder for trained models'
        )

    def handle(self, *args, **options):
        """Execute the initialization."""
        self.stdout.write(self.style.SUCCESS('╔════════════════════════════════════════╗'))
        self.stdout.write(self.style.SUCCESS('║  WasteX Storage Path Initialization    ║'))
        self.stdout.write(self.style.SUCCESS('╚════════════════════════════════════════╝'))
        self.stdout.write('')

        paths = {
            'media_root': {
                'path': options['media_root'],
                'description': 'Uploaded images',
            },
            'datasets_root': {
                'path': options['datasets_root'],
                'description': 'Training datasets',
            },
            'models_root': {
                'path': options['models_root'],
                'description': 'ML models',
            },
        }

        errors = []
        successes = []

        for key, config in paths.items():
            path_str = config['path']
            description = config['description']

            self.stdout.write(f'\n📁 Configuring {description}...')
            self.stdout.write(f'   Path: {path_str}')

            try:
                # Convert to Path object
                path = Path(path_str)

                # Create folder if it doesn't exist
                self.stdout.write(f'   → Creating folder...', ending='')
                path.mkdir(parents=True, exist_ok=True)
                self.stdout.write(self.style.SUCCESS(' ✓'))

                # Check write permission
                self.stdout.write(f'   → Checking permissions...', ending='')
                test_file = path / '.wastex_write_test'
                try:
                    test_file.write_text('WasteX write test')
                    test_file.unlink()
                except PermissionError:
                    raise CommandError(
                        f'No write permission for {path}'
                    )
                self.stdout.write(self.style.SUCCESS(' ✓'))

                # Check disk space
                self.stdout.write(f'   → Checking disk space...', ending='')
                import shutil
                _, _, free = shutil.disk_usage(path)
                free_gb = free / (1024**3)

                min_space_gb = 50
                if free_gb < min_space_gb:
                    raise CommandError(
                        f'Insufficient disk space. '
                        f'Required: {min_space_gb}GB, Available: {free_gb:.1f}GB'
                    )
                self.stdout.write(self.style.SUCCESS(f' ✓ ({free_gb:.1f}GB free)'))

                # Save to database
                self.stdout.write(f'   → Saving configuration...', ending='')
                AppSettings.set(
                    key=key,
                    value=str(path.absolute()),
                    description=description
                )
                self.stdout.write(self.style.SUCCESS(' ✓'))

                successes.append(f'{key}: {path.absolute()}')

            except CommandError as e:
                errors.append(f'{key}: {str(e)}')
                self.stdout.write(self.style.ERROR(f' ✗ {str(e)}'))
            except Exception as e:
                errors.append(f'{key}: {str(e)}')
                self.stdout.write(self.style.ERROR(f' ✗ {str(e)}'))

        # Print summary
        self.stdout.write('\n')
        self.stdout.write(self.style.SUCCESS('╔════════════════════════════════════════╗'))
        self.stdout.write(self.style.SUCCESS('║  Configuration Summary                 ║'))
        self.stdout.write(self.style.SUCCESS('╚════════════════════════════════════════╝'))

        if successes:
            self.stdout.write(self.style.SUCCESS(f'\n✅ Successfully configured {len(successes)} paths:'))
            for success in successes:
                self.stdout.write(f'   • {success}')

        if errors:
            self.stdout.write(self.style.ERROR(f'\n❌ Errors encountered ({len(errors)}):'))
            for error in errors:
                self.stdout.write(f'   • {error}')
            raise CommandError('Initialization failed. Please check errors above.')

        self.stdout.write(self.style.SUCCESS('\n✅ WasteX is ready to use!'))
        self.stdout.write(self.style.WARNING('\n⚠️  NOTE: These paths are set during installation and cannot be changed.'))
        self.stdout.write(self.style.WARNING('   To use different paths, please reinstall WasteX.\n'))

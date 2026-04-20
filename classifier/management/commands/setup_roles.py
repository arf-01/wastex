from django.core.management.base import BaseCommand
from django.contrib.auth.models import Group

class Command(BaseCommand):
    help = 'Create default roles (Groups) for Edge and Master users'

    def handle(self, *args, **options):
        edge_group, created = Group.objects.get_or_create(name='EdgeUsers')
        if created:
            self.stdout.write(self.style.SUCCESS('Successfully created EdgeUsers group.'))
        else:
            self.stdout.write('EdgeUsers group already exists.')

        master_group, created = Group.objects.get_or_create(name='MasterUsers')
        if created:
            self.stdout.write(self.style.SUCCESS('Successfully created MasterUsers group.'))
        else:
            self.stdout.write('MasterUsers group already exists.')

        self.stdout.write(self.style.SUCCESS('\nRoles setup complete. You can now assign users to these groups in the Django Admin.'))

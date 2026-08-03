import django, os
os.environ['DJANGO_SETTINGS_MODULE'] = 'wastex.settings'
django.setup()

from rest_framework.authtoken.models import Token
from django.contrib.auth.models import User

u = User.objects.get(username='edge')
t, created = Token.objects.get_or_create(user=u)
print(f'Token: {t.key}')

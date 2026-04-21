"""
Management command to seed WasteX with its two canonical user accounts.

Usage (called by install.bat — not meant for end-users):

    python manage.py seed_installation \\
        --edge-password  "OperatorChosenPass" \\
        --master-password "CompanyChosenPass"

Behaviour
---------
* Creates the ``EdgeUsers`` and ``MasterUsers`` groups if they do not exist.
* Creates the ``edge`` account (EdgeUsers group) and the ``master`` account
  (MasterUsers group, staff flag set so Django Admin is accessible).
* Fully **idempotent**: if either user already exists the command skips that
  user and prints a notice.  Existing passwords are never overwritten.
* Exits with a non-zero code on any unexpected error so ``install.bat`` can
  detect the failure.
"""

from django.contrib.auth.models import Group, User
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Create EdgeUsers/MasterUsers groups and seed the edge + master accounts."

    # ── Argument definition ──────────────────────────────────────────────────

    def add_arguments(self, parser):
        parser.add_argument(
            "--edge-password",
            type=str,
            required=True,
            metavar="PASSWORD",
            help="Password for the 'edge' operator account.",
        )
        parser.add_argument(
            "--master-password",
            type=str,
            required=True,
            metavar="PASSWORD",
            help="Password for the 'master' (WasteX company) account.",
        )

    # ── Entry point ──────────────────────────────────────────────────────────

    def handle(self, *args, **options):
        self._header()

        # 1. Groups (idempotent)
        edge_group, created = Group.objects.get_or_create(name="EdgeUsers")
        self._log_group("EdgeUsers", created)

        master_group, created = Group.objects.get_or_create(name="MasterUsers")
        self._log_group("MasterUsers", created)

        self.stdout.write("")

        # 2. Accounts
        self._create_user(
            username="edge",
            password=options["edge_password"],
            group=edge_group,
            label="Edge User — daily site operator",
            is_staff=False,
        )
        self._create_user(
            username="master",
            password=options["master_password"],
            group=master_group,
            label="Master User — WasteX company / retraining",
            is_staff=True,
        )

        self._footer()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _create_user(
        self,
        username: str,
        password: str,
        group: Group,
        label: str,
        is_staff: bool,
    ) -> None:
        if User.objects.filter(username=username).exists():
            self.stdout.write(
                self.style.WARNING(
                    f"  [!] '{username}' already exists -- skipped (password unchanged)."
                )
            )
            return

        user = User.objects.create_user(
            username=username,
            password=password,
            is_staff=is_staff,
        )
        user.groups.add(group)
        self.stdout.write(
            self.style.SUCCESS(f"  [+] Created '{username}'  ({label})")
        )

    def _log_group(self, name: str, created: bool) -> None:
        if created:
            self.stdout.write(self.style.SUCCESS(f"  [+] Group '{name}' created."))
        else:
            self.stdout.write(f"  [=] Group '{name}' already exists.")

    # ── Banners ──────────────────────────────────────────────────────────────

    def _header(self) -> None:
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("=========================================="))
        self.stdout.write(self.style.SUCCESS("   WasteX -- Seeding User Accounts"))
        self.stdout.write(self.style.SUCCESS("=========================================="))
        self.stdout.write("")

    def _footer(self) -> None:
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("=========================================="))
        self.stdout.write(self.style.SUCCESS("   Installation seeding complete."))
        self.stdout.write(self.style.SUCCESS("=========================================="))
        self.stdout.write("")
        self.stdout.write("  Accounts ready:")
        self.stdout.write("    edge    -> dashboard, bin monitoring, OOD review")
        self.stdout.write("    master  -> dataset versioning, retraining, model promotion")
        self.stdout.write("")

# Generated by Django 3.2.23 on 2024-01-31 04:28

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('outfit', '0002_outfitinfo_data'),
    ]

    operations = [
        migrations.RenameField(
            model_name='outfitinfo',
            old_name='data',
            new_name='image_data',
        ),
    ]
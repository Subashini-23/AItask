# Generated by Django 5.1.5 on 2025-02-10 21:40

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('productapp', '0002_orderitem'),
    ]

    operations = [
        migrations.DeleteModel(
            name='OrderItem',
        ),
    ]

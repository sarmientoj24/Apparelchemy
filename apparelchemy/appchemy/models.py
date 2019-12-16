from django.db import models

# Create your models here.
class Apparel_Test(models.Model):
    name = models.TextField(blank=True, null=True, default=None)
    image_id = models.CharField(max_length=20, blank=True, null=True, default=None)
    old_name = models.CharField(max_length=50, blank=True, null=True, default=None)
    cluster_id = models.PositiveSmallIntegerField(blank=True, null=True, default=None)
    latent_code1 = models.FloatField(blank=True, null=True, default=None)
    latent_code2 = models.FloatField(blank=True, null=True, default=None)
    latent_code3 = models.FloatField(blank=True, null=True, default=None)
    latent_code4 = models.FloatField(blank=True, null=True, default=None)
    latent_code5 = models.FloatField(blank=True, null=True, default=None)

    def __str__(self):
        return self.name

class Apparel_Dummy(models.Model):
    subcategory = models.CharField(max_length=20, null=True, default=None)
    name = models.TextField(blank=True, null=True, default=None)
    image_id = models.CharField(max_length=20, blank=True, null=True, default=None)
    old_name = models.CharField(max_length=50, blank=True, null=True, default=None)
    cluster_id = models.PositiveSmallIntegerField(blank=True, null=True, default=None)
    latent_code1 = models.FloatField(blank=True, null=True, default=None)
    latent_code2 = models.FloatField(blank=True, null=True, default=None)
    latent_code3 = models.FloatField(blank=True, null=True, default=None)
    latent_code4 = models.FloatField(blank=True, null=True, default=None)
    latent_code5 = models.FloatField(blank=True, null=True, default=None)

    def __str__(self):
        return self.name

class Cluster_Test(models.Model):
    cluster_id = models.PositiveSmallIntegerField()
    latent_code1 = models.FloatField()
    latent_code2 = models.FloatField()
    latent_code3 = models.FloatField()
    latent_code4 = models.FloatField()
    latent_code5 = models.FloatField()

    def __str__(self):
        return self.cluster_id
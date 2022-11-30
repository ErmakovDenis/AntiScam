from django.db import models

class Worker(models.Model):
    class Meta:
        verbose_name ="Worker"
        verbose_name_plural = "Worker"

    name = models.CharField(max_length=20, blank= False)
    second_name = models.CharField(max_length=35,blank=False)


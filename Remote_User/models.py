from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    id = models.AutoField(primary_key=True)

    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    address= models.CharField(max_length=300)
    gender= models.CharField(max_length=30)

class false_data_injection_attack_detection(models.Model):

    URLs= models.CharField(max_length=30000)
    Headline= models.CharField(max_length=30000)
    Body= models.CharField(max_length=30000)
    Prediction= models.CharField(max_length=3000)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)




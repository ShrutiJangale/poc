# Create your models here.
from django.db import models

class UploadedVideo(models.Model):
    """
    A simple model to store uploaded video files.
    This allows Django to manage the files and makes it easy
    to reference them later.
    """
    video = models.FileField(upload_to='videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.video.name

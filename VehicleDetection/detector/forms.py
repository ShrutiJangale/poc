from django import forms
from .models import UploadedVideo

class VideoUploadForm(forms.ModelForm):
    """
    A form for uploading video files, linked to the UploadedVideo model.
    """
    class Meta:
        model = UploadedVideo
        fields = ['video']
        widgets = {
            'video': forms.FileInput(attrs={'class': 'hidden', 'accept': 'video/*'})
        }

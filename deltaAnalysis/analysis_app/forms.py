from django import forms

class UploadForm(forms.Form):
    procurement_sheet = forms.FileField(label="Procurement Sheet")
    trueup_sheet = forms.FileField(label="True Up Sheet")

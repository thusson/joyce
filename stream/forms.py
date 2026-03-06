from django import forms

from .models import Post, Tag


class PostForm(forms.ModelForm):
    tags = forms.ModelMultipleChoiceField(
        queryset=Tag.objects.all(),
        required=False,
        widget=forms.CheckboxSelectMultiple,
    )

    class Meta:
        model = Post
        fields = ["title", "content", "priority", "tags"]
        widgets = {
            "title": forms.TextInput(attrs={"class": "form-input"}),
            "content": forms.Textarea(attrs={"rows": 15, "class": "form-input"}),
            "priority": forms.Select(attrs={"class": "form-input"}),
        }

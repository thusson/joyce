from django import forms
from django.contrib.auth.models import User

from .models import Comment, Image, Post, Tag, UserProfile


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


class CommentForm(forms.ModelForm):
    class Meta:
        model = Comment
        fields = ["content"]
        widgets = {
            "content": forms.Textarea(attrs={"rows": 3, "class": "form-input", "placeholder": "Write a comment..."}),
        }
        labels = {
            "content": "",
        }


class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ["image", "alt_text"]
        widgets = {
            "image": forms.ClearableFileInput(attrs={"class": "form-input", "accept": "image/*"}),
            "alt_text": forms.TextInput(attrs={"class": "form-input", "placeholder": "Image description (optional)"}),
        }
        labels = {
            "alt_text": "Alt text",
        }


class TagForm(forms.ModelForm):
    class Meta:
        model = Tag
        fields = ["name", "slug"]
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-input"}),
            "slug": forms.TextInput(attrs={"class": "form-input"}),
        }


class UserRoleForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ["role"]
        widgets = {
            "role": forms.Select(attrs={"class": "form-input"}),
        }


class UserCreateForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput(attrs={"class": "form-input"}))
    role = forms.ChoiceField(
        choices=UserProfile.Role.choices,
        initial=UserProfile.Role.VIEWER,
        widget=forms.Select(attrs={"class": "form-input"}),
    )

    class Meta:
        model = User
        fields = ["username", "first_name", "last_name", "email"]
        widgets = {
            "username": forms.TextInput(attrs={"class": "form-input"}),
            "first_name": forms.TextInput(attrs={"class": "form-input"}),
            "last_name": forms.TextInput(attrs={"class": "form-input"}),
            "email": forms.EmailInput(attrs={"class": "form-input"}),
        }

from django.contrib import admin

from .models import Comment, Image, Post, ReadStatus, Tag, UserProfile


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ("name", "slug")
    prepopulated_fields = {"slug": ("name",)}


@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    list_display = ("title", "author", "priority", "created_at")
    list_filter = ("priority", "tags", "created_at")
    search_fields = ("title", "content")
    filter_horizontal = ("tags",)


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "role")
    list_filter = ("role",)


@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    list_display = ("author", "post", "created_at")
    list_filter = ("created_at",)
    search_fields = ("content",)


@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ("__str__", "uploader", "uploaded_at")
    list_filter = ("uploaded_at",)


@admin.register(ReadStatus)
class ReadStatusAdmin(admin.ModelAdmin):
    list_display = ("user", "post", "read_at")

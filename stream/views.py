import functools

import markdown
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Q
from django.http import HttpResponseForbidden
from django.shortcuts import get_object_or_404, redirect, render

from django.db import models as db_models

from .forms import CommentForm, ImageUploadForm, PostForm, TagForm, UserCreateForm, UserRoleForm
from .models import Comment, Image, Post, ReadStatus, Tag, UserProfile


def _get_profile(user):
    profile, _ = UserProfile.objects.get_or_create(user=user)
    return profile


@login_required
def post_list(request):
    posts = Post.objects.select_related("author").prefetch_related("tags")

    tag_slug = request.GET.get("tag")
    active_tag = None
    if tag_slug:
        active_tag = get_object_or_404(Tag, slug=tag_slug)
        posts = posts.filter(tags=active_tag)

    show_unread = request.GET.get("unread")
    if show_unread:
        read_post_ids = ReadStatus.objects.filter(user=request.user).values_list(
            "post_id", flat=True
        )
        posts = posts.exclude(pk__in=read_post_ids)

    read_post_ids = set(
        ReadStatus.objects.filter(user=request.user).values_list("post_id", flat=True)
    )

    tags = Tag.objects.all()
    profile = _get_profile(request.user)

    return render(
        request,
        "stream/post_list.html",
        {
            "posts": posts,
            "tags": tags,
            "active_tag": active_tag,
            "show_unread": show_unread,
            "read_post_ids": read_post_ids,
            "profile": profile,
        },
    )


@login_required
def post_detail(request, pk):
    post = get_object_or_404(
        Post.objects.select_related("author").prefetch_related("tags"), pk=pk
    )

    ReadStatus.objects.get_or_create(user=request.user, post=post)

    if request.method == "POST":
        comment_form = CommentForm(request.POST)
        if comment_form.is_valid():
            comment = comment_form.save(commit=False)
            comment.post = post
            comment.author = request.user
            comment.save()
            return redirect("post_detail", pk=post.pk)
    else:
        comment_form = CommentForm()

    md = markdown.Markdown(extensions=["fenced_code", "tables", "nl2br"])
    content_html = md.convert(post.content)
    profile = _get_profile(request.user)
    comments = post.comments.select_related("author")

    return render(
        request,
        "stream/post_detail.html",
        {
            "post": post,
            "content_html": content_html,
            "profile": profile,
            "comments": comments,
            "comment_form": comment_form,
        },
    )


@login_required
def post_create(request):
    profile = _get_profile(request.user)
    if not profile.can_create_post:
        return HttpResponseForbidden("You do not have permission to create posts.")

    if request.method == "POST":
        form = PostForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user
            post.save()
            form.save_m2m()
            return redirect("post_detail", pk=post.pk)
    else:
        form = PostForm()

    return render(request, "stream/post_form.html", {"form": form, "editing": False})


@login_required
def post_edit(request, pk):
    post = get_object_or_404(Post, pk=pk)
    profile = _get_profile(request.user)

    if not (profile.is_admin or (profile.is_contributor and post.author == request.user)):
        return HttpResponseForbidden("You do not have permission to edit this post.")

    if request.method == "POST":
        form = PostForm(request.POST, instance=post)
        if form.is_valid():
            form.save()
            return redirect("post_detail", pk=post.pk)
    else:
        form = PostForm(instance=post)

    return render(request, "stream/post_form.html", {"form": form, "editing": True, "post": post})


@login_required
def post_delete(request, pk):
    post = get_object_or_404(Post, pk=pk)
    profile = _get_profile(request.user)

    if not (profile.is_admin or (profile.is_contributor and post.author == request.user)):
        return HttpResponseForbidden("You do not have permission to delete this post.")

    if request.method == "POST":
        post.delete()
        return redirect("post_list")

    return render(request, "stream/post_confirm_delete.html", {"post": post})


@login_required
def mark_unread(request, pk):
    if request.method == "POST":
        ReadStatus.objects.filter(user=request.user, post_id=pk).delete()
    return redirect("post_list")


@login_required
def search(request):
    query = request.GET.get("q", "").strip()
    posts = Post.objects.none()

    if query:
        posts = (
            Post.objects.select_related("author")
            .prefetch_related("tags")
            .filter(Q(title__icontains=query) | Q(content__icontains=query))
        )

    return render(request, "stream/search.html", {
        "query": query,
        "posts": posts,
    })


@login_required
def image_upload(request):
    profile = _get_profile(request.user)
    if not profile.can_create_post:
        return HttpResponseForbidden("You do not have permission to upload images.")

    uploaded_image = None
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.save(commit=False)
            image.uploader = request.user
            image.save()
            uploaded_image = image
            form = ImageUploadForm()
    else:
        form = ImageUploadForm()

    recent_images = Image.objects.filter(uploader=request.user)[:10]

    return render(request, "stream/image_upload.html", {
        "form": form,
        "uploaded_image": uploaded_image,
        "recent_images": recent_images,
    })


@login_required
def comment_delete(request, pk):
    comment = get_object_or_404(Comment.objects.select_related("post"), pk=pk)
    profile = _get_profile(request.user)

    if not (profile.is_admin or comment.author == request.user):
        return HttpResponseForbidden("You do not have permission to delete this comment.")

    post_pk = comment.post.pk
    if request.method == "POST":
        comment.delete()
        return redirect("post_detail", pk=post_pk)

    return render(request, "stream/comment_confirm_delete.html", {
        "comment": comment,
    })


# ---------------------------------------------------------------------------
# Admin panel views
# ---------------------------------------------------------------------------

def admin_required(view_func):
    @functools.wraps(view_func)
    @login_required
    def wrapper(request, *args, **kwargs):
        profile = _get_profile(request.user)
        if not profile.is_admin:
            return HttpResponseForbidden("Admin access required.")
        return view_func(request, *args, **kwargs)
    return wrapper


@admin_required
def admin_dashboard(request):
    return render(request, "stream/admin/dashboard.html", {
        "post_count": Post.objects.count(),
        "tag_count": Tag.objects.count(),
        "user_count": User.objects.count(),
    })


# -- Tag management --

@admin_required
def admin_tag_list(request):
    tags = Tag.objects.annotate(post_count=db_models.Count("posts"))
    return render(request, "stream/admin/tag_list.html", {"tags": tags})


@admin_required
def admin_tag_create(request):
    if request.method == "POST":
        form = TagForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Tag created.")
            return redirect("admin_tag_list")
    else:
        form = TagForm()
    return render(request, "stream/admin/tag_form.html", {"form": form, "editing": False})


@admin_required
def admin_tag_edit(request, pk):
    tag = get_object_or_404(Tag, pk=pk)
    if request.method == "POST":
        form = TagForm(request.POST, instance=tag)
        if form.is_valid():
            form.save()
            messages.success(request, "Tag updated.")
            return redirect("admin_tag_list")
    else:
        form = TagForm(instance=tag)
    return render(request, "stream/admin/tag_form.html", {"form": form, "editing": True, "tag": tag})


@admin_required
def admin_tag_delete(request, pk):
    tag = get_object_or_404(Tag, pk=pk)
    if request.method == "POST":
        tag.delete()
        messages.success(request, "Tag deleted.")
        return redirect("admin_tag_list")
    return render(request, "stream/admin/tag_confirm_delete.html", {"tag": tag})


# -- User management --

@admin_required
def admin_user_list(request):
    users = User.objects.select_related("profile").order_by("username")
    return render(request, "stream/admin/user_list.html", {"users": users})


@admin_required
def admin_user_create(request):
    if request.method == "POST":
        form = UserCreateForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data["password"])
            user.save()
            UserProfile.objects.create(user=user, role=form.cleaned_data["role"])
            messages.success(request, f"User '{user.username}' created.")
            return redirect("admin_user_list")
    else:
        form = UserCreateForm()
    return render(request, "stream/admin/user_form.html", {"form": form})


@admin_required
def admin_user_edit(request, pk):
    target_user = get_object_or_404(User, pk=pk)
    profile = _get_profile(target_user)
    if request.method == "POST":
        role_form = UserRoleForm(request.POST, instance=profile)
        if role_form.is_valid():
            role_form.save()
            messages.success(request, f"Role updated for '{target_user.username}'.")
            return redirect("admin_user_list")
    else:
        role_form = UserRoleForm(instance=profile)
    return render(request, "stream/admin/user_edit.html", {
        "target_user": target_user,
        "role_form": role_form,
    })


@admin_required
def admin_user_toggle_active(request, pk):
    target_user = get_object_or_404(User, pk=pk)
    if request.method == "POST":
        if target_user == request.user:
            messages.error(request, "You cannot deactivate yourself.")
        else:
            target_user.is_active = not target_user.is_active
            target_user.save()
            status = "activated" if target_user.is_active else "deactivated"
            messages.success(request, f"User '{target_user.username}' {status}.")
    return redirect("admin_user_list")


# -- Post management --

@admin_required
def admin_post_list(request):
    posts = Post.objects.select_related("author").prefetch_related("tags")
    return render(request, "stream/admin/post_list.html", {"posts": posts})

import markdown
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from .forms import PostForm
from .models import Post, ReadStatus, Tag, UserProfile


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

    md = markdown.Markdown(extensions=["fenced_code", "tables", "nl2br"])
    content_html = md.convert(post.content)
    profile = _get_profile(request.user)

    return render(
        request,
        "stream/post_detail.html",
        {
            "post": post,
            "content_html": content_html,
            "profile": profile,
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

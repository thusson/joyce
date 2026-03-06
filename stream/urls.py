from django.urls import path

from . import views

urlpatterns = [
    path("", views.post_list, name="post_list"),
    path("post/<int:pk>/", views.post_detail, name="post_detail"),
    path("post/new/", views.post_create, name="post_create"),
    path("post/<int:pk>/edit/", views.post_edit, name="post_edit"),
    path("post/<int:pk>/delete/", views.post_delete, name="post_delete"),
    path("post/<int:pk>/mark-unread/", views.mark_unread, name="mark_unread"),
    path("comment/<int:pk>/delete/", views.comment_delete, name="comment_delete"),

    # Admin panel
    path("manage/", views.admin_dashboard, name="admin_dashboard"),
    path("manage/tags/", views.admin_tag_list, name="admin_tag_list"),
    path("manage/tags/new/", views.admin_tag_create, name="admin_tag_create"),
    path("manage/tags/<int:pk>/edit/", views.admin_tag_edit, name="admin_tag_edit"),
    path("manage/tags/<int:pk>/delete/", views.admin_tag_delete, name="admin_tag_delete"),
    path("manage/users/", views.admin_user_list, name="admin_user_list"),
    path("manage/users/new/", views.admin_user_create, name="admin_user_create"),
    path("manage/users/<int:pk>/edit/", views.admin_user_edit, name="admin_user_edit"),
    path("manage/users/<int:pk>/toggle-active/", views.admin_user_toggle_active, name="admin_user_toggle_active"),
    path("manage/posts/", views.admin_post_list, name="admin_post_list"),
]

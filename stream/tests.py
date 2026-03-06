from django.contrib.auth.models import User
from django.test import TestCase, Client
from django.urls import reverse

from .models import Post, Tag, ReadStatus, UserProfile


class PostModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="testpass123")
        self.profile = UserProfile.objects.create(user=self.user, role=UserProfile.Role.CONTRIBUTOR)
        self.tag = Tag.objects.create(name="Test Tag", slug="test-tag")
        self.post = Post.objects.create(
            title="Test Post",
            content="# Hello\n\nSome **bold** text.",
            author=self.user,
            priority=Post.Priority.INTERNAL,
        )
        self.post.tags.add(self.tag)

    def test_post_str(self):
        self.assertEqual(str(self.post), "Test Post")

    def test_tag_str(self):
        self.assertEqual(str(self.tag), "Test Tag")

    def test_default_ordering(self):
        post2 = Post.objects.create(title="Second Post", content="text", author=self.user)
        posts = list(Post.objects.all())
        self.assertEqual(posts[0], post2)
        self.assertEqual(posts[1], self.post)

    def test_profile_permissions(self):
        self.assertTrue(self.profile.can_create_post)
        self.assertTrue(self.profile.is_contributor)
        self.assertFalse(self.profile.is_admin)

        viewer = User.objects.create_user(username="viewer", password="testpass123")
        viewer_profile = UserProfile.objects.create(user=viewer, role=UserProfile.Role.VIEWER)
        self.assertFalse(viewer_profile.can_create_post)


class ViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username="testuser", password="testpass123")
        self.profile = UserProfile.objects.create(user=self.user, role=UserProfile.Role.CONTRIBUTOR)
        self.tag = Tag.objects.create(name="News", slug="news")
        self.post = Post.objects.create(
            title="Test Post",
            content="Some content",
            author=self.user,
            priority=Post.Priority.INTERNAL,
        )
        self.post.tags.add(self.tag)

    def test_login_required(self):
        response = self.client.get(reverse("post_list"))
        self.assertEqual(response.status_code, 302)
        self.assertIn("login", response.url)

    def test_post_list(self):
        self.client.login(username="testuser", password="testpass123")
        response = self.client.get(reverse("post_list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test Post")

    def test_post_detail_marks_read(self):
        self.client.login(username="testuser", password="testpass123")
        self.assertFalse(ReadStatus.objects.filter(user=self.user, post=self.post).exists())
        response = self.client.get(reverse("post_detail", args=[self.post.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(ReadStatus.objects.filter(user=self.user, post=self.post).exists())

    def test_filter_by_tag(self):
        self.client.login(username="testuser", password="testpass123")
        response = self.client.get(reverse("post_list") + "?tag=news")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test Post")

    def test_filter_unread(self):
        self.client.login(username="testuser", password="testpass123")
        response = self.client.get(reverse("post_list") + "?unread=1")
        self.assertContains(response, "Test Post")

        ReadStatus.objects.create(user=self.user, post=self.post)
        response = self.client.get(reverse("post_list") + "?unread=1")
        self.assertNotContains(response, "Test Post")

    def test_create_post(self):
        self.client.login(username="testuser", password="testpass123")
        response = self.client.post(reverse("post_create"), {
            "title": "New Post",
            "content": "New content",
            "priority": "internal",
            "tags": [self.tag.pk],
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(Post.objects.filter(title="New Post").exists())

    def test_viewer_cannot_create(self):
        viewer = User.objects.create_user(username="viewer", password="testpass123")
        UserProfile.objects.create(user=viewer, role=UserProfile.Role.VIEWER)
        self.client.login(username="viewer", password="testpass123")
        response = self.client.get(reverse("post_create"))
        self.assertEqual(response.status_code, 403)

    def test_edit_post(self):
        self.client.login(username="testuser", password="testpass123")
        response = self.client.post(reverse("post_edit", args=[self.post.pk]), {
            "title": "Updated Title",
            "content": "Updated content",
            "priority": "internal",
        })
        self.assertEqual(response.status_code, 302)
        self.post.refresh_from_db()
        self.assertEqual(self.post.title, "Updated Title")

    def test_delete_post(self):
        self.client.login(username="testuser", password="testpass123")
        response = self.client.post(reverse("post_delete", args=[self.post.pk]))
        self.assertEqual(response.status_code, 302)
        self.assertFalse(Post.objects.filter(pk=self.post.pk).exists())

    def test_mark_unread(self):
        self.client.login(username="testuser", password="testpass123")
        ReadStatus.objects.create(user=self.user, post=self.post)
        self.client.post(reverse("mark_unread", args=[self.post.pk]))
        self.assertFalse(ReadStatus.objects.filter(user=self.user, post=self.post).exists())


class AdminPanelTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.admin_user = User.objects.create_user(username="admin", password="adminpass123")
        UserProfile.objects.create(user=self.admin_user, role=UserProfile.Role.ADMIN)
        self.contributor = User.objects.create_user(username="contributor", password="testpass123")
        UserProfile.objects.create(user=self.contributor, role=UserProfile.Role.CONTRIBUTOR)
        self.tag = Tag.objects.create(name="News", slug="news")
        self.post = Post.objects.create(
            title="Test Post", content="Content", author=self.admin_user
        )

    def test_non_admin_cannot_access_dashboard(self):
        self.client.login(username="contributor", password="testpass123")
        response = self.client.get(reverse("admin_dashboard"))
        self.assertEqual(response.status_code, 403)

    def test_admin_can_access_dashboard(self):
        self.client.login(username="admin", password="adminpass123")
        response = self.client.get(reverse("admin_dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Manage")

    # -- Tag management --

    def test_admin_tag_list(self):
        self.client.login(username="admin", password="adminpass123")
        response = self.client.get(reverse("admin_tag_list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "News")

    def test_admin_create_tag(self):
        self.client.login(username="admin", password="adminpass123")
        response = self.client.post(reverse("admin_tag_create"), {
            "name": "Updates",
            "slug": "updates",
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(Tag.objects.filter(slug="updates").exists())

    def test_admin_edit_tag(self):
        self.client.login(username="admin", password="adminpass123")
        response = self.client.post(reverse("admin_tag_edit", args=[self.tag.pk]), {
            "name": "Breaking News",
            "slug": "breaking-news",
        })
        self.assertEqual(response.status_code, 302)
        self.tag.refresh_from_db()
        self.assertEqual(self.tag.name, "Breaking News")

    def test_admin_delete_tag(self):
        self.client.login(username="admin", password="adminpass123")
        response = self.client.post(reverse("admin_tag_delete", args=[self.tag.pk]))
        self.assertEqual(response.status_code, 302)
        self.assertFalse(Tag.objects.filter(pk=self.tag.pk).exists())

    # -- User management --

    def test_admin_user_list(self):
        self.client.login(username="admin", password="adminpass123")
        response = self.client.get(reverse("admin_user_list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "admin")
        self.assertContains(response, "contributor")

    def test_admin_create_user(self):
        self.client.login(username="admin", password="adminpass123")
        response = self.client.post(reverse("admin_user_create"), {
            "username": "newuser",
            "password": "newpass123",
            "role": "viewer",
            "first_name": "",
            "last_name": "",
            "email": "",
        })
        self.assertEqual(response.status_code, 302)
        new_user = User.objects.get(username="newuser")
        self.assertTrue(new_user.check_password("newpass123"))
        self.assertEqual(new_user.profile.role, UserProfile.Role.VIEWER)

    def test_admin_edit_user_role(self):
        self.client.login(username="admin", password="adminpass123")
        response = self.client.post(reverse("admin_user_edit", args=[self.contributor.pk]), {
            "role": "admin",
        })
        self.assertEqual(response.status_code, 302)
        self.contributor.profile.refresh_from_db()
        self.assertEqual(self.contributor.profile.role, UserProfile.Role.ADMIN)

    def test_admin_toggle_user_active(self):
        self.client.login(username="admin", password="adminpass123")
        self.assertTrue(self.contributor.is_active)
        self.client.post(reverse("admin_user_toggle_active", args=[self.contributor.pk]))
        self.contributor.refresh_from_db()
        self.assertFalse(self.contributor.is_active)
        # Toggle back
        self.client.post(reverse("admin_user_toggle_active", args=[self.contributor.pk]))
        self.contributor.refresh_from_db()
        self.assertTrue(self.contributor.is_active)

    def test_admin_cannot_deactivate_self(self):
        self.client.login(username="admin", password="adminpass123")
        self.client.post(reverse("admin_user_toggle_active", args=[self.admin_user.pk]))
        self.admin_user.refresh_from_db()
        self.assertTrue(self.admin_user.is_active)

    # -- Post management --

    def test_admin_post_list(self):
        self.client.login(username="admin", password="adminpass123")
        response = self.client.get(reverse("admin_post_list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test Post")

    def test_non_admin_cannot_access_admin_views(self):
        self.client.login(username="contributor", password="testpass123")
        admin_urls = [
            reverse("admin_dashboard"),
            reverse("admin_tag_list"),
            reverse("admin_tag_create"),
            reverse("admin_user_list"),
            reverse("admin_user_create"),
            reverse("admin_post_list"),
        ]
        for url in admin_urls:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 403, f"Expected 403 for {url}")

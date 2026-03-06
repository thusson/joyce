from django.conf import settings
from django.db import models


class Tag(models.Model):
    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(max_length=100, unique=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class Post(models.Model):
    class Priority(models.TextChoices):
        INTERNAL = "internal", "Internal"
        DIVISIONAL = "divisional", "Divisional"
        ORGANIZATIONAL = "organizational", "Organizational"

    title = models.CharField(max_length=300)
    content = models.TextField(help_text="Markdown formatted content.")
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="posts",
    )
    priority = models.CharField(
        max_length=20,
        choices=Priority.choices,
        default=Priority.INTERNAL,
    )
    tags = models.ManyToManyField(Tag, blank=True, related_name="posts")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.title


class ReadStatus(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="read_statuses",
    )
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name="read_statuses")
    read_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "post")

    def __str__(self):
        return f"{self.user} read {self.post}"


class UserProfile(models.Model):
    class Role(models.TextChoices):
        ADMIN = "admin", "Admin"
        CONTRIBUTOR = "contributor", "Contributor"
        VIEWER = "viewer", "Viewer"

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="profile",
    )
    role = models.CharField(
        max_length=20,
        choices=Role.choices,
        default=Role.VIEWER,
    )

    def __str__(self):
        return f"{self.user.username} ({self.get_role_display()})"

    @property
    def is_admin(self):
        return self.role == self.Role.ADMIN

    @property
    def is_contributor(self):
        return self.role == self.Role.CONTRIBUTOR

    @property
    def can_create_post(self):
        return self.role in (self.Role.ADMIN, self.Role.CONTRIBUTOR)

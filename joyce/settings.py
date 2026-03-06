"""
Django settings for joyce project.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = "django-insecure-=hif$_o6&iv(1-c@y)=5ot3-5)5pxd*p1)gj4!3l%mvspek^9^"

DEBUG = True

ALLOWED_HOSTS = []

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "stream",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "joyce.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "joyce.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "post_list"
LOGOUT_REDIRECT_URL = "login"

# ActiveDirectory / LDAP authentication
# To enable, install django-auth-ldap and uncomment the following:
#
# import ldap
# from django_auth_ldap.config import LDAPSearch, ActiveDirectoryGroupType
#
# AUTHENTICATION_BACKENDS = [
#     "django_auth_ldap.backend.LDAPBackend",
#     "django.contrib.auth.backends.ModelBackend",
# ]
#
# AUTH_LDAP_SERVER_URI = "ldaps://ad.example.com"
# AUTH_LDAP_BIND_DN = "CN=svc_joyce,OU=Service Accounts,DC=example,DC=com"
# AUTH_LDAP_BIND_PASSWORD = ""  # Set via environment variable in production
# AUTH_LDAP_USER_SEARCH = LDAPSearch(
#     "OU=Users,DC=example,DC=com",
#     ldap.SCOPE_SUBTREE,
#     "(sAMAccountName=%(user)s)",
# )
# AUTH_LDAP_USER_ATTR_MAP = {
#     "first_name": "givenName",
#     "last_name": "sn",
#     "email": "mail",
# }

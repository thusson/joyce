# joyce -- a simple comment-tag-post activity stream

## Goal of this project

This project implements a simple browser-based activity stream with the ability for users to post blog-like content complete with tags and comment sections.  Site content can be labeled for its level of priority -- internal, divisional, organizational -- which determines which users it gets pushed to.

## Stream characteristics

Generally resembles TheOldReader -- simple, list-based.  

- Each post contains a text title, markdown-formatted content, tags, and comments
- Defaults to newest-first sorting
- Can be filtered by tags
- Is marked as read/unread
- Includes text search
- Full audit trail of all activity
- Posts can contain links and inline images but no attachments

## Tech stack and usage characteristics

- Python
- Django
- SQLite
- ZERO JavaScript, TypeScript, or associated dependencies
- User authentication through ActiveDirectory
- Expected user base ~200, maximum 3000
- Expected number of admins ~5
- Expected number of contributors ~20

## User types

- Users are manually assigned their type by admin
- Admins (internal priority)
  - Can add/edit/delete any content
  - Can define available tags
  - Can comment on any post
- Contributors (internal)
  - Can add/edit/delete their own content
  - Can tag their own content
  - Can comment on any post
- Viewers (divisional, and organizational)
  - Can comment on any post

## Development plan

1. Posts, list view, and authentication
2. Admin panel for content and user management
3. Comments on posts
4. Add inline images to posts
5. Search
6. Audit trail

# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Authentication for user filing issue (must have read/write access to repository to add issue to)
USERNAME = 'shazh-ms'
ISSUE_TO_COMMENT = 8

# The repository to add this issue to
REPO_OWNER = 'shaofei zhang'
REPO_NAME = 'ArXivDaily_hassem'

# Set new submission url of subject
NEW_SUB_URL = 'https://arxiv.org/list/eess.AS/new'
TARGET_TITLES = [
    "New submissions",
    "Cross submissions"
]

# Keywords to search
KEYWORD_LIST = [
    "text-to-speech",
    "text to speech",
    "tts",
    "LLM-based",
    "speech",
    "voice"
]

# Keywords to exclude
KEYWORD_EX_LIST = []

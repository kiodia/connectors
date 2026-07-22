"""
Gmail connector class for reading, deleting emails and extracting attachments.

Uses the Google Gmail API via google-api-python-client.
The user must have:
1. Created a Google Cloud project with Gmail API enabled
2. Created OAuth 2.0 credentials (credentials.json)
3. Completed the OAuth consent flow (which generates token.json)

Required packages:
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
"""

import logging
import re
import base64
import os
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from connectors.mail import Mail, Attachment

log = logging.getLogger(__name__)

# If modifying these scopes, delete the file token.json.
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify',
]


class Gmail:
    """
    Gmail connector that provides methods to read, delete emails
    and extract attachments from a Gmail account.

    Usage:
        gmail = Gmail(
            credentials_file='path/to/credentials.json',
            token_file='path/to/token.json',
            email_address='user@gmail.com'
        )
        emails = gmail.read(date='2026-03-13', sender_regex=r'.*@arxiv\\.org')
        attachments = gmail.read_attachments(message_id='msg_id_123')
        gmail.delete(message_id='msg_id_123')
    """

    def __init__(self, credentials_file: str, token_file: str, email_address: str):
        """
        Initialize the Gmail connector and authenticate with the Gmail API.

        Args:
            credentials_file: Path to the OAuth 2.0 credentials JSON file
                downloaded from Google Cloud Console.
            token_file: Path to the token JSON file that stores the user's
                access and refresh tokens. Created automatically after the
                first authorization flow.
            email_address: The Gmail email address to access.
        """
        self.email_address = email_address
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None

        self._authenticate()
        log.info(f"Gmail connector initialized for {self.email_address}")

    def _authenticate(self) -> None:
        """
        Authenticate with the Gmail API using OAuth 2.0 credentials.
        If a valid token exists, it will be reused. If the token is expired,
        it will be refreshed. Otherwise, the OAuth consent flow is initiated.
        """
        creds = None

        # Load existing token if available
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
            log.info("Loaded existing credentials from token file")

        # If there are no valid credentials, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                log.info("Refreshing expired credentials")
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(
                        f"Credentials file not found: {self.credentials_file}. "
                        f"Download it from Google Cloud Console."
                    )
                log.info("Initiating OAuth consent flow")
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
            log.info("Credentials saved to token file")

        self.service = build('gmail', 'v1', credentials=creds)
        log.info("Gmail API service created successfully")

    def _get_header(self, headers: List[Dict], name: str) -> str:
        """Extract a header value from a list of Gmail message headers."""
        for header in headers:
            if header['name'].lower() == name.lower():
                return header['value']
        return ''

    def _extract_email_address(self, sender_field: str) -> str:
        """Extract the email address from a sender field like 'Name <email@example.com>'."""
        match = re.search(r'<([^>]+)>', sender_field)
        if match:
            return match.group(1)
        return sender_field.strip()

    def read(self, date: str, sender_regex: str = r'.*') -> List[Dict[str, Any]]:
        """
        Read all emails received on a given date, filtered by sender regex.

        Args:
            date: Date string in format 'YYYY-MM-DD' to filter emails by received date.
            sender_regex: Regular expression pattern to filter emails by sender address.
                Defaults to '.*' (matches all senders).

        Returns:
            List of dictionaries, each containing:
                - id: Gmail message ID
                - thread_id: Gmail thread ID
                - sender: Sender email address (full From header)
                - sender_email: Extracted email address only
                - subject: Email subject line
                - date: Date the email was received
                - snippet: Short preview of the email body
                - body: Full email body text
                - has_attachments: Whether the email has attachments
        """
        # Build Gmail search query for the given date
        # Gmail query format: after:YYYY/MM/DD before:YYYY/MM/DD
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        query = f"after:{date_obj.strftime('%Y/%m/%d')} before:{date_obj.strftime('%Y/%m/%d')}"

        # Add one day to the 'before' to include the entire target date
        from datetime import timedelta
        next_day = date_obj + timedelta(days=1)
        query = f"after:{date_obj.strftime('%Y/%m/%d')} before:{next_day.strftime('%Y/%m/%d')}"

        log.info(f"Searching Gmail with query: {query}")

        emails = []
        page_token = None
        pattern = re.compile(sender_regex, re.IGNORECASE)

        try:
            while True:
                # List messages matching the query
                results = self.service.users().messages().list(
                    userId='me',
                    q=query,
                    pageToken=page_token,
                    maxResults=500
                ).execute()

                messages = results.get('messages', [])
                if not messages:
                    break

                for msg_ref in messages:
                    # Fetch full message details
                    msg = self.service.users().messages().get(
                        userId='me',
                        id=msg_ref['id'],
                        format='full'
                    ).execute()

                    headers = msg.get('payload', {}).get('headers', [])
                    sender = self._get_header(headers, 'From')
                    sender_email = self._extract_email_address(sender)

                    # Apply sender regex filter
                    if not pattern.search(sender) and not pattern.search(sender_email):
                        continue

                    subject = self._get_header(headers, 'Subject')
                    msg_date = self._get_header(headers, 'Date')

                    # Extract body text
                    body = self._extract_body(msg.get('payload', {}))

                    # Check for attachments
                    has_attachments = self._has_attachments(msg.get('payload', {}))

                    email_data = {
                        'id': msg['id'],
                        'thread_id': msg.get('threadId', ''),
                        'sender': sender,
                        'sender_email': sender_email,
                        'subject': subject,
                        'date': msg_date,
                        'snippet': msg.get('snippet', ''),
                        'body': body,
                        'has_attachments': has_attachments,
                    }
                    emails.append(email_data)

                # Check for next page
                page_token = results.get('nextPageToken')
                if not page_token:
                    break

            log.info(f"Found {len(emails)} emails for date {date} matching sender pattern '{sender_regex}'")
            return emails

        except Exception as e:
            log.error(f"Error reading emails: {e}")
            return []

    def _extract_body(self, payload: Dict) -> str:
        """
        Recursively extract the plain text body from a Gmail message payload.

        Args:
            payload: The Gmail message payload dictionary.

        Returns:
            The decoded plain text body of the email.
        """
        body_text = ''

        if 'body' in payload and payload['body'].get('data'):
            body_text = base64.urlsafe_b64decode(
                payload['body']['data']
            ).decode('utf-8', errors='replace')

        # Recurse into parts (multipart messages)
        if 'parts' in payload:
            for part in payload['parts']:
                mime_type = part.get('mimeType', '')
                if mime_type == 'text/plain':
                    data = part.get('body', {}).get('data', '')
                    if data:
                        body_text += base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
                elif mime_type.startswith('multipart/'):
                    body_text += self._extract_body(part)

        return body_text

    def _has_attachments(self, payload: Dict) -> bool:
        """Check if a message payload contains attachments."""
        if 'parts' in payload:
            for part in payload['parts']:
                disposition = part.get('body', {}).get('attachmentId')
                filename = part.get('filename', '')
                if disposition or (filename and filename != ''):
                    return True
                # Recurse into nested parts
                if 'parts' in part:
                    if self._has_attachments(part):
                        return True
        return False

    def delete(self, message_id: str) -> bool:
        """
        Delete an email message from the Gmail account.
        This moves the message to the Trash folder.

        Args:
            message_id: The Gmail message ID to delete.

        Returns:
            True if the message was successfully deleted, False otherwise.
        """
        try:
            self.service.users().messages().trash(
                userId='me',
                id=message_id
            ).execute()
            log.info(f"Message {message_id} moved to trash")
            return True
        except Exception as e:
            log.error(f"Error deleting message {message_id}: {e}")
            return False

    def read_attachments(self, message_id: str) -> List[Dict[str, Any]]:
        """
        Read and return all attachments from a given email message.

        Args:
            message_id: The Gmail message ID to extract attachments from.

        Returns:
            List of dictionaries, each containing:
                - filename: Name of the attached file
                - mime_type: MIME type of the attachment
                - size: Size of the attachment in bytes
                - data: Raw bytes of the attachment content
        """
        attachments = []

        try:
            msg = self.service.users().messages().get(
                userId='me',
                id=message_id,
                format='full'
            ).execute()

            payload = msg.get('payload', {})
            self._collect_attachments(payload, message_id, attachments)

            log.info(f"Found {len(attachments)} attachments in message {message_id}")
            return attachments

        except Exception as e:
            log.error(f"Error reading attachments from message {message_id}: {e}")
            return []

    def _collect_attachments(self, payload: Dict, message_id: str, attachments: List[Dict]) -> None:
        """
        Recursively collect attachments from a message payload.

        Args:
            payload: The Gmail message payload dictionary.
            message_id: The Gmail message ID (needed to fetch attachment data).
            attachments: List to append attachment dictionaries to.
        """
        if 'parts' in payload:
            for part in payload['parts']:
                filename = part.get('filename', '')
                mime_type = part.get('mimeType', '')
                body = part.get('body', {})
                attachment_id = body.get('attachmentId')

                if attachment_id and filename:
                    # Fetch the actual attachment data
                    try:
                        att = self.service.users().messages().attachments().get(
                            userId='me',
                            messageId=message_id,
                            id=attachment_id
                        ).execute()

                        data = base64.urlsafe_b64decode(att['data'])

                        attachments.append({
                            'filename': filename,
                            'mime_type': mime_type,
                            'size': len(data),
                            'data': data,
                        })
                        log.info(f"Extracted attachment: {filename} ({mime_type}, {len(data)} bytes)")

                    except Exception as e:
                        log.error(f"Error fetching attachment {filename}: {e}")

                # Recurse into nested multipart parts
                if 'parts' in part:
                    self._collect_attachments(part, message_id, attachments)

    def _extract_html_body(self, payload: Dict) -> Optional[str]:
        """
        Recursively extract the HTML body from a Gmail message payload.

        Args:
            payload: The Gmail message payload dictionary.

        Returns:
            The decoded HTML body of the email, or None if not found.
        """
        html_text = ''

        if 'parts' in payload:
            for part in payload['parts']:
                mime_type = part.get('mimeType', '')
                if mime_type == 'text/html':
                    data = part.get('body', {}).get('data', '')
                    if data:
                        html_text += base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
                elif mime_type.startswith('multipart/'):
                    result = self._extract_html_body(part)
                    if result:
                        html_text += result

        return html_text if html_text else None

    def _extract_recipients(self, headers: List[Dict]) -> List[str]:
        """Extract recipient email addresses from To, Cc headers."""
        recipients = []
        for header_name in ['To', 'Cc']:
            value = self._get_header(headers, header_name)
            if value:
                # Split by comma and extract email addresses
                for addr in value.split(','):
                    addr = addr.strip()
                    email_addr = self._extract_email_address(addr)
                    if email_addr:
                        recipients.append(email_addr)
        return recipients

    def _extract_labels(self, msg: Dict) -> List[str]:
        """Extract label IDs from a Gmail message."""
        return msg.get('labelIds', [])

    def read_as_mail(self, date: str, sender_regex: str = r'.*') -> List[Mail]:
        """
        Read all emails received on a given date, filtered by sender regex,
        and return them as Mail objects.

        Args:
            date: Date string in format 'YYYY-MM-DD' to filter emails by received date.
            sender_regex: Regular expression pattern to filter emails by sender address.
                Defaults to '.*' (matches all senders).

        Returns:
            List of Mail objects containing the email data.
        """
        # Build Gmail search query for the given date
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        from datetime import timedelta
        next_day = date_obj + timedelta(days=1)
        query = f"after:{date_obj.strftime('%Y/%m/%d')} before:{next_day.strftime('%Y/%m/%d')}"

        log.info(f"Searching Gmail with query: {query}")

        mails: List[Mail] = []
        page_token = None
        pattern = re.compile(sender_regex, re.IGNORECASE)

        try:
            while True:
                results = self.service.users().messages().list(
                    userId='me',
                    q=query,
                    pageToken=page_token,
                    maxResults=500
                ).execute()

                messages = results.get('messages', [])
                if not messages:
                    break

                for msg_ref in messages:
                    msg = self.service.users().messages().get(
                        userId='me',
                        id=msg_ref['id'],
                        format='full'
                    ).execute()

                    headers = msg.get('payload', {}).get('headers', [])
                    sender = self._get_header(headers, 'From')
                    sender_email = self._extract_email_address(sender)

                    # Apply sender regex filter
                    if not pattern.search(sender) and not pattern.search(sender_email):
                        continue

                    subject = self._get_header(headers, 'Subject')
                    msg_date = self._get_header(headers, 'Date')
                    recipients = self._extract_recipients(headers)
                    body_plain = self._extract_body(msg.get('payload', {}))
                    body_html = self._extract_html_body(msg.get('payload', {}))
                    labels = self._extract_labels(msg)

                    mail = Mail(
                        id=msg['id'],
                        thread_id=msg.get('threadId', ''),
                        sender=sender,
                        recipients=recipients if recipients else [self.email_address],
                        subject=subject,
                        body_plain=body_plain if body_plain else None,
                        body_html=body_html,
                        date_received=msg_date,
                        is_read='UNREAD' not in labels,
                        labels=labels,
                    )
                    mails.append(mail)

                page_token = results.get('nextPageToken')
                if not page_token:
                    break

            log.info(f"Found {len(mails)} emails for date {date} matching sender pattern '{sender_regex}'")
            return mails

        except Exception as e:
            log.error(f"Error reading emails: {e}")
            return []



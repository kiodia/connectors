'''
Credential lookup shared by the connectors.

The .env file belongs to the calling application (keopy, MatchMake, ...), not
to this library, so a plain ``load_dotenv()`` at import time searches from this
package upwards and misses it whenever the application is started from another
working directory. Credentials are therefore resolved at call time, and the
.env is looked up from the working directory upwards.

Resolving late also keeps importing a connector free of side effects: a missing
key must not break consumers that never call the connector it belongs to.

@author: vankomme
'''

import os

from dotenv import load_dotenv, find_dotenv


class MissingCredential(RuntimeError):
    '''
    Raised when a credential is configured neither in the environment nor in
    the .env of the calling application.

    A distinct type so callers can tell a configuration problem apart from a
    transient API failure (and skip retrying it).
    '''


def get_credential(name, default=""):
    '''
    Read a credential from the environment, falling back to the .env file.

    PARAMETERS
    ----------
    name : string
            The environment variable to read, e.g. "BING_SEARCH_KEY"
    default : string
            Returned when the variable is set nowhere

    RETURNS
    -------
        the credential value, or default when it is not configured
    '''

    value = os.getenv(name, "")
    if not value:
        # Not exported in the environment: look for a .env from the working
        # directory upwards (find_dotenv() searches from this file by default).
        load_dotenv(find_dotenv(usecwd=True))
        value = os.getenv(name, default)
    return value


def require_credential(name):
    '''
    Read a credential and fail loudly when it is missing.

    PARAMETERS
    ----------
    name : string
            The environment variable to read, e.g. "BING_SEARCH_KEY"

    RETURNS
    -------
        the credential value

    RAISES
    ------
        MissingCredential : when the variable is set nowhere
    '''

    value = get_credential(name)
    if not value:
        raise MissingCredential(
            f"{name} is not configured (set it in the environment or in the "
            ".env of the application you are running)")
    return value

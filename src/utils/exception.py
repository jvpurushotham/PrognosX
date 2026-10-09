"""
PrognosX — Custom Exception Handling
======================================
A single PrognosXException wraps any underlying error with the
file name and line number where it occurred, which makes
debugging data/ML pipelines far easier than a bare traceback.
"""

import sys


def error_message_detail(error: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "unknown"
    line_number = exc_tb.tb_lineno if exc_tb else -1
    return (
        f"Error occurred in script [{file_name}] "
        f"at line [{line_number}]: {str(error)}"
    )


class PrognosXException(Exception):
    """Base exception for all PrognosX pipeline errors."""

    def __init__(self, error: Exception, error_detail: sys = sys):
        super().__init__(str(error))
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self):
        return self.error_message

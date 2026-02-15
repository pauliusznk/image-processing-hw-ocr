from __future__ import annotations

import sys
import time
import threading
from typing import Optional


class Spinner:
    """Simple loading spinner for showing progress."""

    def __init__(self, message: str = "Loading"):
        self.message = message
        self.running = False
        self.thread: Optional[threading.Thread] = None
        # Animation frames
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.frame_idx = 0

    def _spin(self):
        """Internal spinning animation loop."""
        while self.running:
            frame = self.frames[self.frame_idx % len(self.frames)]
            sys.stdout.write(f"\r{frame} {self.message}...")
            sys.stdout.flush()
            self.frame_idx += 1
            time.sleep(0.1)

    def start(self):
        """Start the spinner animation."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self, final_message: Optional[str] = None):
        """Stop the spinner animation."""
        if not self.running:
            return
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the line
        sys.stdout.write("\r" + " " * (len(self.message) + 20) + "\r")
        if final_message:
            print(final_message)
        sys.stdout.flush()

    def update_message(self, new_message: str):
        """Update the spinner message while it's running."""
        self.message = new_message


def with_spinner(message: str):
    """Decorator to wrap a function with a spinner animation."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            spinner = Spinner(message)
            spinner.start()
            try:
                result = func(*args, **kwargs)
                spinner.stop()
                return result
            except Exception as e:
                spinner.stop(f"❌ {message} failed")
                raise e
        return wrapper
    return decorator

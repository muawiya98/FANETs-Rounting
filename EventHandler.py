from collections import deque


class EventHandler:
    def __init__(self):
        self.event_listener = deque()

    def __str__(self):
        return f"Events is {self.event_listener}"

    def __iadd__(self, event):
        self.event_listener.append(event)
        return self

    def is_empty(self):
        return len(self.event_listener) == 0

    def get_top_message(self):
        # removes the message
        event = self.event_listener.popleft()
        return event

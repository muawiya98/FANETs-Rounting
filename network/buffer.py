class Buffer:
    def __init__(self, max_length):
        self.max_length = max_length
        self.buffer = []

    def __iadd__(self, event):
        self.buffer.append(event)
        return self

    def __isub__(self, event_id):
        self.buffer.pop(event_id)
        return self

    def has_space(self):
        return len(self.buffer) < self.max_length

    def get_top_packet(self):
        try:
            return self.buffer[0]
        except (Exception,):
            return None

    def clear(self):
        self.buffer.clear()

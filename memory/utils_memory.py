from typing import List
from datetime import datetime

class Message:
    def __init__(self, sender: str, content: str, timestamp: float = None):
        self.sender = sender
        self.content = content
        self.timestamp = timestamp or datetime.now().timestamp()

    def to_dict(self):
        return {
            "sender": self.sender,
            "content": self.content,
            "timestamp": self.timestamp
        }

    @staticmethod
    def from_dict(data):
        return Message(data["sender"], data["content"], data["timestamp"])

def format_messages(messages: List[Message]) -> str:
    return "\n".join([f"{msg.sender}: {msg.content}" for msg in messages])

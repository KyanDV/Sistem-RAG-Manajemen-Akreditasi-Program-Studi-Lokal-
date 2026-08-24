import json
import os
import uuid
from dataclasses import dataclass, asdict, field
from typing import Optional

CHATS_DIR = "./data/chats"


@dataclass
class Chat:
    id: str
    title: str
    messages: list = field(default_factory=list)


class ChatStore:
    def __init__(self, user_id: str):
        self.path = os.path.join(CHATS_DIR, f"chats_{user_id}.json")
        os.makedirs(CHATS_DIR, exist_ok=True)
        self._chats: list[Chat] = []
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._chats = [Chat(**c) for c in data]

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self._chats], f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def create(self, title: str = "") -> Chat:
        chat = Chat(id=str(uuid.uuid4()), title=title or "Chat Baru", messages=[])
        self._chats.insert(0, chat)
        self.save()
        return chat

    def list(self) -> list[Chat]:
        return self._chats

    def get(self, chat_id: str) -> Optional[Chat]:
        for c in self._chats:
            if c.id == chat_id:
                return c
        return None

    def update(self, chat_id: str, messages: list) -> Optional[Chat]:
        chat = self.get(chat_id)
        if not chat:
            return None
        chat.messages = messages
        if messages and chat.title == "Chat Baru":
            first_q = ""
            if isinstance(messages[0], list):
                first_q = messages[0][0]
            elif isinstance(messages[0], dict):
                first_q = messages[0].get("content", "")
            if first_q:
                chat.title = (first_q[:60] + "...") if len(first_q) > 60 else first_q
        self.save()
        return chat

    def delete(self, chat_id: str):
        self._chats = [c for c in self._chats if c.id != chat_id]
        self.save()

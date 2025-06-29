import redis
import json
import os

REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = os.getenv('REDIS_PORT', 6379)

class LastStateStorage:
    def __init__(self):
        self.redis = redis.Redis(
            host=REDIS_HOST,
            port=int(REDIS_PORT),
            decode_responses=True
        )

    def get_order(self, user_id: str) -> dict:
        key = f"last_state_order:{user_id}"
        value = self.redis.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {"order": []}
        return {"order": []}

    def set_order(self, user_id: str, order: dict):
        key = f"last_state_order:{user_id}"
        self.redis.set(key, json.dumps(order, ensure_ascii=False))

    def clear_order(self, user_id: str):
        self.redis.delete(f"last_state_order:{user_id}")

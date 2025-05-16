import os
import logging
import asyncio
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message

from llm_api import LLMService

# Загрузка .env
load_dotenv()
TOKEN = os.environ["BOT_KEY"]

# Инициализация бота и диспетчера
bot = Bot(token=TOKEN)
dp = Dispatcher()

# Создаем экземпляр LLM-сервиса один раз
llm_service = LLMService()

@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "Привет, я бот, у которого можно создать и редактировать заказ с помощью текста или голоса"
    )

@dp.message(F.voice)
async def handle_voice(message: Message):
    await message.answer("Получено голосовое сообщение")

@dp.message(F.text)
async def handle_text(message: Message):
    user_query = message.text
    user_id = str(message.from_user.id)
    llm_answer = llm_service.generate(user_id, user_query)
    await message.answer(llm_answer)

async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())


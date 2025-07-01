import os
import logging
import asyncio
from dotenv import load_dotenv
import tempfile
from pathlib import Path
from uuid import uuid4
import torch

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery

from llm_api import LLMService
from transformers import pipeline

load_dotenv()
TOKEN = os.getenv("BOT_KEY")

bot = Bot(token=TOKEN)
dp = Dispatcher()

llm_service = LLMService(model="gpt-4.1-mini")
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)

def get_order_keyboard():
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🛒 Оформить заказ", callback_data="submit_order")
            ]
        ]
    )
    return keyboard

@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "Привет, я бот, у которого можно создать и редактировать заказ с помощью текста или голоса"
    )

@dp.message(F.voice)
async def handle_voice(message: Message):
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            voice_path = tmp_path / f"voice_{str(uuid4())}.ogg"
            voice_file = await bot.get_file(message.voice.file_id)
            await bot.download_file(voice_file.file_path, destination=str(voice_path))
            try:
                with open(voice_path, "rb") as audio_file:
                    transcript = asr(audio_file.read())
                user_query = transcript["text"]
            except Exception as e:
                logging.error(f"Ошибка распознавания голоса: {e}", exc_info=True)
                await message.answer("Не удалось распознать голос. Попробуйте ещё раз.")

            user_id = str(message.from_user.id)
            try:
                llm_answer = llm_service.generate(user_id, user_query)
            except Exception as e:
                logging.error(f"Ошибка генерации ответа LLM: {e}", exc_info=True)
                await message.answer("Произошла ошибка при генерации ответа. Попробуйте позже.")

            answer_lower = llm_answer.lower()

            if "ваш заказ пуст" in answer_lower:
                await message.answer(llm_answer)
            elif "ваш заказ:" in answer_lower:
                await message.answer(
                    llm_answer,
                    reply_markup=get_order_keyboard()
                )
            else:
                await message.answer(llm_answer)

    except Exception as e:
        logging.error(f"Неизвестная ошибка при обработке голосового сообщения: {e}", exc_info=True)
        await message.answer("Произошла неизвестная ошибка. Попробуйте позже.")

@dp.message(F.text)
async def handle_text(message: Message):
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    user_query = message.text
    user_id = str(message.from_user.id)
    try:
        llm_answer = llm_service.generate(user_id, user_query)
    except Exception as e:
        logging.error(f"Ошибка генерации ответа LLM: {e}", exc_info=True)
        await message.answer("Произошла ошибка при генерации ответа. Попробуйте позже.")

    answer_lower = llm_answer.lower()

    if "ваш заказ пуст" in answer_lower:
        await message.answer(llm_answer)
    elif "ваш заказ:" in answer_lower:
        await message.answer(
            llm_answer,
            reply_markup=get_order_keyboard()
        )
    else:
        await message.answer(llm_answer)

@dp.message(~(F.content_type.in_({"text", "voice"})))
async def handle_other(message: Message):
    await message.answer("Извините, я понимаю только текст и голосовые сообщения.")

@dp.callback_query(F.data == "submit_order")
async def process_order_submission(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    await callback.answer()
    await callback.message.answer("✅ Ваш заказ оформлен! Спасибо за покупку.")
    llm_service.last_state_storage.clear_order(user_id)
    llm_service.clear_user_history(user_id)

async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

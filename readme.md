# Идея для проекта и данные были взяты здесь: https://stepik.org/course/178846/promo?search=7551128053
# Телеграмм бот оператор:
⚠️ **Внимание!**\
**Все цены сгенерированы с помощью LLM.**
## Как запустить?
1.  Создать переменные окружения
```bash
export BOT_KEY=<your_value> # Telegram Bot Token
export BASE_URL=<your_value> # Base langchain URL for openai requests
export LLM_KEY=<your_value> # OPENAI Token
```
2. Клонировать репозиторий
```bash
mkdir ~/operator_bot
cd ~/operator_bot
git clone https://github.com/ShuvalovDR/rag_bot.git .
```
3. Запустить docker compose
```bash
docker compose up --build
```
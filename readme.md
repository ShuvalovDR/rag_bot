# Operator Telegram Bot

## How to run
1.  Create enviroment variables
```bash
export BOT_KEY=<your_value> # Telegram Bot Token
export BASE_URL=<your_value> # Base langchain URL for openai requests
export LLM_KEY=<your_value> # OPENAI Token
```
2. Clone repo
```bash
mkdir ~/operator_bot
cd ~/operator_bot
git clone https://github.com/ShuvalovDR/rag_bot.git .
```
3. Run docker compose
```bash
docker compose up --build
```
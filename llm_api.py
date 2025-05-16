import os
import json
import re
from json_repair import repair_json
from typing import List
from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import trim_messages, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers.string import StrOutputParser
from langchain_redis import RedisChatMessageHistory

from vector_store import init_ensemble_retriever

class LLMService:
    def __init__(self):
        self.model = GigaChat(
            credentials=os.environ["GIGACHAT_KEY"],
            model="GigaChat-2-Max",
            scope="GIGACHAT_API_PERS",
            verify_ssl_certs=False
        )

        self.retriever = init_ensemble_retriever(
            file_path="./rag_data/knowledgebase.csv",
            device="cuda:0",
            k=3
        )

        with open("./prompt_templates/system_prompt.txt", encoding="utf-8") as f:
            self.system_prompt = SystemMessage(content=f.read())

        with open("./prompt_templates/user_prompt.txt", encoding="utf-8") as f:
            template = f.read()
        self.prompt_template = PromptTemplate.from_template(template)

        self.rag_chain = self.retriever | RunnableLambda(self._format_recommendations)
        self.llm_chain = self.model | StrOutputParser()

    def _format_recommendations(self, docs: List[Document]) -> str:
        user_queries = []
        recommendations = []
        for doc in docs:
            user_queries.append(doc.metadata["source"])
            recommendations.append(doc.metadata["Products_name"])
        rag_recommendations = []
        for q, r in zip(user_queries, recommendations):
            rag_recommendations.append(f"User query: {q} -> Recommendations: {r}")
        return "\n".join(rag_recommendations)

    def _parse_json(self, llm_output: str) -> str:

        # Удаляем обрамляющие кавычки вокруг json-блока
        clean_text = re.sub(r"```(?:json)?", "", llm_output, flags=re.IGNORECASE).replace("```", "").strip()

        # Ищем JSON-блок
        start = clean_text.find("{")
        end = clean_text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            return "Это мы не проходили, это нам не задавали. Парам пам пам"

        raw_json = clean_text[start:end + 1]
        before_json = clean_text[:start].strip()
        after_json = clean_text[end + 1:].strip()

        try:
            order = json.loads(raw_json)
        except json.decoder.JSONDecodeError:
            return "Это мы не проходили, это нам не задавали. Парам пам пам"

        # Обработка формата {"order": ["название"]}
        items = order.get("order", [])
        if isinstance(items, list) and all(isinstance(i, str) for i in items):
            items = [{"name": name, "count": 1} for name in items]

        lines = ["🧾 Ваш заказ:"]
        for item in items:
            name = item.get("name", "Без названия")
            count = item.get("count", 0)
            if name == "Без названия" or count <= 0:
                continue
            lines.append(f"• {name} — x{count} шт.")

        order_text = "\n".join(lines) if len(lines) > 1 else "ℹ️ Ваш заказ пуст"

        parts = []
        if before_json:
            parts.append(before_json)
        parts.append(order_text)
        if after_json:
            parts.append(after_json)

        return "\n\n".join(parts)


    def generate(self, user_id: str, user_query: str) -> str:
        user_history = RedisChatMessageHistory(session_id=user_id)

        if len(user_history.messages) == 0:
            user_history.add_message(self.system_prompt)

        recommendations = self.rag_chain.invoke(user_query)
        prompt = self.prompt_template.format(rag_examples=recommendations, user_query=user_query)

        user_history.add_message(HumanMessage(content=prompt))

        last_messages = trim_messages(
            user_history.messages,
            token_counter=len,
            max_tokens=3,
            strategy="last",
            start_on="human",
            include_system=True,
            allow_partial=False
        )

        last_messages.append(HumanMessage(content=prompt))
        llm_output = self.llm_chain.invoke(last_messages)
        with open("logs.txt", "a", encoding="utf-8") as f:
            f.write(llm_output + "\n\n")
        user_history.add_message(AIMessage(content=llm_output))
        parsed_answer = self._parse_json(llm_output)
        return parsed_answer


g = LLMService()
query = "Хочу что-то острое!"
print(g.rag_chain.invoke())
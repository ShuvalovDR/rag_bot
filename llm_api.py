import os
import json
import re
from typing import List
import torch
from langchain_openai import ChatOpenAI
from langchain_core.messages import trim_messages, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers.string import StrOutputParser
from langchain_redis import RedisChatMessageHistory

from vector_store import init_ensemble_retriever
from last_state_storage import LastStateStorage

REDIS_URL = f"redis://{os.getenv('REDIS_HOST', 'redis')}:{os.getenv('REDIS_PORT', 6379)}"

class LLMService:
    def __init__(
            self, 
            model: str = "gpt-4.1-mini", 
            rag_file_path: str = "./rag_data/knowledgebase.csv",
            system_prompt_template_path: str = "./prompt_templates/system_prompt.txt",
            user_prompt_template_path: str = "./prompt_templates/user_prompt.txt"
    ):
        self.model = ChatOpenAI(
            base_url=os.getenv("BASE_URL"),
            model=model,
            api_key=os.getenv("LLM_KEY")
        )

        self.retriever = init_ensemble_retriever(
            file_path=rag_file_path,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            k=3
        )

        self.last_state_storage = LastStateStorage()

        with open(system_prompt_template_path, encoding="utf-8") as f:
            self.system_prompt = SystemMessage(content=f.read())

        with open(user_prompt_template_path, encoding="utf-8") as f:
            template = f.read()
        self.prompt_template = PromptTemplate.from_template(template)

        self.rag_chain = self.retriever | RunnableLambda(self._format_recommendations)
        self.llm_chain = self.model | StrOutputParser()

    def _parse_json(self, llm_output: str):
        """
        Извлекает JSON-блок из ответа модели и возвращает текст с разметкой заказа.
        Если JSON невалиден — возвращает сообщение об ошибке.
        """
        match = re.search(r"```(?:json)?(.*?)```", llm_output, re.DOTALL | re.IGNORECASE)
        if not match:
            return None, "К сожалению, я Вас не понял, попробуем ещё раз."

        raw_json = match.group(1).strip()
        before_json = llm_output[:match.start()].strip()
        after_json = llm_output[match.end():].strip()

        try:
            order_data = json.loads(raw_json)
        except json.JSONDecodeError:
            return None, "К сожалению, я Вас не понял, попробуем ещё раз."

        items = order_data.get("order", [])
        if isinstance(items, list) and all(isinstance(i, str) for i in items):
            items = [{"name": name, "count": 1} for name in items]

        lines = ["ℹ️ Ваш заказ:"]
        for item in items:
            name = item.get("name", None)
            count = item.get("count", 0)
            if name is None or count <= 0:
                continue
            lines.append(f"• {name} — x{count} шт.")

        if len(lines) == 1:
            order_text = "ℹ️ Ваш заказ пуст"
        else:
            order_text = "\n".join(lines)

        parts = []
        if before_json:
            parts.append(before_json)
        parts.append(order_text)
        if after_json and after_json != raw_json:
            parts.append(after_json)

        return order_data, "\n\n".join(parts)

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

    def clear_user_history(self, user_id):
        user_history = RedisChatMessageHistory(redis_url=REDIS_URL, session_id=user_id)
        user_history.clear()

    def generate(self, user_id: str, user_query: str) -> str:
        user_history = RedisChatMessageHistory(redis_url=REDIS_URL, session_id=user_id)
        last_state = self.last_state_storage.get_order(user_id)

        if len(user_history.messages) == 0:
            user_history.add_message(self.system_prompt)
        last_state = self.last_state_storage.get_order(user_id)
        recommendations = self.rag_chain.invoke(user_query)
        prompt = self.prompt_template.format(rag_examples=recommendations, user_query=user_query, last_state=last_state)

        user_history.add_message(HumanMessage(content=prompt))

        last_messages = trim_messages(
            user_history.messages,
            token_counter=len,
            max_tokens=5,
            strategy="last",
            start_on="human",
            include_system=True,
            allow_partial=False
        )

        last_messages.append(HumanMessage(content=prompt))
        llm_output = self.llm_chain.invoke(last_messages)
        order_data, formatted_text = self._parse_json(llm_output)
        if order_data is not None:
            self.last_state_storage.set_order(user_id, order_data)
        user_history.add_message(AIMessage(content=llm_output))
        return formatted_text

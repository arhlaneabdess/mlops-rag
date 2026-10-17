import os
import asyncio
from threading import Thread
from typing import AsyncGenerator, Dict, Any, List
import torch
import bentoml
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from legal_rag.retriever import LegalRetriever
from legal_rag.text_normalizer import normalize_arabic

LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


class AskPayload(BaseModel):
    question: str = Field(..., min_length=3, description="Arabic legal question")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of articles to retrieve")


@bentoml.service(
    name="arabic_legal_rag_service",
    traffic={"timeout": 60},
    resources={"cpu": "2"}
)
class LegalRAGService:
    def __init__(self):
        print("[*] Initializing Legal RAG Service...")
        # 1. Initialize Vector Retriever
        self.retriever = LegalRetriever()
        print(f"[✓] Connected to ChromaDB. Total indexed chunks: {self.retriever.count()}")

        # 2. Load Generative LLM & Tokenizer directly onto CPU
        print(f"[*] Loading Generative LLM: {LLM_MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        self.llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            dtype=torch.float32,
        )
        self.llm.eval()
        print(f"[✓] Generative LLM {LLM_MODEL_NAME} ready on CPU.")

    def _rerank_candidates(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 2) -> List[Dict[str, Any]]:
        """Distilled re-ranking to prune context to top-2 highest-precision chunks."""
        if not candidates:
            return []
        query_terms = set(query.split())
        scored = []
        for cand in candidates:
            doc_terms = set(cand["document"].split())
            overlap = len(query_terms.intersection(doc_terms)) / max(len(query_terms), 1)
            dense_score = cand.get("score", 0.5)
            score = (dense_score * 0.6) + (overlap * 0.4)
            c = dict(cand)
            c["rerank_score"] = round(score, 4)
            scored.append(c)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_n]

    def _build_prompt(self, question: str, hits: List[Dict[str, Any]]) -> str:
        """Constructs a grounded Arabic legal prompt citing Civil Code articles."""
        context_blocks = "\n\n".join([
            f"[{h['metadata']['citation']}]:\n{h['document']}"
            for h in hits
        ])

        messages = [
            {
                "role": "system",
                "content": (
                    "أنت مستشار قانوني مصري خبير متخصص في القانون المدني المصري (القانون رقم 131 لسنة 1948). "
                    "مهمتك: الإجابة على استفسار المستشير بدقة قانونية عالية وصياغة واضحة استناداً حصراً إلى المواد القانونية المرفقة أدناه.\n"
                    "القواعد الإلزامية:\n"
                    "1. اذكر صراحة رقم المادة القانونية وسندك في الإجابة (مثال: طبقاً للمادة ... من القانون المدني).\n"
                    "2. لا تختلق أحكاماً غير واردة في المواد المرفقة."
                )
            },
            {
                "role": "user",
                "content": f"المواد القانونية المسترجعة من القانون المدني:\n{context_blocks}\n\nسؤال المستشير:\n{question}"
            }
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @bentoml.api
    async def ask(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Synchronous REST endpoint returning full synthesized LLM response and citations."""
        norm_q = normalize_arabic(question)
        candidates = self.retriever.search(norm_q, top_k=top_k)
        hits = self._rerank_candidates(norm_q, candidates, top_n=2)

        if not hits:
            return {
                "answer": "لم يتم العثور على نصوص قانونية مطابقة في القانون المدني المصري.",
                "sources": []
            }

        prompt = self._build_prompt(question, hits)
        inputs = self.tokenizer([prompt], return_tensors="pt")

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
                do_sample=False,
                repetition_penalty=1.1
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        ai_answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        sources = list(dict.fromkeys(h["metadata"]["citation"] for h in hits))

        return {
            "answer": ai_answer,
            "sources": sources,
            "top_article": hits[0]["metadata"]["article_number"]
        }

    @bentoml.api
    async def ask_stream(self, question: str, top_k: int = 5) -> AsyncGenerator[str, None]:
        """Streaming endpoint generating AI tokens progressively via TextIteratorStreamer."""
        norm_q = normalize_arabic(question)
        candidates = self.retriever.search(norm_q, top_k=top_k)
        hits = self._rerank_candidates(norm_q, candidates, top_n=2)

        if not hits:
            yield "لم يتم العثور على مواد قانونية مطابقة في القانون المدني."
            return

        sources = list(dict.fromkeys(h["metadata"]["citation"] for h in hits))
        yield f"[*] السند القانوني المسترجع: {', '.join(sources)}\n\n"
        await asyncio.sleep(0.01)

        prompt = self._build_prompt(question, hits)
        inputs = self.tokenizer([prompt], return_tensors="pt")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=False,
            repetition_penalty=1.1
        )

        thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
        thread.start()

        for new_token in streamer:
            yield new_token
            await asyncio.sleep(0.001)
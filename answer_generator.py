import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from mistralai.client import MistralClient

load_dotenv()

class LLMAnswerGenerator:
    def __init__(self, model_name: str = "mistral-large-latest"):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set in the environment.")

        self.client = MistralClient(api_key=api_key)
        self.model_name = model_name

    def _build_context_and_sources(self, retrieved_chunks: List[Dict[str, Any]]):
        """
        Build a big context string + a clean list of sources.
        Expected chunk format (from retrieval):
        {
          "text": str,
          "score": float,
          "source_filename": str,
          "chunk_number": int,
          "chunk_id": str, ...
        }
        """
        context_blocks = []
        sources_used = []
        context_entries = []

        for idx, chunk in enumerate(retrieved_chunks, 1):
            text = chunk.get("text") or chunk.get("page_content") or ""
            source_filename = chunk.get("source_filename", "unknown")
            chunk_number = chunk.get("chunk_number", "unknown")
            chunk_id = chunk.get("chunk_id", None)
            score = chunk.get("score", None)

            label = f"[Source: {source_filename} | chunk {chunk_number}]"

            # For LLM context
            context_blocks.append(f"{label}\n{text}")

            # For 'Sources Used' section
            sources_used.append({
                "source_filename": source_filename,
                "chunk_number": chunk_number,
                "chunk_id": chunk_id,
                "score": score,
            })

            # For 'Retrieved Context' section
            context_entries.append({
                "label": label,
                "text": text,
            })

        context_str = "\n\n".join(context_blocks)
        return context_str, sources_used, context_entries

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        1. Build context from retrieved chunks
        2. Call Mistral LLM
        3. Return structured output:
           - final_answer
           - sources_used
           - retrieved_context
        """
        if not retrieved_chunks:
            return {
                "final_answer": "I could not find any relevant information in the retrieved context.",
                "sources_used": [],
                "retrieved_context": [],
            }

        context_str, sources_used, context_entries = self._build_context_and_sources(
            retrieved_chunks
        )

        system_prompt = (
            "You are an assistant for a Retrieval-Augmented Generation (RAG) system.\n"
            "You MUST answer strictly based only on the provided context chunks.\n"
            "If the answer is not present in the context, reply that you don't know.\n"
            "When you use a fact, include a citation in the form:\n"
            "[Source: file.pdf | chunk 12] or 'According to Document: file.pdf (Chunk #4)'.\n"
            "Do NOT introduce any external knowledge.\n"
        )

        user_prompt = (
            f"User question:\n{question}\n\n"
            f"Context chunks:\n{context_str}\n\n"
            "Now provide a concise, clear answer grounded ONLY in the context above. "
            "Include inline citations as requested."
        )

        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        final_answer = response.choices[0].message.content

        return {
            "final_answer": final_answer,
            "sources_used": sources_used,
            "retrieved_context": context_entries,
        }

# query_rag_functions.py
# Contém as funções principais do RAG para serem usadas pelo Streamlit
# query_rag_functions.py
# Contém as funções principais do RAG para serem usadas pelo Streamlit

import os
import sys
from typing import List, Optional, Tuple

import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from google.generativeai import GenerativeModel 
from langchain_core.documents import Document

# Nível de corte de distância para determinar a relevância do contexto.
DISTANCE_THRESHOLD = 0.5 

def setup_gemini(api_key: str, model: str = "gemini-2.5-flash") -> GenerativeModel:
    """Configura Gemini com API key e modelo."""
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model)
    except Exception as e:
        raise SystemExit(f"Erro ao configurar o Gemini: {e}")

def format_context_for_gemini(docs: List[Tuple[Document, float]]) -> Tuple[str, str, List[Document]]:
    """
    Formata documentos, filtra por score e determina a instrução do prompt.
    Retorna (contexto_formatado, instrucao, used_docs).
    """
    relevant_docs = []
    
    # Filtra documentos que estão abaixo do limite de distância (scores BAIXOS = perto/bom)
    for doc, score in docs:
        if score < DISTANCE_THRESHOLD:
            relevant_docs.append(doc)

    if not relevant_docs:
        # Modo de fallback (Conhecimento Geral)
        instruction = """
        Não encontrei informações específicas no banco de dados AgroPragas. 
        Por favor, use seu conhecimento geral para responder a esta pergunta. 
        Mantenha a resposta concisa.
        """
        return "", instruction, []
        
    # Modo Híbrido/Local: Usa o contexto local e força a formatação estruturada
    context = "Contexto relevante do Banco de Dados AgroPragas:\n\n"
    for i, doc in enumerate(relevant_docs, 1):
        source = doc.metadata.get('source', 'desconhecido')
        content = doc.page_content.replace('\n', ' ').strip()
        context += f"[Trecho {i} de {source}]\n{content}\n\n"
        
    # REAPLICANDO A INSTRUÇÃO PARA RESPOSTAS CONCISAS (Foco na sugestão do orientador)
    instruction = """
    Com base no Contexto AgroPragas fornecido, responda à pergunta. 
    Se a informação estiver incompleta, complemente com seu conhecimento geral.
    Sua resposta deve ser estruturada APENAS nos seguintes tópicos obrigatórios, utilizando TÓPICOS CURTOS e Markdown para clareza:

1. **Nome Comum e Científico:** Indique o nome e o agente causador (Fungo, Bactéria, Inseto, etc.).
2. **Sintomas Chave:** Liste os 3 a 5 principais sinais visuais de ocorrência.
3. **Fatores de Ocorrência:** Descreva as condições climáticas (temperatura/umidade) que favorecem a praga.
4. **Plano de Ação (Cultural):** Liste 3 a 5 práticas de manejo cultural (Ex: Drenagem, espaçamento, desfolha).
5. **Plano de Ação (Químico/Biológico):** Liste as formas de controle químico e biológico recomendadas, enfatizando a rotação de produtos.

Mantenha a linguagem direta e evite parágrafos longos, utilizando **listas** e **negrito** (Markdown) consistentemente.
    """


    
    return context, instruction, relevant_docs

def create_vector_store(vectorstore_path: str):
    """Cria e retorna o objeto ChromaDB (vectorstore)."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # Inicializa o ChromaDB com a pasta persistente
    return Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embeddings
    )

def generate_with_gemini(
    model: GenerativeModel, 
    question: str,
    context: str,
    instruction: str,
    temperature: float = 0.1
) -> str:
    """Gera a resposta do LLM."""
    prompt = f"""{instruction}

{context}

Pergunta: {question}

Resposta:"""

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature
        )
    )
    return response.text
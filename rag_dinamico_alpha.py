#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
query_rag.py (MODO HÍBRIDO/DINÂMICO)

Interface para consulta da base de conhecimento usando LangChain e Google Gemini.
- Se o contexto local for RELEVANTE, responde com base nele.
- Se o contexto local for IRRELEVANTE, usa o conhecimento geral do LLM.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import google.generativeai as genai

# Importações corrigidas
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from google.generativeai import GenerativeModel 
from langchain_core.documents import Document # Importação para tipagem do LangChain

# ===============================================================
# FUNÇÕES CORE DO RAG
# ===============================================================

# Nível de corte de similaridade (Cosine Similarity): Abaixo disso, o contexto é considerado fraco.
# O score retornado pelo HuggingFaceEmbeddings é a distância euclidiana, não a similaridade.
# Um valor baixo (próximo de 0) indica alta similaridade, mas para simplificar, 
# vamos usar um corte de distância no prompt.
DISTANCE_THRESHOLD = 0.5 

def setup_gemini(api_key: str, model: str = "gemini-2.5-flash") -> GenerativeModel:
    """Configura Gemini com API key e modelo."""
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model)
    except Exception as e:
        raise SystemExit(f"Erro ao configurar o Gemini: {e}")

def format_context_for_gemini(docs: List[Tuple[Document, float]]) -> Tuple[str, str]:
    """
    Formata documentos e determina se o contexto é forte.
    Retorna (contexto_formatado, instrucao_de_prompt).
    """
    relevant_docs = []
    
    # Filtra documentos para ver se algum deles está abaixo do limite de distância (alto score = perto)
    for doc, score in docs:
        if score < DISTANCE_THRESHOLD:
            relevant_docs.append(doc)

    if not relevant_docs:
        # Se nenhum documento for forte, o modelo deve usar o conhecimento geral
        return "", "Não encontrei informações específicas no banco de dados AgroPragas. Por favor, use seu conhecimento geral para responder a esta pergunta."
        
    # Formata o contexto forte
    context = "Contexto relevante do Banco de Dados AgroPragas:\n\n"
    for i, doc in enumerate(relevant_docs, 1):
        source = doc.metadata.get('source', 'desconhecido')
        content = doc.page_content.replace('\n', ' ').strip()
        context += f"[Trecho {i} de {source}]\n{content}\n\n"
        
    # Se encontramos contexto, a instrução é usá-lo, mas sem ser estritamente restritiva
    instruction = "Com base no Contexto AgroPragas fornecido, responda à pergunta. Se a informação estiver incompleta, complemente com seu conhecimento geral."
    
    return context, instruction

def create_vector_store(vectorstore_path: str):
    """Cria e retorna o objeto ChromaDB (vectorstore)."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embeddings
    )
    # Retorna o vectorstore diretamente (usaremos similarity_search_with_score)
    return vectordb

def generate_with_gemini(
    model: GenerativeModel, 
    question: str,
    context: str,
    instruction: str,
    temperature: float = 0.1
) -> str:
    """Generate an answer using Gemini model."""
    
    # Prompt Dinâmico (sem a restrição "Com base apenas...")
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

# ===============================================================
# MAIN E EXECUÇÃO
# ===============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Query the LangChain knowledge base"
    )
    parser.add_argument(
        "--vectorstore",
        default="modelos_ia/vectorstore",
        help="Path to ChromaDB directory"
    )
    parser.add_argument(
        "--query",
        help="Query string. If omitted, runs in interactive mode"
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash", 
        help="Gemini model to use"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for generation"
    )
    args = parser.parse_args()

    if not Path(args.vectorstore).exists():
        raise SystemExit(f"Vector store not found: {args.vectorstore}. Rode o script de indexação primeiro.")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set GOOGLE_API_KEY environment variable to use Gemini (e.g., $env:GOOGLE_API_KEY=\"...\")"
        )
    
    model = setup_gemini(api_key, model=args.model)
    vectorstore = create_vector_store(args.vectorstore)
    
    def process_query(q: str):
        # OBTÉM DOCUMENTOS COM SCORE (distância euclidiana)
        # O score aqui é a DISTÂNCIA, e não a similaridade. Queremos scores BAIXOS.
        docs_with_score = vectorstore.similarity_search_with_score(q, k=4)
        
        print(f"\n--- Busca de Contexto ---\nEncontrei {len(docs_with_score)} documentos candidatos.")
        
        # Formata o contexto e obtém a instrução (dinâmico)
        context, instruction = format_context_for_gemini(docs_with_score)
        
        # Filtra os documentos que realmente foram usados no contexto (score < threshold)
        used_docs = [doc for doc, score in docs_with_score if score < DISTANCE_THRESHOLD]
        
        try:
            # Gera a resposta
            answer = generate_with_gemini(
                model,
                q,
                context,
                instruction, # Passa a instrução dinâmica
                temperature=args.temperature
            )
            print("\n--- RESPOSTA DO GEMINI (Modo Híbrido) ---")
            print(answer)
            
            print("\n--- FONTES CONSULTADAS (Score < 0.5) ---")
            if used_docs:
                for doc in used_docs:
                    print(f"- {doc.metadata.get('source')}")
            else:
                print("Nenhuma fonte local forte utilizada (Resposta baseada em conhecimento geral).")
                
        except Exception as e:
            print(f"\n--- ERRO NA GERAÇÃO DA RESPOSTA ---\nErro: {e}")
    
    if args.query:
        process_query(args.query)
        return

    print(f"\nModo interativo RAG Híbrido iniciado. Vectorstore: {args.vectorstore}")
    print(f"Distância de Corte (Threshold): {DISTANCE_THRESHOLD}. Acima disso, usa conhecimento geral.")
    print("Digite suas perguntas (Ctrl+C para sair)")
    
    while True:
        try:
            q = input("\nPergunta> ")
        except KeyboardInterrupt:
            print("\nSaindo...")
            break
        
        if not q.strip():
            continue
            
        try:
            process_query(q)
        except Exception as e:
            print(f"Erro: {e}")

if __name__ == "__main__":
    main()
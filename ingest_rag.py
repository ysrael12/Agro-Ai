#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingest_rag.py (CORRIGIDO)

Script para indexar os dados da AgroPragas (JSON e TXT) e criar o Vector Store (ChromaDB).
Cria o diretório modelos_ia/vectorstore/ que será usado pelo script query_rag.py.
"""

import argparse
import os
import sys
import logging
import shutil
from pathlib import Path
from typing import List

# CORREÇÕES DE IMPORTAÇÃO:
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# NOVO PACOTE NECESSÁRIO: langchain-text-splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter 

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ingest_rag")

# -------- CONFIGURAÇÃO DE DIRETÓRIOS --------
# Diretórios de origem (onde seus dados brutos estão)
DATA_ROOT = Path("./dataset") # Assumindo que suas pastas 'plans' e 'requests_cache' estão aqui

# Diretório de destino (o Vector Store que o query_rag.py espera)
VECTOR_STORE_PATH = Path("modelos_ia/vectorstore")
# ---------------------------------------------


def load_documents(data_root: Path) -> List:
    """
    Carrega todos os arquivos .json e .txt das pastas especificadas.
    """
    documents = []
    logger.info("Iniciando carregamento de documentos...")

    # Carrega arquivos TXT (Planos de Ação)
    plans_dir = data_root / "plans"
    if plans_dir.is_dir():
        txt_loader = DirectoryLoader(
            str(plans_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            silent_errors=True
        )
        documents.extend(txt_loader.load())
        logger.info(f"  -> Carregados {len(documents)} arquivos TXT dos planos de ação.")
    
    # Carrega arquivos JSON (Cache de Requisições/Metadados)
    json_dir = data_root / "requests_cache"
    if json_dir.is_dir():
        # Usa TextLoader para JSONs, que os carrega como texto puro,
        # ideal para RAG sem um parser de JSON complexo.
        json_loader = DirectoryLoader(
            str(json_dir),
            glob="**/*.json",
            loader_cls=TextLoader,
            silent_errors=True
        )
        json_docs = json_loader.load()
        documents.extend(json_docs)
        logger.info(f"  -> Carregados {len(json_docs)} arquivos JSON de cache.")

    return documents

def create_vector_store(documents: List, vectorstore_path: Path):
    """
    Divide documentos em chunks, cria embeddings e persiste no ChromaDB.
    """
    if not documents:
        logger.warning("Nenhum documento encontrado para indexar. Abortando.")
        return

    logger.info("\n--- 2. DIVISÃO DE TEXTO (CHUNKING) ---")
    
    # Divide o texto em pedaços menores para melhor recuperação
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    logger.info(f"Documentos divididos em {len(texts)} pedaços (chunks).")
    
    # Limpa a pasta anterior do Vector Store (para evitar dados duplicados)
    if vectorstore_path.exists():
        logger.warning(f"Removendo diretório anterior do Vector Store em: {vectorstore_path}")
        shutil.rmtree(vectorstore_path)
    
    logger.info("\n--- 3. CRIAÇÃO DE EMBEDDINGS E INDEXAÇÃO ---")
    
    # Usa um modelo de embeddings do HuggingFace (gratuito e local)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Cria o banco de dados ChromaDB e o popula com os vetores
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=str(vectorstore_path)
    )
    
    # Salva o índice no disco
    vectordb.persist()
    
    logger.info(f"\n✅ Indexação CONCLUÍDA com sucesso.")
    logger.info(f"Vector Store persistido em: {vectorstore_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Indexador de dados da AgroPragas para criação do Vector Store (ChromaDB)."
    )
    parser.add_argument(
        "--data_root",
        default=str(DATA_ROOT),
        help="Diretório raiz contendo as pastas 'plans' e 'requests_cache'."
    )
    args = parser.parse_args()

    # 1. Carregar
    documents = load_documents(Path(args.data_root))
    
    # 2. Indexar
    create_vector_store(documents, VECTOR_STORE_PATH)


if __name__ == "__main__":
    main()

""" 
rag != fine tuning
"""
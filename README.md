RAG (Retrieval-Augmented Generation) — construção de índice local

Este diretório contém utilitários para criar um índice de recuperação (FAISS)
e um pequeno fluxo RAG a partir dos arquivos de texto presentes no projeto.

O fluxo padrão:
- `build_corpus.py` — percorre `dataset_v2_ofc/` (plans, requests_cache, metadata.csv, label.txt), extrai textos, gera embeddings com `sentence-transformers` e constrói um índice FAISS.
- `query_rag.py` — carrega o índice e busca os documentos mais relevantes para uma consulta; opcionalmente chama a API do OpenAI (se variável de ambiente OPENAI_API_KEY estiver definida) para gerar uma resposta usando os documentos recuperados como contexto.

Instalação (recomendo criar um virtualenv):

Windows PowerShell:
```powershell
$env:GOOGLE_API_KEY = "SUA CHAVE"
python modelos_ia/query_rag.py --vectorstore modelos_ia/vectorstore --openai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r modelos_ia/requirements.txt
```

Exemplos de uso:

# construir o índice (leva alguns minutos)
python modelos_ia/build_corpus.py --data-dir dataset_v2_ofc --out modelos_ia/index

# buscar e imprimir documentos mais relevantes
python modelos_ia/query_rag.py --index modelos_ia/index

# buscar e pedir para o OpenAI gerar uma resposta (é necessário definir OPENAI_API_KEY)
python modelos_ia/query_rag.py --index modelos_ia/index --openai

Observações:
- O script usa o modelo de embedding `sentence-transformers/all-MiniLM-L6-v2` por padrão.
- Se não quiser chamar o OpenAI, a geração é opcional — o script sempre mostra os trechos recuperados.

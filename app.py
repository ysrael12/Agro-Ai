# app.py (Final com Paleta de Cores Atualizada)
import streamlit as st
import os
import sys
import time
from pathlib import Path
import warnings
from io import BytesIO
import base64 # Necessário para injetar a logo via CSS

# Dependências do PDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Image

# O Streamlit lida com multitarefas, então ignoramos warnings de pacotes antigos no topo
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Adiciona o diretório atual ao PATH para importar query_rag_functions
sys.path.append(str(Path(__file__).parent / "modelos_ia")) 

try:
    # Importa as funções dos módulos (Assumindo que estão na pasta modelos_ia)
    from query_rag_functions import (
        setup_gemini, 
        create_vector_store, 
        format_context_for_gemini, 
        generate_with_gemini,
        DISTANCE_THRESHOLD
    )
    from classifier_inference import (
        load_and_configure_classifier,
        preprocess_image,
        predict,
        MODEL_PATH,
        DATASET_ROOT
    )
    
    # Define o path do VectorStore
    VECTOR_STORE_PATH = Path("modelos_ia/vectorstore")
    
except ImportError as e:
    st.error(f"Erro ao carregar módulos. Verifique se 'query_rag_functions.py' e 'classifier_inference.py' existem na pasta 'modelos_ia' e se todas as bibliotecas estão instaladas: {e}")
    st.stop()


# ===============================================================
# CONFIGURAÇÃO DE TEMA E LOGO
# ===============================================================

# PALETA DE CORES ATUALIZADA
COLORS = {
    "background_light": "#FFFFFF",  # Branco (Fundo principal)
    "background_dark": "#386641",   # Verde Escuro (Sidebar)
    "primary": "#A7C957",           # Verde Lima (Cor principal/botões)
    "secondary": "#386641",         # Verde Escuro (Destaque/Download)
    "text_dark": "#000000",         # Preto (Texto principal)
    "text_light": "#FFFFFF"         # Branco (Texto na sidebar/botões escuros)
}

LOGO_FILE = Path("assets/agro_ai_logo.png") # Caminho da logo

def inject_custom_css():
    """Injeta CSS customizado para aplicar a paleta de cores."""
    
    # Prepara a logo para injeção CSS (se existir)
    logo_b64 = ""
    if LOGO_FILE.exists():
        with open(LOGO_FILE, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode("utf-8")

    # CSS para tema e logo
    css = f"""
    <style>
        /* Cor de fundo principal */
        .stApp {{
            background-color: {COLORS['background_light']};
            color: {COLORS['text_dark']};
        }}

        /* Cor de fundo da barra lateral */
        .stSidebar {{
            background-color: {COLORS['background_dark']};
            color: {COLORS['text_light']};
        }}
        
        /* Cor de texto dentro da sidebar (títulos e navegação) */
        .stSidebar .stRadio div, .stSidebar h2, .stSidebar label {{
            color: {COLORS['text_light']} !important;
        }}

        /* Cor primária para botões e sliders (Verde Lima) */
        .stButton>button, .stSlider>div>div:first-child {{
            background-color: {COLORS['background_dark']};
            color: {COLORS['text_dark']};
            border-color: {COLORS['secondary']};
        }}
        
        /* Ajuste do botão de download (Verde Escuro) */
        .stDownloadButton>button {{
            background-color: {COLORS['secondary']};
            color: {COLORS['text_light']};
            border-color: {COLORS['secondary']};
        }}
        
        /* Imagem da Logo na Sidebar */
        [data-testid="stSidebarHeader"] {{
            background-image: url("data:image/png;base64,{logo_b64}");
            background-size: 80px;
            background-repeat: no-repeat;
            background-position: left center;
            height: 100px;
            padding-top: 15px;
            font-size: 1.5rem;
            color: {COLORS['text_light']};
            text-align: right;
            border-bottom: 2px solid {COLORS['primary']};
        }}
        
        /* Títulos de seção */
        h1, h2, h3, h4 {{
            color: {COLORS['secondary']};
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ===============================================================
# FUNÇÕES DE SESSÃO E CACHE (Não alteradas)
# ===============================================================

@st.cache_resource
def initialize_rag_components():
    """Inicializa LLM e Vector Store."""
    st.write("Inicializando componentes RAG...")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Variável de ambiente GOOGLE_API_KEY não definida.")
        st.stop()

    try:
        model = setup_gemini(api_key, model="gemini-2.5-flash")
    except Exception as e:
        st.error(f"Falha ao configurar o modelo Gemini: {e}")
        st.stop()
    
    if not VECTOR_STORE_PATH.exists():
        st.error(f"Vector Store não encontrado em: {VECTOR_STORE_PATH}. Rode o script de indexação primeiro!")
        st.stop()
        
    vectorstore = create_vector_store(str(VECTOR_STORE_PATH))
    
    return model, vectorstore

@st.cache_resource
def initialize_classifier():
    """Carrega o modelo de classificação EfficientNet."""
    st.write("Carregando modelo EfficientNetV2-M...")
    if not MODEL_PATH.exists():
        st.warning(f"Modelo treinado não encontrado em: {MODEL_PATH.resolve()}")
        st.warning("A página de Classificação não funcionará.")
        return None, None
        
    try:
        model, class_names = load_and_configure_classifier(Path(__file__).parent)
        st.success("Modelo EfficientNetV2-M carregado com sucesso.")
        return model, class_names
    except Exception as e:
        st.error(f"Falha ao carregar o classificador: {e}")
        return None, None

# ===============================================================
# FUNÇÕES DE UTILIDADE (PDF e RAG)
# ===============================================================

def filter_non_agri_query(model, prompt: str) -> bool:
    """Usa o LLM para verificar se a pergunta está no domínio agrícola/pragas."""
    check_prompt = f"""
    A pergunta a seguir está relacionada a Agricultura, Pragas, Doenças de Plantas, Manejo, Fruticultura, ou temas diretamente associados?
    Pergunta: "{prompt}"
    Responda APENAS 'SIM' ou 'NAO'.
    """
    
    try:
        response = generate_with_gemini(model, check_prompt, context="", instruction="Aja como um classificador de tópicos.", temperature=0.0).strip().upper()
        return response == 'SIM'
    except Exception:
        return True


def run_rag_query(model, vectorstore, prompt: str, temperature: float, k_value: int, target_container=None):
    """Executa a lógica RAG e retorna a resposta e os detalhes."""
    
    if target_container is None:
        target_container = st

    with st.spinner(f"Buscando {k_value} documentos relevantes..."):
        docs_with_score = vectorstore.similarity_search_with_score(prompt, k=k_value)

    context, instruction, used_docs = format_context_for_gemini(docs_with_score)

    with st.spinner("Gerando resposta com Gemini..."):
        try:
            answer = generate_with_gemini(
                model, 
                prompt, 
                context, 
                instruction, 
                temperature=temperature
            )
            return answer, docs_with_score, used_docs, instruction

        except Exception as e:
            return f"Erro na Geração do LLM: {e}", docs_with_score, used_docs, instruction


# app.py (Trecho Corrigido para Geração de PDF)
# ...
import re # Certifique-se que o 're' está importado no topo do app.py
# ...

def generate_pdf_report(report_text: str, class_name: str) -> bytes:
    """Gera um PDF na memória (BytesIO) com o texto do relatório, corrigindo tags Markdown."""
    
    buffer = BytesIO()
    
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    
    styles.add(ParagraphStyle(name='HeadingPraga', fontSize=18, spaceAfter=12, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='SubHeading', fontSize=12, spaceAfter=6, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='BodyTextCustom', fontSize=10, spaceAfter=6, leading=12, fontName='Helvetica'))

    story = []
    
    # Adicionar Logo (mantido o caminho de exemplo)
    logo_path = Path(__file__).parent / "assets" / "agro_ai_logo.png"
    if logo_path.exists():
        logo = Image(str(logo_path), width=100, height=30) 
        story.append(logo)
        story.append(Spacer(1, 12)) 
    
    story.append(Paragraph("Relatório de Manejo AgroPragas IA", styles['Heading1']))
    story.append(Paragraph(f"Praga/Doença Detectada: <b>{class_name}</b>", styles['HeadingPraga']))
    story.append(Spacer(1, 18))
    
    story.append(Paragraph("--- Plano de Ação e Sintomas ---", styles['SubHeading']))
    
    # 🌟 ETAPAS DE CORREÇÃO E LIMPEZA 🌟
    
    # 1. Substitui quebras de linha por tag HTML
    formatted_text = report_text.replace('\n', '<br/>')

    # 2. Substitui listas (hífens ou asteriscos) por HTML de lista seguro
    # Isso impede que o negrito aninhe com a estrutura de lista
    formatted_text = re.sub(r'<br/>\s*[\*-] ', '<br/>&bull; ', formatted_text) 
    formatted_text = formatted_text.replace('* ', '&bull; ') # Limpa listas que não começam com <br/>
    formatted_text = formatted_text.replace('- ', '&bull; ') 

    # 3. Traduz negrito do Markdown (**) para tag HTML (<b>) APÓS a limpeza de listas
    # O regex r'\1' captura o texto dentro dos ** **
    formatted_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', formatted_text)
    
    # 4. Remove a tag <br/> se for a primeira coisa no início (limpeza)
    formatted_text = formatted_text.strip('<br/>')

    # Fim das correções

    story.append(Paragraph(formatted_text, styles['BodyTextCustom']))
    
    story.append(Spacer(1, 24))
    story.append(Paragraph(f"Relatório Gerado em: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles['Italic']))
    
    try:
        doc.build(story)
    except ValueError as e:
        # Se falhar, tenta usar o texto plano, mas salva o erro para o Streamlit
        st.error(f"Erro ao gerar PDF (Reportlab): {e}. Tentei gerar um PDF simples.")
        # Retorna um PDF de fallback para não falhar completamente
        c = canvas.Canvas(buffer, pagesize=letter)
        c.drawString(72, 750, "Erro de formatação. O relatório completo está abaixo.")
        c.drawString(72, 730, f"Erro: {e}")
        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer.getvalue()


    buffer.seek(0)
    return buffer.getvalue()


# ===============================================================
# PÁGINAS DO STREAMLIT
# ===============================================================

def rag_page():
    st.title("🌱 Chatbot de Consulta Agro.ai")
    st.markdown("Faça perguntas sobre os planos de manejo e metadados das pragas.")

    model, vectorstore = initialize_rag_components()
    
    # Sidebar para configurações
    st.sidebar.header("⚙️ Configurações RAG")

    k_value = st.sidebar.slider("Nº de Documentos (K)", 1, 10, 4, key='rag_k')
    st.session_state.temperature = st.sidebar.slider("Temperatura do LLM", 0.0, 1.0, 0.1, key='rag_temp')

    # Lógica de consulta automática (se vier da página de classificação)
    if 'auto_query_text' in st.session_state and st.session_state.auto_query_text:
        prompt = st.session_state.auto_query_text
        del st.session_state.auto_query_text
        st.info(f"Consulta RAG automática disparada para: **{prompt}**")
        
        answer, docs_with_score, used_docs, instruction = run_rag_query(
            model, vectorstore, prompt, st.session_state.temperature, k_value
        )
        st.subheader("Relatório de Manejo (Gerado pela Classificação)")
        st.success(answer)
        
    else:
        prompt = st.chat_input("Digite sua pergunta sobre as pragas...")

    if prompt:
        # 1. FILTRO TEMÁTICO
        if not filter_non_agri_query(model, prompt):
            st.warning("A pergunta não está relacionada a temas agrícolas ou de pragas. Por favor, mantenha o foco no domínio AgroPragas.")
            return

        # 2. Executa a Consulta RAG
        answer, docs_with_score, used_docs, instruction = run_rag_query(
            model, vectorstore, prompt, st.session_state.temperature, k_value
        )

        # 3. Exibição
        st.subheader("Resposta")
        st.info(answer)

        with st.expander("Detalhes do RAG e Fontes"):
            st.write(f"**Instrução para o LLM:** {instruction}")
            st.write(f"**Distância de Corte (Threshold):** {DISTANCE_THRESHOLD}")
            
            if used_docs:
                st.subheader("Fontes Locais Utilizadas:")
                for doc in used_docs:
                    source = doc.metadata.get('source', 'N/A')
                    score = next((s for d, s in docs_with_score if d == doc), 'N/A')
                    st.markdown(f"**{source}** (Distância: {score:.4f})")
                    st.caption(doc.page_content[:300] + "...")
            else:
                st.warning("Nenhuma fonte local forte utilizada.")


def classification_page():
    st.title("📸 Módulo de Previsão: Classificação de Pragas")
    st.markdown("Faça o upload de uma imagem para identificar a praga ou doença.")

    model_rag, vectorstore = initialize_rag_components() 
    model_cls, class_names = initialize_classifier() 

    if model_cls is None:
        st.warning("O classificador não foi carregado. Verifique os logs de erro.")
        return

    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_file = st.file_uploader("Faça o upload de uma imagem de praga ou sintoma (.jpg, .png):", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Imagem Carregada", use_column_width=True)
            
            if st.button("Executar Classificação", use_container_width=True):
                st.session_state.classification_result = None 
                with st.spinner("Classificando imagem..."):
                    try:
                        input_tensor = preprocess_image(uploaded_file)
                        class_name, probability, top_predictions = predict(model_cls, input_tensor, class_names)
                        
                        st.session_state.classification_result = {
                            "class_name": class_name,
                            "probability": probability,
                            "top_predictions": top_predictions
                        }
                    except Exception as e:
                        st.error(f"Erro ao processar a imagem: {e}")
            
    with col2:
        if 'classification_result' in st.session_state and st.session_state.classification_result:
            result = st.session_state.classification_result
            class_name = result['class_name']
            
            # Exibe resultado principal
            st.subheader("Resultado da Previsão Principal")
            st.success(f"Praga/Doença Detectada: {class_name}")
            st.metric("Confiança", f"{result['probability']:.2%}")
            
            # Tabela de Top 5 Previsões
            with st.expander("Top 5 Previsões"):
                top_data = {
                    "Praga/Doença": [p[0] for p in result['top_predictions']],
                    "Confiança": [f"{p[1]:.2%}" for p in result['top_predictions']]
                }
                st.table(top_data)
                
            # GERAÇÃO DE RELATÓRIO PÓS-CLASSIFICAÇÃO
            st.subheader(f"Relatório de Manejo para {class_name}")
            
            # Define a query para o RAG
            rag_prompt = f"Gere um plano de manejo e formas de controle detalhadas, sintomas e ocorrência para a doença/praga: {class_name}. Sua resposta deve ser estruturada em tópicos curtos e negrito."
            
            # Executa o RAG
            report_text, _, _, _ = run_rag_query(
                model_rag, vectorstore, rag_prompt, 0.1, 4, target_container=st.container()
            )
            
            # Exibe o relatório
            st.markdown(report_text) 
            st.caption("Relatório gerado pelo LLM baseado nas fontes locais da AgroPragas.")

            # BOTÃO DE DOWNLOAD PDF
            if report_text and not report_text.startswith("Erro"):
                
                # Gera o PDF a partir do texto do relatório
                pdf_bytes = generate_pdf_report(report_text, class_name)
                
                # Exibe o botão de download
                st.download_button(
                    label="⬇️ Baixar Relatório (PDF)",
                    data=pdf_bytes,
                    file_name=f"Relatorio_Manejo_{class_name}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )


# ===============================================================
# NAVEGAÇÃO MULTIPÁGINA
# ===============================================================

def main_app():
    # 🌟 INJEÇÃO DO CSS CUSTOMIZADO NO INÍCIO
    inject_custom_css()
    
    st.set_page_config(
        page_title="AgroPragas IA",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.sidebar.title("Navegação")
    page = st.sidebar.radio("Ir para:", ["Consulta RAG", "Módulo de Previsão"])

    if page == "Consulta RAG":
        rag_page()
    elif page == "Módulo de Previsão":
        classification_page()

if __name__ == "__main__":
    main_app()
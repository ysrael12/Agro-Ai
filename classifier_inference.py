# classifier_inference.py
import streamlit as st # Necessário para st.cache_resource
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from PIL import Image
import io
import os
from pathlib import Path
from typing import List, Tuple

# --- CONFIGURAÇÕES DO MODELO ---
MODEL_ARCH = 'efficientnet_v2_m'
PROJECT_NAME = 'agropragas_efficientnet_v2m_robust'

# Caminhos (ajustados para serem resolvidos no contexto da execução do app.py)
# Estes caminhos são relativos ao diretório onde o app.py está rodando.
MODEL_PATH = Path(f'agropragas_efficientnet_v2m_robust_best_model.pth')
DATASET_ROOT = Path('./combined_agropragas_dataset/test') 
# -------------------------------

# Verifica se a GPU está disponível
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_and_configure_classifier(base_dir: Path):
    """
    Carrega o modelo EfficientNetV2-M, os pesos treinados e os nomes das classes.
    Usa base_dir para resolver o caminho completo do modelo.
    """
    
    # Resolve o caminho completo do arquivo .pth
    full_model_path = base_dir / MODEL_PATH 

    try:
        # 1. Obter nomes de classes e contagem da estrutura de pastas
        # O ImageFolder é a maneira mais confiável de garantir que a ordem das classes (labels) 
        # seja a mesma usada durante o treinamento.
        dataset = datasets.ImageFolder(DATASET_ROOT)
        class_names = dataset.classes
        num_classes = len(class_names)
        
    except Exception as e:
        raise ValueError(f"Falha ao carregar classes do dataset em {DATASET_ROOT}: {e}")

    # 2. Carregar a arquitetura do modelo (sem pesos pré-treinados)
    model = models.efficientnet_v2_m(weights=None)
    
    # 3. Reconfigurar a camada final (Classificador), conforme o treinamento
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    
    # 4. Carregar os pesos treinados
    if not full_model_path.exists():
        raise FileNotFoundError(f"Arquivo do modelo treinado não encontrado: {full_model_path.resolve()}")
    
    try:
        model.load_state_dict(torch.load(full_model_path, map_location=device))
        model.eval() # Define o modelo para o modo de avaliação
        model = model.to(device)
        return model, class_names
    except Exception as e:
        raise Exception(f"Erro ao carregar os pesos do modelo: {e}. Verifique o arquivo .pth.")

def preprocess_image(uploaded_file) -> torch.Tensor:
    """Processa o arquivo carregado do Streamlit para o formato de entrada do modelo."""
    
    # Usa as mesmas transformações determinísticas da fase de validação/teste
    weights = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
    
    preprocess = transforms.Compose([
        transforms.Resize(weights.transforms().resize_size),
        transforms.CenterCrop(weights.transforms().crop_size),
        transforms.ToTensor(),
        transforms.Normalize(weights.transforms().mean, weights.transforms().std)
    ])
    
    # Converte o arquivo carregado (bytes) em objeto Image PIL
    # O Streamlit armazena o upload em um objeto File, precisamos ler os bytes.
    uploaded_file.seek(0) # Retorna ao início do arquivo
    img = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')
    
    # Adiciona a dimensão do batch (Batch size = 1)
    return preprocess(img).unsqueeze(0).to(device)

def predict(model: nn.Module, input_tensor: torch.Tensor, class_names: List[str]) -> Tuple[str, float, List[Tuple[str, float]]]:
    """Realiza a inferência, retorna a classe principal e as top 5 previsões."""
    
    with torch.no_grad():
        output = model(input_tensor)
        
        # Converte logits (saída da rede) para probabilidades
        probabilities = nn.functional.softmax(output, dim=1).cpu().squeeze()
        
        # Obtém as 5 principais previsões
        # Verifica se o número de classes é menor que 5 para evitar erro
        top_k = min(5, len(class_names))
        top_p, top_class_indices = probabilities.topk(top_k)

        top_predictions = []
        for index, prob in zip(top_class_indices.numpy(), top_p.numpy()):
            top_predictions.append((class_names[index], prob))
            
        predicted_class_name = top_predictions[0][0]
        predicted_probability = top_predictions[0][1]
        
    return predicted_class_name, predicted_probability, top_predictions

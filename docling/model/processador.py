import os
import json
from pathlib import Path
import easyocr
from huggingface_hub import hf_hub_download
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Define o caminho base do script para localizar outros diretórios
script_dir = Path(__file__).resolve().parent
# Constrói o caminho para a pasta de guias
guias_path = script_dir.parent / "guias_sigeduc"
# Define e cria o diretório de saída para os arquivos JSON
output_path = script_dir.parent / "json_output"
output_path.mkdir(parents=True, exist_ok=True)

# Define e cria o diretório de artefatos dentro da pasta 'docling'
artifacts_path = script_dir.parent / "artifacts"
artifacts_path.mkdir(parents=True, exist_ok=True)

# --- Passo de Aquecimento para o Modelo de Layout (YOLOX) ---
print("Verificando modelo de análise de layout...")
try:
    repo_id = "unstructuredio/yolox"
    model_file = "model.safetensors"
    config_file = "config.json"
    
    # Verifica e baixa o model.safetensors se não existir
    if not (artifacts_path / model_file).exists():
        print(f"Baixando {model_file} de {repo_id}...")
        hf_hub_download(repo_id=repo_id, filename=model_file, local_dir=artifacts_path)
        print("Download concluído.")

    # Verifica e baixa o config.json se não existir
    if not (artifacts_path / config_file).exists():
        print(f"Baixando {config_file} de {repo_id}...")
        hf_hub_download(repo_id=repo_id, filename=config_file, local_dir=artifacts_path)
        print("Download concluído.")
        
    print("Modelo de análise de layout está pronto.")
except Exception as e:
    print(f"Ocorreu um erro durante o download do modelo de layout: {e}")
    # exit()

# --- Passo de Aquecimento para o EasyOCR ---
# Força o download dos modelos para o diretório de artefatos se eles não existirem.
print("Verificando modelos do EasyOCR...")
try:
    # Inicializa o EasyOCR apontando para o diretório de modelos.
    # O EasyOCR fará o download dos modelos para este local se não os encontrar.
    easyocr_model_path = artifacts_path / "EasyOcr"
    easyocr.Reader(['pt', 'en'], model_storage_directory=str(easyocr_model_path), download_enabled=True)
    print("Modelos do EasyOCR estão prontos.")
except Exception as e:
    print(f"Ocorreu um erro durante a verificação/download dos modelos do EasyOCR: {e}")
    # Decide se quer parar a execução ou continuar tentando
    # exit() # Descomente se quiser parar o script em caso de falha no download

# Configura as opções do pipeline, agora que os modelos já devem existir localmente
pipeline_options = PdfPipelineOptions(
    artifacts_path=str(artifacts_path),
    easy_ocr_options=EasyOcrOptions(download_enabled=False) # Pode ser False agora
)

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)

# Verifica se o diretório de guias existe
if not guias_path.is_dir():
    print(f"Erro: O diretório '{guias_path}' não foi encontrado.")
else:
    print(f"Buscando arquivos PDF em: {guias_path}")
    # Lista todos os arquivos PDF no diretório
    pdf_files = list(guias_path.glob("*.pdf"))

    if not pdf_files:
        print("Nenhum arquivo PDF encontrado no diretório.")
    else:
        print(f"Encontrados {len(pdf_files)} arquivos PDF. Processando...")
        # Itera sobre cada arquivo PDF e o processa
        for pdf_path in pdf_files:
            print(f" - Processando: {pdf_path.name}")
            try:
                # Chama o método de conversão (supondo que retorna um dicionário ou objeto serializável)
                resultado = doc_converter.convert(source=str(pdf_path))

                # Define o caminho do arquivo de saída JSON
                json_output_path = output_path / f"{pdf_path.stem}.json"

                # Salva o resultado em um arquivo JSON
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    # Supondo que o resultado seja um dicionário. Se for um objeto, pode ser necessário
                    # chamar um método como .to_dict() ou .json() antes de usar json.dump.
                    json.dump(resultado, f, ensure_ascii=False, indent=4)
                
                print(f"   -> Salvo em: {json_output_path}")

            except Exception as e:
                print(f"   Erro ao processar {pdf_path.name}: {e}")

print("\nProcessamento concluído.")
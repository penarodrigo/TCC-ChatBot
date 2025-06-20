# c:\Users\c057545\Downloads\CBCODIMPROVED\tests\test_document_processor.py
from pathlib import Path
import pytest
from rag_gemini_improved import DocumentProcessor

# Crie uma pasta tests/test_data e coloque arquivos de exemplo lá
TEST_DATA_DIR = Path(__file__).parent / "test_data"
SAMPLE_TXT_FILE = TEST_DATA_DIR / "sample.txt"
SAMPLE_PDF_FILE = TEST_DATA_DIR / "sample.pdf" # Você precisará de um PDF de exemplo

@pytest.fixture(scope="session", autouse=True)
def create_sample_files():
    """Cria arquivos de teste de exemplo se não existirem."""
    TEST_DATA_DIR.mkdir(exist_ok=True)
    if not SAMPLE_TXT_FILE.exists():
        with open(SAMPLE_TXT_FILE, "w", encoding="utf-8") as f:
            f.write("Este é um texto de exemplo.\nCom múltiplas linhas.")
    # Para o PDF, você precisaria adicionar um arquivo PDF real à pasta tests/test_data
    # Se não, pule o teste do PDF ou mock a biblioteca pdfplumber
    # if not SAMPLE_PDF_FILE.exists():
    #     print(f"AVISO: Arquivo PDF de exemplo não encontrado em {SAMPLE_PDF_FILE}")
    #     # Crie um PDF simples programaticamente se necessário, ou adicione um manualmente
    yield # Permite que os testes rodem
    # Limpeza opcional após os testes (geralmente não necessário para arquivos de dados de teste)
    # if SAMPLE_TXT_FILE.exists():
    #     SAMPLE_TXT_FILE.unlink()


def test_extract_text_from_txt():
    if not SAMPLE_TXT_FILE.exists():
        pytest.skip("Arquivo TXT de exemplo não encontrado, pulando teste.")

    content = DocumentProcessor.extract_text(SAMPLE_TXT_FILE)
    assert "Este é um texto de exemplo." in content
    assert "Com múltiplas linhas." in content

def test_extract_text_from_pdf():
    if not SAMPLE_PDF_FILE.exists():
        pytest.skip("Arquivo PDF de exemplo não encontrado, pulando teste.")

    # Esta é uma suposição do conteúdo. Você precisará de um PDF real
    # e saber qual texto esperar.
    content = DocumentProcessor.extract_text(SAMPLE_PDF_FILE)
    # Exemplo: assert "Algum texto esperado do seu PDF" in content
    assert isinstance(content, str) # Pelo menos verifica se retorna uma string

def test_unsupported_file_type():
    unsupported_file = TEST_DATA_DIR / "sample.unsupported"
    if not unsupported_file.exists():
         with open(unsupported_file, "w") as f: # Cria um arquivo vazio para o teste
            f.write("dummy")

    content = DocumentProcessor.extract_text(unsupported_file)
    assert content == ""
    # Opcional: verificar se um warning foi logado (requer configuração de captura de log do pytest)

    if unsupported_file.exists(): # Limpa o arquivo criado para o teste
        unsupported_file.unlink()


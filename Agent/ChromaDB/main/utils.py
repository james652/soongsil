# utils.py
# - PDF 텍스트 추출
# - 토큰 기준 청크(1000) + 오버랩(200)
# - OpenAI 임베딩 생성
# - ChromaDB에 배치 추가

from typing import List, Dict, Optional
import fitz  # PyMuPDF
import tiktoken
from openai import OpenAI
import math

# ---- 1) PDF 텍스트 추출 ------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    PyMuPDF로 PDF 전체 텍스트를 추출합니다.
    """
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        parts.append(page.get_text())
    return "\n".join(parts)

# ---- 2) 토큰 기준 청크 분할 (1000 토큰, 200 오버랩) --------------------------
def chunk_by_tokens(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    model_encoding: str = "cl100k_base",
) -> List[str]:
    """
    tiktoken으로 정확히 '토큰' 기준 분할.
    - chunk_size: 1000
    - overlap: 200
    """
    assert chunk_size > 0 and overlap >= 0 and overlap < chunk_size, \
        "chunk_size>0, 0<=overlap<chunk_size 여야 합니다."

    enc = tiktoken.get_encoding(model_encoding)
    tokens = enc.encode(text)

    chunks: List[str] = []
    start = 0
    n = len(tokens)

    while start < n:
        end = min(start + chunk_size, n)
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)

        if end >= n:
            break
        # 다음 시작점: 오버랩을 고려해 뒤로 200토큰 물러서 시작
        start = end - overlap
        if start < 0:
            start = 0

    return chunks

# ---- 3) OpenAI 임베딩 --------------------------------------------------------
def embed_texts_openai(
    texts: List[str],
    model: str = "text-embedding-3-large",
    client: Optional[OpenAI] = None,
    batch_size: int = 64,
) -> List[List[float]]:
    """
    OpenAI 임베딩을 배치로 생성하여 반환합니다.
    - model: text-embedding-3-large (3072차원) / text-embedding-3-small (1536차원)
    - batch_size: API 호출당 입력 개수
    - 환경변수 OPENAI_API_KEY 사용
    """
    if client is None:
        client = OpenAI()

    vectors: List[List[float]] = []
    total = len(texts)
    if total == 0:
        return vectors

    num_batches = math.ceil(total / batch_size)
    for b in range(num_batches):
        s = b * batch_size
        e = min((b + 1) * batch_size, total)
        batch = texts[s:e]
        resp = client.embeddings.create(model=model, input=batch)
        # resp.data는 입력 순서대로 반환됩니다.
        vectors.extend([d.embedding for d in resp.data])
    return vectors

# ---- 4) 배치 추가: ChromaDB에 문서 + 임베딩 ----------------------------------
def add_chunks_with_embeddings(
    collection,
    chunks: List[str],
    metadata: Dict,
    id_prefix: str,
    openai_model: str = "text-embedding-3-large",
    batch_size: int = 64,
    client: Optional[OpenAI] = None,
) -> None:
    """
    - chunks에 대해 OpenAI 임베딩을 생성하고
    - Chroma 컬렉션에 (documents, embeddings, metadatas, ids)로 배치 추가합니다.
    """
    if not chunks:
        return

    # 임베딩 계산
    vectors = embed_texts_openai(chunks, model=openai_model, batch_size=batch_size,client=client)

    # metadatas & ids 준비
    metadatas = [metadata for _ in chunks]
    ids = [f"{id_prefix}_chunk{i}" for i in range(len(chunks))]

    # Chroma에 추가(한 번에 추가해도 되고, 길면 나눠서 추가해도 됩니다)
    collection.add(
        documents=chunks,
        embeddings=vectors,
        metadatas=metadatas,
        ids=ids,
    )

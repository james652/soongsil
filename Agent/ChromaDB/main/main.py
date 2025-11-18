import chromadb
from openai import OpenAI
from utils import extract_text_from_pdf, chunk_by_tokens, add_chunks_with_embeddings

# (선택) 호스트/포트 지정 가능
chroma_client = chromadb.HttpClient(host="127.0.0.1", port=8000)

# 컬렉션 생성/가져오기
collection = chroma_client.get_or_create_collection(name="papers")

# OpenAI 클라이언트 (API 키 전달)
openai_client = OpenAI(api_key="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")   # ← 환경변수 쓰면 이 줄 생략 가능

pdf1 = "/home/jun/work/soongsil/Agent/ChromaDB/paper/Abbasi_BrainWash_A_Poisoning_Attack_to_Forget_in_Continual_Learning_CVPR_2024_paper.pdf"
pdf2 = "/home/jun/work/soongsil/Agent/ChromaDB/paper/NeurIPS-2021-accumulative-poisoning-attacks-on-real-time-data-Paper.pdf"
pdf3 = "/home/jun/work/soongsil/Agent/ChromaDB/paper/NeurIPS-2021-accumulative-poisoning-attacks-on-real-time-data-Paper.pdf"
pdf4 = "/home/jun/work/soongsil/Agent/ChromaDB/paper/NeurIPS_2023-label_poisoning_is_all_you_need_Paper_Conference.pdf"

metadata_1 = {
    "title": "BrainWash: A Poisoning Attack to Forget in Continual Learning",
    "authors": "Ali Abbasi; Parsa Nooralinejad; Hamed Pirsiavash; Soheil Kolouri",
    "year": 2024,
}
metadata_2 = {
    "title": "Accumulative Poisoning Attacks on Real-time Data",
    "authors": "논문2 저자1; 논문2 저자2",
    "year": 2021,
}
metadata_3 = {
    "title": "Accumulative Poisoning Attacks on Real-time Data (dup)",
    "authors": "논문2 저자1; 논문2 저자2",
    "year": 2021,
}
metadata_4 = {
    "title": "Label Poisoning Is All You Need",
    "authors": "NeurIPS 2023 Authors",
    "year": 2023,
}

def process_pdf(pdf_path: str, meta: dict, id_prefix: str):
    print(f"[+] Extracting: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)

    print(f"[+] Chunking into 1000-token chunks with 200-token overlap...")
    chunks = chunk_by_tokens(text, chunk_size=1000, overlap=200)

    print(f"[+] Adding {len(chunks)} chunks with OpenAI embeddings to Chroma ({id_prefix})")
    add_chunks_with_embeddings(
        collection=collection,
        chunks=chunks,
        metadata=meta,
        id_prefix=id_prefix,
        openai_model="text-embedding-3-large", 
        batch_size=64,
        client=openai_client,  
    )

process_pdf(pdf1, metadata_1, "paper1")
process_pdf(pdf2, metadata_2, "paper2")
process_pdf(pdf3, metadata_3, "paper3")
process_pdf(pdf4, metadata_4, "paper4")

print(" 모든 논문이 Chroma DB에 성공적으로 추가되었습니다.")

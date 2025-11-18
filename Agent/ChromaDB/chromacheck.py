# chromacheck.py
import sys
import chromadb
from openai import OpenAI

# (선택) 서버 호스트/포트가 다르면 아래 주석 해제해서 지정
# chroma_client = chromadb.HttpClient(host="127.0.0.1", port=8000)
chroma_client = chromadb.HttpClient()

try:
    collection = chroma_client.get_collection("papers")
    print(f"컬렉션에 저장된 아이템 개수: {collection.count()}")
except Exception as e:
    print(f"컬렉션을 가져오는 중 오류가 발생했습니다: {e}")
    print("→ Chroma 서버가 켜져 있는지, 컬렉션 이름이 맞는지, 데이터가 저장되었는지 확인하세요.")
    sys.exit(1)

# OpenAI API 키는 환경변수 OPENAI_API_KEY로 설정해두는 것을 권장합니다.
# export OPENAI_API_KEY="sk-...."
oa = OpenAI()  # 환경변수에서 키 자동 로드

query_text = "poisoning attack"

# 질의 텍스트를 문서와 동일한 모델로 임베딩(3072차원) → query_embeddings 로 검색
emb = oa.embeddings.create(
    model="text-embedding-3-large",   # 문서 저장에 쓴 모델과 동일해야 함
    input=[query_text]
).data[0].embedding

try:
    results = collection.query(
        query_embeddings=[emb],
        n_results=5,
        include=["metadatas", "documents", "distances"],
    )
except Exception as e:
    print(f"쿼리 중 오류가 발생했습니다: {e}")
    print("→ 임베딩 차원 불일치가 의심되면, 저장에 사용한 모델과 동일한 임베딩 모델로 질의하세요.")
    sys.exit(1)

print(f"\n'{query_text}'에 대한 결과입니다.\n")

ids = results.get("ids", [[]])[0]
distances = results.get("distances", [[]])[0]
metadatas = results.get("metadatas", [[]])[0]
documents = results.get("documents", [[]])[0]

if not ids:
    print("검색 결과가 없습니다.")
else:
    for i in range(len(ids)):
        print(f"--- 결과 {i+1} ---")
        print(f"ID: {ids[i]}")
        # cosine distance를 쓰는 경우, 값이 낮을수록 유사합니다.
        if i < len(distances):
            print(f"유사도 점수(distances): {distances[i]:.4f}")
        if i < len(metadatas):
            print(f"메타데이터: {metadatas[i]}")
        if i < len(documents):
            snippet = documents[i][:500].replace("\n", " ")
            print(f"내용 (Chunk): {snippet}...\n")

print("\n검색이 완료되었습니다.")

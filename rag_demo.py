import streamlit as st
import dashscope
from dashscope import Generation, TextEmbedding
import chromadb
from chromadb.utils import embedding_functions
import tempfile
import os
import re

# ========== 配置你的 API Key ==========
import os
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

st.set_page_config(page_title="代练知识库助手", page_icon="📚")
st.title("📚 代练知识库助手 - RAG 问答（无需下载模型）")

# 自定义 embedding 函数，调用 dashscope 接口
class DashScopeEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, api_key):
        self.api_key = api_key
        dashscope.api_key = api_key

    def __call__(self, texts):
        # 调用 dashscope 的 embedding 接口
        resp = TextEmbedding.call(
            model=TextEmbedding.Models.text_embedding_v2,
            input=texts
        )
        if resp.status_code == 200:
            # 返回向量列表，每个向量是 list of float
            return [item['embedding'] for item in resp.output['embeddings']]
        else:
            st.error(f"Embedding 调用失败：{resp.message}")
            return [[0.0] * 1536] * len(texts)  # 降级返回零向量

# 初始化 chromadb 客户端（本地持久化）
client = chromadb.PersistentClient(path="./chroma_db")

# 创建或获取 collection
collection_name = "knowledge_base"
try:
    client.delete_collection(collection_name)
except:
    pass
collection = client.create_collection(
    name=collection_name,
    embedding_function=DashScopeEmbeddingFunction(dashscope.api_key)
)

# 1. 上传文档
uploaded_file = st.file_uploader("上传 .txt 文档（价格表、FAQ）", type=["txt"])

def chunk_text(text, chunk_size=500, overlap=50):
    """简单切分文本，按段落或固定长度"""
    # 先按段落分割
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += para + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    # 如果有 overlap，简单处理：不做复杂滑动窗口，因为短文档足够
    return chunks

if uploaded_file:
    # 读取文本
    content = uploaded_file.read().decode("utf-8")
    chunks = chunk_text(content)
    st.write(f"✅ 文档已切分为 {len(chunks)} 个段落")
    
    # 存入 chromadb
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        ids=ids
    )
    st.session_state["collection"] = collection
    st.success("知识库已建立！")

# 2. 问答
if "collection" in st.session_state:
    collection = st.session_state["collection"]
    question = st.text_input("请输入你的问题：")
    if st.button("提问") and question:
        # 检索相似段落
        results = collection.query(query_texts=[question], n_results=3)
        contexts = results['documents'][0]  # list of strings
        
        context_text = "\n\n".join(contexts)
        
        prompt = f"""你是一个游戏代练客服。请根据以下资料回答用户的问题。
如果资料里没有相关信息，就回答“资料中没有提到”，不要自己编造。

资料：
{context_text}

用户问题：{question}

回答："""
        
        response = Generation.call(
            model='qwen-turbo',
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=300,
            temperature=0.3
        )
        answer = response.output.text.strip()
        st.success("回答：")
        st.write(answer)
        
        with st.expander("查看参考的资料段落"):
            for i, ctx in enumerate(contexts):
                st.write(f"**段落 {i+1}:**")
                st.write(ctx)
else:
    st.info("请先上传一个 .txt 格式的知识文档。")
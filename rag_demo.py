import streamlit as st
import dashscope
from dashscope import Generation, TextEmbedding
import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 从环境变量获取 API Key
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
if not dashscope.api_key:
    st.error("请在 Secrets 中设置 DASHSCOPE_API_KEY")
    st.stop()

st.set_page_config(page_title="代练知识库助手", page_icon="📚")
st.title("📚 代练知识库助手 - 轻量 RAG 问答")

# 辅助函数：获取文本向量
def get_embedding(text):
    resp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v2,
        input=[text]
    )
    if resp.status_code == 200:
        return resp.output['embeddings'][0]['embedding']
    else:
        st.error(f"向量生成失败：{resp.message}")
        return None

# 辅助函数：切分文档
def chunk_text(text, chunk_size=500, overlap=50):
    # 按空行分段，再按长度切分
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) <= chunk_size:
            current += p + "\n"
        else:
            if current:
                chunks.append(current.strip())
            current = p + "\n"
    if current:
        chunks.append(current.strip())
    return chunks

# 初始化 session 状态
if "chunks" not in st.session_state:
    st.session_state.chunks = []      # 文本块列表
    st.session_state.embeddings = []  # 对应向量列表

# 1. 上传文档
uploaded_file = st.file_uploader("上传 .txt 文档（价格表、FAQ）", type=["txt"])
if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    chunks = chunk_text(content)
    st.write(f"✅ 文档已切分为 {len(chunks)} 个段落")
    
    # 计算所有块的向量
    embeddings = []
    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        if emb:
            embeddings.append(emb)
        else:
            st.warning(f"第 {i+1} 段向量生成失败，已跳过")
        progress_bar.progress((i+1) / len(chunks))
    
    if embeddings:
        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        st.success("知识库已建立！")
    else:
        st.error("没有成功生成任何向量，请检查 API Key 或网络")

# 2. 问答
if st.session_state.chunks:
    question = st.text_input("请输入你的问题：")
    if st.button("提问") and question:
        q_emb = get_embedding(question)
        if q_emb is None:
            st.error("问题向量化失败")
        else:
            # 转为 numpy 数组计算余弦相似度
            q_vec = np.array(q_emb).reshape(1, -1)
            doc_vecs = np.array(st.session_state.embeddings)
            sims = cosine_similarity(q_vec, doc_vecs)[0]
            # 取相似度最高的 3 个段落
            top_idx = np.argsort(sims)[-3:][::-1]
            contexts = [st.session_state.chunks[i] for i in top_idx]
            context_str = "\n\n".join(contexts)
            
            prompt = f"""你是一个游戏代练客服。请根据以下资料回答用户的问题。
如果资料里没有相关信息，就回答“资料中没有提到”，不要自己编造。

资料：
{context_str}

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
# RAG 混合检索与生成（全链路）笔记

> 主题：把「PDF → 切块 → 向量化 → 混合检索 → 大模型生成 → 后处理」整条链路跑通 一句话定位：**检索负责找材料，大模型只负责读材料写答案，两者职责绝不混淆**

## 一、整条链路的位置与分工

```
PDF 解析 → 清洗 → 分块 Chunking → Embedding 向量化 → 存 Pinecone
                                      ↓
                        用户提问 → 问题向量化 → 向量检索（语义）
                                      ↓
                        用户提问 → jieba 分词 → BM25（关键词）
                                      ↓
                              两路结果 RRF 融合
                                      ↓
                        Top-N chunk 拼成 context + 用户问题 → 送入 LLM
                                      ↓
                        后处理：安全过滤 → 追加引用 → 输出
```

| 环节           | 由谁完成                 | 干什么                       | 不干什么                   |
| -------------- | ------------------------ | ---------------------------- | -------------------------- |
| Embedding 模型 | 通义 `text-embedding-v3` | 文字 → 1024 维向量           | 不会回答问题               |
| 向量库         | Pinecone                 | 余弦相似度近邻检索           | 不理解语义逻辑             |
| 关键词检索     | BM25（本地内存）         | 字面词频匹配                 | 不懂同义词                 |
| 融合           | RRF（纯数学）            | 合并两路排名                 | 不涉及任何模型             |
| 大模型         | DeepSeek / 通义          | 读 context + 问题 → 生成答案 | 不做检索、看不到向量和分数 |

> 核心认知：**LLM 收到的只是一段纯文本，它不知道这段文字是向量召回还是 BM25 召回，也不知道排名分数。**

## 二、组装 Prompt 与大模型调用

```
context_parts = [f"【文档{idx+1}】\n{doc}" for idx, doc in enumerate(top3)]
context = "\n\n".join(context_parts)
answer = llm_generate_answer(question=query, context=context)
```

发给 LLM 的完整内容 = **规则指令 + 参考资料(context) + 用户问题**，三段缺一不可：

| 成分         | 作用                                         |
| ------------ | -------------------------------------------- |
| 规则指令     | 禁止编造、无答案时明确说查不到、要求条理清晰 |
| 【参考资料】 | 检索到的 Top-N chunk，编号便于溯源           |
| 【用户问题】 | 原始问句                                     |

三个工程细节：

- `**base_url="**`[`**https://api.deepseek.com**`](https://api.deepseek.com)`**"**`：DeepSeek 兼容 OpenAI 协议，可直接用 openai SDK

- `**temperature=0.3**`：知识库问答要稳、不要发散，低温度优先

- **异常重试**：指数退避 `2*(i+1)` 秒，`402 余额不足` 这类硬错误重试也不会好，应单独识别

> 关键区分：`【文档1】` 指 **RRF 排名第 1 的 chunk**，不是 PDF 第 1 页，也不是第 1 个文件。编号顺序 = 融合排序。

## 三、后处理：答案出模型之后才执行

顺序不能颠倒：**先安检，再加引用**。

```
answer = safety_check(answer)              # ① 敏感词拦截
answer = add_citations(answer, top3)       # ② 追加参考原文
```

| 函数            | 做什么                             | 局限                                              |
| --------------- | ---------------------------------- | ------------------------------------------------- |
| `safety_check`  | 关键词命中 → 返回统一拒绝话术      | 只能挡字面词，同义/委婉说法无效；生产要用审核模型 |
| `add_citations` | 由检索结果拼接来源，不依赖模型自觉 | 来源是固定拼在末尾，格式生硬                      |

**两种溯源方案对比**：

| 方案          | 做法                       | 优点                       | 缺点               |
| ------------- | -------------------------- | -------------------------- | ------------------ |
| Prompt 内约束 | 让 LLM 自己写"来源：文档1" | 与正文融合自然             | 模型可能漏写、标错 |
| 后处理拼接    | 代码按 top3 自动追加       | 来源由检索决定，绝不会标错 | 样式固定           |

> 两个同时开 = 双重标注，属于冗余。工程上二选一，推荐后处理版。

## 四、提示词外部化（prompt.txt）

```
def load_prompt(file_path="prompt.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


prompt_template = load_prompt()
final_prompt = prompt_template.format(context=context, question=question)
```

- txt 里保留 `{context}` / `{question}` 占位符，交给 `.format()` 填充

- **必须写** `**encoding="utf-8"**`，否则 Windows 下中文乱码

- 改 prompt 不用改代码，也不用重启进程之外的东西，重跑脚本即生效

- 进阶：换 YAML 可同时管理多版 prompt + 描述字段，便于工程化

## 五、踩坑记录（真实调试过程）

### 坑 1：`.env` 里 key 多套了一层 `sk-`

```
DEEPSEEK_API_KEY=sk-sk-20990ac32b5d...
```

症状：接口报鉴权失败。修法：删掉重复前缀，只留一个 `sk-`。

### 坑 2：.env.txt` 后缀（Windows 记事本经典坑）

python-dotenv 只认文件名严格等于 `.env`，不认 `.env.txt`。 检查：资源管理器 → 查看 → 勾选【文件扩展名】。

## 六、完整代码

```
import os
import time
import jieba
import pdfplumber
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
import dashscope
from dashscope import TextEmbedding
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi
# from sentence_transformers import CrossEncoder  # 先注释rerank，避免内存报错

# 加载环境变量
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

# ========== 【读取外部prompt文件】 ==========
def load_prompt(file_path="prompt.txt"):
    """加载外部提示词模板"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
# ==========================================

# ========== 【答案后处理函数】 ==========
def add_citations(answer, source_chunks):
    """给答案追加引用来源，source_chunks就是召回的top3文本列表"""
    answer += "\n\n==== 参考原文片段 ===="
    for idx, chunk in enumerate(source_chunks):
        short_preview = chunk[:80].replace("\n", "")
        answer += f"\n[{idx+1}] {short_preview}..."
    return answer

def safety_check(answer):
    """敏感词安全过滤"""
    harmful_keywords = ["自杀", "暴力", "违法", "毒品"]
    for kw in harmful_keywords:
        if kw in answer:
            return "抱歉，这个问题涉及敏感内容，建议咨询专业人士。"
    return answer
# ==============================================


# ========== 封装通义百炼 Embedding ==========
class DashScopeEmbedding(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = TextEmbedding.call(
            model="text-embedding-v3",
            input=texts
        )
        if resp.status_code != 200:
            raise Exception(f"embedding失败：{resp.message}")
        embeddings = [item["embedding"] for item in resp.output["embeddings"]]
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

# ========== 【DeepSeek LLM问答函数】 ==========
from openai import OpenAI
def llm_generate_answer(question: str, context: str, retry_times=3):
    client = OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com"
    )
    prompt_template = load_prompt()
    final_prompt = prompt_template.format(context=context, question=question)
    for i in range(retry_times):
        try:
            resp = client.chat.completions.create(
                model="deepseek-flash",
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.3
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"LLM请求异常：{e}，等待重试...")
            wait_sec = 2 * (i+1)
            time.sleep(wait_sec)
    return "多次重试失败，请检查DeepSeek API Key和网络"

# ========== 读取PDF文本 ==========
def load_pdf_text(pdf_file_path):
    full_text = ""
    with pdfplumber.open(pdf_file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n\n"
        return full_text

# ========== 分块 ==========
def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "！", "？"]
    )
    chunks = splitter.split_text(text)
    print(f"一共切出 {len(chunks)} 个chunk")
    return chunks

# ========== RRF 融合算法（Reciprocal Rank Fusion）==========
def rrf_fuse(results_list, k=60):
    score_dict = {}
    for text, rank in results_list:
        score = 1.0 / (k + rank)
        if text in score_dict:
            score_dict[text] += score
        else:
            score_dict[text] = score
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    return [txt for txt, s in sorted_items]


if __name__ == "__main__":
    embedding = DashScopeEmbedding()
    index_name = "aspirin-demo"

    pc = Pinecone(api_key=pinecone_api_key)
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"索引 {index_name} 不存在，开始创建...")
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    print("等待索引就绪...")
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(2)
    index = pc.Index(index_name)
    print("✅ 索引已经就绪！")

    # ===================== 在这里改你的PDF路径！=====================
    pdf_path = r"d:\code\agent\test.pdf"
    # =================================================================

    # 入库（首次运行取消注释，跑完立刻注释！）
    # raw_text = load_pdf_text(pdf_path)
    # chunks = split_text(raw_text)
    # # 向量化上传pinecone
    # vecs = embedding.embed_documents(chunks)
    # ids = [f"id-{i}" for i in range(len(chunks))]
    # metadata_list = [{"text": c} for c in chunks]
    # to_upsert = list(zip(ids, vecs, metadata_list))
    # index.upsert(vectors=to_upsert)
    # print("✅ 入库完成")

    # ---------------- BM25 初始化：本地内存加载全部chunk文本 ----------------
    raw_text = load_pdf_text(pdf_path)
    chunks = split_text(raw_text)
    stop_words = {"有", "哪些", "？", "。", "，", "的", "和", "是"}

    # jieba分词 + 停用词过滤，构建bm25
    tokenized_chunks = []
    for c in chunks:
        words = jieba.lcut(c)
        words = [w for w in words if w not in stop_words and w.strip()]
        tokenized_chunks.append(words)
    bm25 = BM25Okapi(tokenized_chunks)

    # 【注释Rerank，防止内存报错】
    # reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    print("\n----------【混合检索 + LLM生成 完整RAG】----------")
    query = "忘记打卡之后怎么办"

    # 1. 向量检索（Pinecone） top10
    q_vec = embedding.embed_query(query)
    vec_res = index.query(vector=q_vec, top_k=10, include_metadata=True)
    vec_texts = [m["metadata"]["text"] for m in vec_res["matches"]]
    vec_rank_data = [(txt, i+1) for i, txt in enumerate(vec_texts)]

    # 2. BM25关键词检索 top10 （jieba分词）
    raw_tokens = jieba.lcut(query)
    query_tokens = [w for w in raw_tokens if w not in stop_words and w.strip()]
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top10_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
    bm25_texts = [chunks[idx] for idx in bm25_top10_idx]
    bm25_rank_data = [(txt, i+1) for i, txt in enumerate(bm25_texts)]

    # 3. RRF融合两路结果
    all_rank_data = vec_rank_data + bm25_rank_data
    fused_candidates = rrf_fuse(all_rank_data)
    print(f"\nRRF融合后候选数量：{len(fused_candidates)}")

    # 【因为关掉Rerank，直接取融合后的前3条】
    top3 = fused_candidates[:3]

    # ==========【组装上下文 + 调用DeepSeek LLM生成答案】==========
    context_parts = []
    for idx, doc_text in enumerate(top3):
        context_parts.append(f"【文档{idx+1}】\n{doc_text}")
    context = "\n\n".join(context_parts)

    answer = llm_generate_answer(question=query, context=context)

    # ========== ✅ 后处理顺序：先安全过滤，再加引用 ==========
    answer = safety_check(answer)
    answer = add_citations(answer, top3)

    print("\n==================== 🤖 LLM最终回答 ====================")
    print(answer)

```

### prompt.txt：

```
你是知识库问答助手，请严格遵守下面规则：
1. 只能基于【参考资料】里面的内容回答问题；
2. 如果参考资料没有相关信息，直接说：当前知识库没有查到相关内容，不要编造；
3. 回答条理清晰，并且标注用到的文档来源。

【参考资料】
{context}

【用户问题】
{question}
```

症状：接口报鉴权失败。修法：删掉重复前缀，只留一个 `sk-`。

## 七、当前版本 vs 生产级差距

**已具备**

- 向量 + 关键词双路召回，RRF 融合，量纲问题已规避

- 检索结果可溯源（chunk 编号 + 原文预览）

- 答案有安全兜底

- prompt 与代码解耦

**生产还需补**

- **Rerank 重排**（CrossEncoder）：融合后精排，把"团建/下午茶"这类语义相近的干扰段压下去

- chunk 元数据（文档名、页码、章节），才能真正定位到"哪份文件的第几页"

- BM25 索引持久化，避免每次请求重建

- 查询改写 / 多轮指代消解

- 检索评估集（Recall@K、命中率）、LLM 答案事实性校验

- 流式输出、并发控制、限流退避、日志埋点

## 八、下一步衔接：Rerank

| 环节         | 模型             | 输入                         | 输出                 |
| ------------ | ---------------- | ---------------------------- | -------------------- |
| 召回（粗排） | Embedding + BM25 | 全库                         | Top-10 候选          |
| 重排（精排） | CrossEncoder     | (query, 单个 chunk) 成对打分 | 相关性分数，重新排序 |

- Bi-Encoder（Embedding）：query 和 doc **分别**编码，可离线建库 → 快但粗

- CrossEncoder（Rerank）：query 和 doc **拼在一起**过模型 → 准但慢，只能对少量候选做

- 位置铁律：**Rerank 一定在融合之后、送入 LLM 之前**

> 段落边界、chunk_overlap 为什么重要：Rerank 和 LLM 都只会看到"你切出来的那一块"，切碎了语义，后面再准的模型也救不回来。

## 附：速查清单

- 分工：检索找材料，LLM 写答案，别让 LLM 干检索的活

- 原文进 metadata，向量库只存向量

- 两路检索词表规则必须一致（都 jieba + 同停用词表）

- RRF 只用排名不用分数，`1/(k+rank)`，k=60

- 发给 LLM 的三段：规则 + context + 问题，缺一不可

- 低温度 0.3，异常重试要退避，硬错误（402/401）不重试

- 后处理顺序：先安检 → 再引用；两套溯源别同时开

- prompt 外置 txt，务必 utf-8

- 排查环境变量：先 print，再看报错

- 下一步优化：加 Rerank，且只在融合后加
- # RAG 查询优化（Multi Query 多查询）笔记

> 主题：把一个用户问题改写成多条同义问句分别检索，扩大召回的查询侧优化策略
>
> 一句话定位：**Multi Query 改的是「怎么问」，不是「存了什么」——同一个意图，多种说法，多路检索，最后合并排序**

## 一、在整条 RAG 流程中的位置

```
用户提问 →【Multi Query 查询改写】→ 多条 query 分别检索 → 结果收集 → 去重 + 融合排序 → 组装 context → LLM 生成答案
```

它处在**用户提问之后、正式检索之前**，属于「查询侧优化」：

- 不动文档解析、不动清洗、不动分块、不动向量库内容

- 只改造**送进检索器的那一句 query**

- 因此可以和已有的【向量检索 + BM25 混合检索】、【RRF 融合】、【Rerank 重排】直接叠加

**重要认知**：优化 RAG 有两条完全不同的路线，别混。

| 路线       | 改什么             | 典型手段                                | 解决什么问题                   |
| ---------- | ------------------ | --------------------------------------- | ------------------------------ |
| 查询侧优化 | 输入检索器的 query | Multi Query、HyDE、查询改写、子问题分解 | 用户问得短、问得糊，**召不全** |
| 检索侧优化 | 召回与排序机制     | 混合检索、RRF、Rerank、阈值过滤         | 召回来的东西**太杂**           |

## 二、核心思想：一个问题，多种问法

用户一句「阿司匹林怎么吃？」，背后可能藏着剂量、时间、禁忌、适用人群多个维度；只用原始这一句去检索，很容易因为**措辞差异**漏掉相关文档。

Multi Query 的做法：让 LLM 生成 N 条**语义意图一致、表达方式不同**的 query，每条独立检索，最后合并结果。

```
原始问题：阿司匹林怎么吃？
    ↓ LLM 改写
Q1 阿司匹林服用方法
Q2 阿司匹林用法用量
Q3 阿司匹林服用要求
Q4 阿司匹林正确服用方式
    ↓ 每条独立检索
4 组结果 → 去重 → RRF 融合 → Top-N
```

三个收益：

| 收益         | 说明                                         |
| ------------ | -------------------------------------------- |
| 🎯 提高召回率 | 不同措辞能匹配到原始问句匹配不到的相关 chunk |
| 🔍 覆盖多角度 | 简短模糊的问题被自动摊开成多个检索视角       |
| 💪 鲁棒性强   | 某一条 query 检索效果差，其余 query 可以兜底 |

## 三、这份代码的执行流程（按顺序）

**前置固定步骤（与原单 query 版本完全一致）**

1. 读 `.env`，初始化 `DashScopeEmbedding`，连接 Pinecone

1. 检查索引是否存在，不存在则创建；轮询等待索引 `ready`

1. `load_pdf_text` 读 PDF → `split_text` 切 chunk（chunk_size=400，overlap=80）

1. jieba 分词 + 停用词过滤，用全部 chunk 在**本地内存**建 BM25 索引

**新增的 Multi Query 环节**

1. 拿到原始问题 `original_query`

1. **调用一次 DeepSeek** → `generate_multi_queries` 产出 4 条同义改写 query

**循环：每条改写 query 各跑一遍混合检索**

1. `single_query_hybrid_rank_data` 内部：

- 调通义百炼 Embedding → 拿到 query 向量

- Pinecone 向量检索 top10 → 得到 `vec_rank_data`

- jieba 分词 → BM25 本地打分 → 取 top10 → 得到 `bm25_rank_data`

- 两路 rank 合并返回

1. 4 轮结果全部 `extend` 进 `total_rank_data`

**融合与生成**

1. `rrf_fuse(total_rank_data)` 全局融合（纯本地计算）

1. 取全局 top3 → 拼成 `【文档1/2/3】` 形式的 context

1. **调用第二次 DeepSeek** → `llm_generate_answer` 生成答案

1. `safety_check` 安全过滤 → `add_citations` 追加引用 → 打印

## 四、四个关键函数拆解

1. ### `generate_multi_queries` —— 唯一新增的"改写器"

```
prompt = f"""
你是RAG查询改写助手。
任务：针对用户问题，生成{num}条【语义意图一致、表达方式不同】的检索查询。
规则：
1. 只是同义改写，不要拆成不同方向的子问题！
2. 适合向量检索 + BM25 关键词检索
3. 严格只输出 json 数组，不要任何解释
"""
```

| 设计点                          | 为什么这么做                                         |
| ------------------------------- | ---------------------------------------------------- |
| `temperature=0.2`               | 要的是稳定多样的改写，不要发散乱编                   |
| prompt 内放 ✅/❌ 对照例子        | few-shot 抑制模型"跑偏成子问题"                      |
| 截取 `[` 到 `]` 再 `json.loads` | 模型常在数组外多说一句"好的，以下是……"，直接解析必崩 |
| 输出条数 `num=4`                | 3~5 是工业常用区间，再多只增成本                     |

1. ### `single_query_hybrid_rank_data` —— 一次 query 的完整召回

```
vec_rank_data  = [(txt, i+1) for i, txt in enumerate(vec_texts)]
bm25_rank_data = [(txt, i+1) for i, txt in enumerate(bm25_texts)]
return vec_rank_data + bm25_rank_data
```

- 返回的是 **(文本, 名次)** 元组列表，**不是分数**——这是为 RRF 服务的

- 两路各取 top10，一条 query 就产生最多 20 条候选项

- 4 条 query 全部跑完，候选池最多 80 条（去重前）

1. ### `rrf_fuse` —— 全局统一排序

```
score = 1.0 / (k + rank)      # k = 60
score_dict[text] += score     # 同一文本多路命中，分数累加
```

- 只在**名次**上做加权，天然绕开"余弦相似度 0~1、BM25 分数无上界"的量纲冲突

- Multi Query 场景下它额外承担了**去重**职责：4 条 query 召回同一 chunk，文本相同即合并计分

1. ### 主流程的收尾

```
top3 = fused_candidates[:3]
answer = llm_generate_answer(question=original_query, context=context)
```

⚠️ 注意这里用的是 `**original_query**`，不是某条改写后的 query：

改写只服务于"把材料找全"，召回更全面，答案要回答的仍是用户**原本那句问题**。

## 完整代码

```
import os
import time
import jieba
import json
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

# ========== Multi Query 生成同义改写query ==========
def generate_multi_queries(original_query: str, num=4):
    client = OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com"
    )
    prompt = f"""
你是RAG查询改写助手。
任务：针对用户问题，生成{num}条【语义意图一致、表达方式不同】的检索查询，用于文档检索。
规则：
1. 只是同义改写，不要拆成不同方向的子问题！
例子：
输入：阿司匹林怎么吃
✅正确：["阿司匹林服用方法","阿司匹林用法用量","阿司匹林服用要求","阿司匹林正确服用方式"]
❌错误（子问题分解，不要这样）：["阿司匹林剂量多少","阿司匹林什么时候吃","阿司匹林副作用"]
2. 适合向量检索+BM25关键词检索
3. 严格只输出json数组，不要任何解释、不要markdown、不要多余文字

用户问题：{original_query}
"""
    resp = client.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    content = resp.choices[0].message.content
    # 清洗json，截取[]内容
    start = content.find("[")
    end = content.rfind("]") + 1
    json_str = content[start:end]
    multi_q_list = json.loads(json_str)
    return multi_q_list

# ========== 单条query执行【向量+BM25混合检索】，返回rank数据 ==========
def single_query_hybrid_rank_data(query, index, embedding, bm25, chunks, stop_words, topk=10):
    # --------向量检索----------
    q_vec = embedding.embed_query(query)
    vec_res = index.query(vector=q_vec, top_k=topk, include_metadata=True)
    vec_texts = [m["metadata"]["text"] for m in vec_res["matches"]]
    vec_rank_data = [(txt, i+1) for i, txt in enumerate(vec_texts)]

    # --------BM25检索----------
    raw_tokens = jieba.lcut(query)
    query_tokens = [w for w in raw_tokens if w not in stop_words and w.strip()]
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:topk]
    bm25_texts = [chunks[idx] for idx in bm25_top_idx]
    bm25_rank_data = [(txt, i+1) for i, txt in enumerate(bm25_texts)]

    one_query_all_rank = vec_rank_data + bm25_rank_data
    return one_query_all_rank

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

    print("\n----------【Multi Query 多查询 + 混合检索 实操测试】----------")
    original_query = "忘记打卡之后怎么办"

    # 1. LLM生成多条同义改写query
    multi_q = generate_multi_queries(original_query, num=4)
    print("✅ 生成的Multi Query列表：")
    for i, q in enumerate(multi_q):
        print(f"  Q{i+1}: {q}")

    # 收集所有query返回的rank数据
    total_rank_data = []
    for q in multi_q:
        print(f"\n--- 正在检索：【{q}】 ---")
        one_rank = single_query_hybrid_rank_data(q, index, embedding, bm25, chunks, stop_words, topk=10)
        total_rank_data.extend(one_rank)

    # 2. 全部结果统一RRF融合排序
    fused_candidates = rrf_fuse(total_rank_data)
    print(f"\nRRF融合后候选数量：{len(fused_candidates)}")

    # 取全局top3送入LLM
    top3 = fused_candidates[:3]

    # ==========【组装上下文 + 调用DeepSeek LLM生成答案】==========
    context_parts = []
    for idx, doc_text in enumerate(top3):
        context_parts.append(f"【文档{idx+1}】\n{doc_text}")
    context = "\n\n".join(context_parts)

    answer = llm_generate_answer(question=original_query, context=context)

    # ========== ✅ 后处理顺序：先安全过滤，再加引用 ==========
    answer = safety_check(answer)
    answer = add_citations(answer, top3)

    print("\n==================== 🤖 LLM最终回答 ====================")
    print(answer)

```

## 五、开销账：到底多调用了什么

| 调用项                         | 原单 query 版本 | Multi Query 版本          | 变化                   |
| ------------------------------ | --------------- | ------------------------- | ---------------------- |
| DeepSeek 生成答案              | 1 次            | 1 次                      | 不变                   |
| **DeepSeek 生成改写 query**    | 0 次            | **1 次**                  | **新增 1 次**          |
| 通义 Embedding（query 向量化） | 1 次            | 4 次（每条 query 各一次） | ×4                     |
| Pinecone 向量检索              | 1 次            | 4 次                      | ×4                     |
| BM25 打分                      | 1 次            | 4 次                      | ×4（本地计算，不花钱） |
| RRF 融合                       | 1 次            | 1 次                      | 不变                   |

> 所以「只多了一次大模型调用」这句话只说对一半：
>
> 大模型确实只多 1 次，但**Embedding 和向量检索都被放大了 num 倍**，时延和费用主要花在这里。

**另外两个隐藏成本**：候选池从 20 条涨到 80 条，噪声变多；context 里混入无关片段的风险上升。

## 六、三个容易混的策略，必须分清

| 策略                        | 改什么                                   | 例子                                   | 发生在哪                    |
| --------------------------- | ---------------------------------------- | -------------------------------------- | --------------------------- |
| **Multi Query**             | 同一意图，换说法                         | 阿司匹林怎么吃 → 服用方法 / 用法用量   | 检索**前**，扩大召回池      |
| **子问题分解 Sub-Question** | 拆成不同方向的新问题，通常各自回答再汇总 | 阿司匹林怎么吃 → 剂量 / 时间 / 副作用  | 检索前，且**多轮 LLM 生成** |
| **Rerank**                  | 不改 query，对已召回候选精确打分         | 用 CrossEncoder 给 (query, chunk) 打分 | 检索**后**，压缩噪声        |

一句话记：**Multi Query 治"漏"，子问题治"复杂"，Rerank 治"杂"。**

## 七、踩坑记录

### 坑 1：以为多 query 就能提高精度

候选池变大 ≠ 答案变好。4 条 query 会带回大量语义相近但无关的 chunk，直接取前 3 反而更容易混进干扰段。

**修法**：融合后必须做重排（RRF 至少要有，理想再加 Rerank），并控制 `per_query_topk`。

### 坑 2：LLM 返回不是纯 JSON

模型爱输出「好的，以下是改写结果：\n[...]」，直接 `json.loads` 抛异常。

**修法**：`content[content.find("["): content.rfind("]")+1]` 截取；或 prompt 里硬性要求"只输出数组"。

### 坑 3：改写跑偏成子问题分解

不加约束时，模型会把「阿司匹林怎么吃」拆成「剂量多少 / 什么时候吃 / 副作用」，这就变成了另一种策略，召回方向完全错位。

**修法**：prompt 里放 ✅/❌ 对照示例，把边界说死。

### 坑 4：不去重

多条 query 很可能召回**完全相同的 chunk**，不去重则同一段文本重复占上下文窗口、浪费 token。

**修法**：合并时按文本去重（`set` 或 `dict.fromkeys`）；RRF 天然也会合并同文本分数。

### 坑 5：拿改写后的 query 去问 LLM

`llm_generate_answer(question=某条改写query, ...)` 会让回答偏移成"回答那个改写句"，而不是用户真正的问题。

**修法**：生成答案永远用 `original_query`。

## 八、当前版本 vs 生产级的差距

**已具备**

- 查询改写 → 多路混合检索 → 全局 RRF 融合的完整闭环

- 同义改写与子问题的边界用 prompt 约束住了

- 改写结果做了 JSON 清洗，能稳定解析

- 最终回答仍锚定原始问题，来源可溯源

**生产还需补**

- `generate_multi_queries` 缺重试与降级（LLM 抖动时整个链路直接断）

- OpenAI client 每次调用都新建，应做全局单例

- 改写结果去重（避免两条一模一样的 query 白跑一次 Embedding）

- 每条 query 的召回贡献可加权（重要 query 权重更高），而不是平均累加

- 时延预算与开关：高并发线上默认关闭，或加缓存

- 检索质量评估：Recall@K 对比单 query / Multi Query 的召回差异（需要标注测试集）

## 附：速查清单

- Multi Query 改的是 query，不改文档、不改索引

- 条数 3~5，多了只增成本与噪声

- 必须同义改写，不能拆成子问题（prompt 里给 ✅/❌ 例子）

- 每条 query 独立跑混合检索，收集 → 去重 → RRF 融合

- RRF 只用名次不用分数，天然免去量纲问题，也顺手做去重

- 多出来的开销：1 次 LLM + Embedding/检索 ×N，主要花在后者

- 生成答案永远用 `original_query`，不是改写后的 query

- LLM 返回的 JSON 要截取 `[...]` 再解析
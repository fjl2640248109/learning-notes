- # RAG 查询优化（Step-Back 回退策略）笔记

> 主题：Step-Back Prompting 回退提问——把过细的问题向上抽象成更宽泛的问题再检索
>
> 一句话定位：**原始问题太窄、直接检索容易漏；先用 LLM 拔高成一个高层回退问题去检索，拿到更广的上下文，再结合原始问题推理回答**

## 一、在整条 RAG 流程中的位置

```
用户细粒度原始问题
      ↓
【Step-Back】
1. LLM 抽象 → 生成更通用的高层回退问题
2. 用回退问题检索 → 拿到更广范围的文档
3. 文档 + 回退问题 + 原始问题 → LLM 推理 → 回答原始问题
```

与前面几种查询策略的区别：

| 策略                     | 核心行为                               | Query 粒度变化 |
| ------------------------ | -------------------------------------- | -------------- |
| Multi Query / RAG-Fusion | 同义改写，换说法多路召回               | 粒度不变       |
| Decomposition            | 拆成多个平级独立子问题，各自作答再汇总 | 拆成更小粒度   |
| **Step-Back**            | 向上抽象出一个更宽泛的问题去检索       | **粒度放大**   |

> 类比：直接问"2023年10月发布的那个新 React 框架叫什么"容易检索不到；先退一步问"最近有哪些新的 React 框架"，拿到一批资料后再定位目标。

## 二、原版 Step-Back 工作原理（当前代码使用）

| 步骤 | 名称             | 说明                                                         |
| ---- | ---------------- | ------------------------------------------------------------ |
| 1    | 抽象 Abstraction | LLM 接收原始细粒度问题，生成一条更宽泛的回退问题             |
| 2    | 检索 Retrieval   | **只用回退问题检索**，获取更广的上下文；原版不拿原始问题检索 |
| 3    | 推理 Reasoning   | 把文档、回退问题、**原始问题**一起交给 LLM，回答原本的具体问题 |

示例：

- 原始问题：技术中心员工每月最多允许补卡多少次？

- 回退问题：公司忘记打卡补卡的全部规章制度有哪些？

### 两条铁律

1. **检索用回退问题，推理必须带原始问题**——不带原始问题，模型只会回答宽泛主题，答非所问。

1. **回退检索不保证一定含答案**：抽象跑偏、范围过大，top3 可能被无关文档挤占；知识库本身没有对应信息时，再完美的回退问题也拿不到答案。

## 三、改良工程变种（笔记记录，当前代码未启用）

原版只靠回退问题检索，**没有兜底**；工业上常见改良：**回退问题 + 原始问题两路同时检索，合并后 RRF 融合**。

- 回退问题：捞更广、上层的上下文

- 原始问题：兜底，捞精准匹配细节的文档

- 两路 rank 合并 → 统一 RRF 打分 → 取 Top-N

```
rank_back   = single_query_hybrid_rank_data(step_back_q, ...)
rank_origin = single_query_hybrid_rank_data(original_q, ...)
fused       = rrf_fuse(rank_back + rank_origin)
```

> 代价：多一次完整混合检索，增加耗时；收益：回退问题生成出错时有原始问题这条备份路线兜底。

## 四、适用 / 不适用场景

✅ 适合

- 时序、限定条件类细节问题（某岗位、某时间、某限定条件下的规则）

- 需要先理解高层概念、再定位细节的复杂问题

- 原始问题过细，直接检索匹配不到文档

❌ 不适合

- 简单事实问答（多一次 LLM 调用，纯浪费 token）

- 需要实时外部数据的场景

- 纯计算类任务

## 五、代码模块说明（原版）

| 函数                       | 职责                                                         |
| -------------------------- | ------------------------------------------------------------ |
| `generate_step_back_query` | 原始问题 → LLM 生成单条回退问题，`temperature=0` 求稳定，失败重试 2 次 |
| `step_back_run`            | 完整链路：生成回退问题 → **用回退问题**混合检索 → RRF 融合取 top3 → 组装 prompt 调用 LLM |
| 底座复用                   | `DashScopeEmbedding`、Pinecone、BM25、`rrf_fuse`、PDF 读取与分块 |

关键实现点：

- 检索入参是 `step_back_q`，不是 `original_q`

- 最终 prompt 同时包含 `context`、`step_back_q`、`original_q` 三者

- 回退问题只生成 **1 条**，不像 Multi Query 生成多条

## 六、踩坑记录

### 坑 1：回退问题抽象失控

太宽泛 → 召回大量噪声；太保守 → 和原始问题几乎一样，策略失效。修法：prompt 里给 ✅/❌ 示例约束抽象幅度。

### 坑 2：检索错用原始问题

依然拿原始问题检索，Step-Back 完全失效。修法：确认检索入参是生成后的 `step_back_q`。

### 坑 3：推理阶段丢掉原始问题

模型只回答宽泛主题，不回答真实提问。修法：最终 prompt 必须传入 `original_q`。

### 坑 4：简单问题无脑开启

事实类问题没必要走回退，白白多一次 LLM 调用。修法：进阶可加前置判断，只对适合的问题启用。

### 坑 5：把回退检索当成"一定命中"

回退只是**扩大检索视野、提高命中概率**，不担保内容一定覆盖原始问题所需信息；所以工程上才会加原始问题做兜底。

## 七、当前版本 vs 生产级差距

**已具备**

- 完整原版链路：原始问题 → 回退问题 → 回退检索 → 结合原问题推理

- LLM 调用异常重试

**生产还需补**

- 前置判断：自动识别问题是否适合 Step-Back

- 改良双路检索（回退 + 原始问题兜底）

- 回退生成失败时降级为普通 RAG（直接用原始问题检索）

- 日志：打印原始问题、回退问题、召回文档，便于排查

> 定位：课程 Demo 合格；上线需补降级、策略判断与兜底检索。

## 附：速查清单

- Step-Back = 细粒度问题向上抽象出高层回退问题，用它检索，再回答原问题

- 原版只用回退问题检索；推理阶段必须带原始问题

- 回退检索不 100% 保证命中；改良版用"回退 + 原始问题"双路兜底

- 适合细节、限定、时序类问题；简单事实问答不要用

- 代价：多一次 LLM 调用生成回退问题                                                     

## 附：完整可运行代码

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
# from sentence_transformers import CrossEncoder  # 注释rerank，避免内存报错
from openai import OpenAI

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

# ===================== Step‑Back 回退策略模块 =====================
def generate_step_back_query(original_q: str, retry_times=2):
    """
    Step‑Back：把细粒度原始问题，向上抽象生成更宽泛的回退问题
    """
    client = OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com"
    )
    prompt = f"""
你是问答助手，任务：将用户过于具体、细节化的问题，向上抽象生成一条更通用、范围更广的回退检索问题。
不要多个问题，只输出1条回退问题，不要多余解释。

示例1
原问题：特斯拉Model 3在2023年Q4的销量是多少？
回退问题：特斯拉Model 3历年销量趋势有哪些？

示例2
原问题：张三在2020‑2022年担任什么职位？
回退问题：张三完整的职业生涯经历？

用户原始问题：{original_q}
回退问题：
"""
    for i in range(retry_times):
        try:
            resp = client.chat.completions.create(
                model="deepseek-flash",
                messages=[{"role":"user","content":prompt}],
                temperature=0
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"生成回退问题失败，重试 {i+1}：{e}")
            time.sleep(1.2)
    raise Exception("生成Step‑Back回退问题多次失败")


def step_back_run(original_q, index, embedding, bm25, chunks, stop_words, topk=10):
    """
    完整Step‑Back执行链路
    1.生成回退问题
    2.【使用回退问题】复用已有的混合检索+RRF
    3.组装上下文，同时传入原始问题+回退问题，调用LLM输出答案
    """
    # ①生成回退问题
    step_back_q = generate_step_back_query(original_q)
    print(f"✅原始问题：{original_q}")
    print(f"✅回退抽象问题：{step_back_q}")

    # ②❗重点：用【回退问题】去做混合检索，不是原始问题
    rank_data = single_query_hybrid_rank_data(step_back_q, index, embedding, bm25, chunks, stop_words, topk=topk)
    fused_candidates = rrf_fuse(rank_data)
    top3 = fused_candidates[:3]

    # 组装上下文
    context_parts = []
    for idx, doc_text in enumerate(top3):
        context_parts.append(f"【文档{idx+1}】\n{doc_text}")
    context = "\n\n".join(context_parts)

    # ③推理：上下文 + 回退问题 + 用户原始问题 三者全部传入
    client = OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com"
    )
    final_prompt = f"""
你是公司员工知识库问答助手，严格依据提供的文档回答，禁止编造内容。
参考文档：
{context}

回退检索问题：{step_back_q}
用户真正的原始问题：{original_q}

请基于上面文档，准确回答【用户真正的原始问题】，条理清晰输出答案。
"""
    resp = client.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role":"user","content":final_prompt}],
        temperature=0.3
    )
    return resp.choices[0].message.content
# =================================================================


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
    print("✅ Pinecone索引就绪！")

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

    original_query = "技术中心员工每月最多允许补卡多少次？"

    print("\n----------【Step‑Back 回退策略模式】----------")
    final_ans = step_back_run(original_query, index, embedding, bm25, chunks, stop_words)
    print("\n==================== 🎯 Step‑Back最终回答 ====================")
    print(final_ans)

```
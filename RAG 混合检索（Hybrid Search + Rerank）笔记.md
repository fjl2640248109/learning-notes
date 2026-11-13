# RAG 混合检索（Hybrid Search + Rerank）笔记

> 主题：把「向量语义检索」和「BM25 关键词检索」两路召回结果融合，再用 Rerank 精细重排，提升 TopK 召回质量 一句话定位：**混合检索 = 两路粗召回取长补短 + RRF 融合 + Rerank 精排**

## 一、这份代码在整个 RAG 流程中的位置

```
文档上传 → 数据清洗 → 文本分块(Chunking) → 向量化(Embedding) → 存入向量库 → 【检索】 → 交给大模型生成答案
```

检索是 RAG 的**质量瓶颈**：检索召回错了，再强的大模型也只能胡说。

**重要认知**：检索不是「向量检索一路走到黑」。

- 向量检索强在**语义**：问「胃不舒服」也能召回「胃肠道反应」

- 向量检索弱在**字面**：药品编号、专有名词、数字、缩写容易漏

- BM25 正好相反 → 所以两路一起上，这就是混合检索的出发点

## 二、两路召回的本质区别

| 维度     | 向量检索（Dense）                        | BM25 关键词检索（Sparse）                   |
| -------- | ---------------------------------------- | ------------------------------------------- |
| 原理     | Embedding 转成 1024 维向量，算余弦相似度 | 词频统计（TF-IDF 的进阶版），算词项匹配权重 |
| 存储位置 | Pinecone 云端（向量 + metadata 原文）    | **本地内存**，每次启动重建                  |
| 擅长     | 同义、转述、语义相近                     | 专有名词、编号、数字、固定术语              |
| 弱项     | 精确字面匹配、长尾词                     | 完全不懂语义，同义词匹配不到                |
| 需要分词 | ❌ 不用（模型内部自带 tokenizer）         | ✅ 必须手动分词（中文用 jieba）              |
| 调用开销 | 网络请求（百炼 + Pinecone）              | 纯 CPU 本地计算                             |
| 返回     | id + score + metadata                    | 文档下标 + 分数                             |

> 关键结论：**Embedding 不需要分词，BM25 必须分词**。这是两路检索最容易被混淆的点。

## 三、代码结构总览（5 个模块）

| 模块         | 函数 / 类            | 干什么                                              |
| ------------ | -------------------- | --------------------------------------------------- |
| ① 向量化封装 | `DashScopeEmbedding` | 调百炼 text-embedding-v3，把文本转向量              |
| ② 文档加载   | `load_pdf_text`      | pdfplumber 一页页抽文字，拼成一整段                 |
| ③ 分块       | `split_text`         | 递归分块，chunk_size=400 / overlap=80               |
| ④ 融合算法   | `rrf_fuse`           | 两路排名合并打分                                    |
| ⑤ 主流程     | `__main__`           | 建索引 → BM25 → 双路召回 → RRF → Rerank → 输出 Top3 |

## 四、RRF 融合算法（核心）

### 4.1 为什么不用「直接加权分数」

```
final_score = alpha * 向量分数 + (1-alpha) * BM25分数
```

问题：**两路分数不在同一个量纲**。向量是余弦相似度（0~1），BM25 是词频权重（可能是 0~20 的任何数）。强行加权，等于拿「米」和「斤」相加。

### 4.2 RRF 的做法：只看排名，不看分数

```
score(doc) = Σ 1 / (k + rank_i)
```

- `rank_i`：这条文档在第 i 路检索里排第几（从 1 开始）

- `k`：平滑常数，默认 60，防止排名第一的文档分数过于碾压

- 同一条文档在多路里都靠前 → **分数累加** → 排上来

### 4.3 手算例子

| 文档 | 向量排名 | BM25 排名 | RRF 分数             |
| ---- | -------- | --------- | -------------------- |
| A    | 1        | 未出现    | 1/61 ≈ 0.0164        |
| B    | 3        | 2         | 1/63 + 1/62 ≈ 0.0320 |
| C    | 未出现   | 1         | 1/61 ≈ 0.0164        |

结论：B 在**两路都靠前**，直接超过各自单路的第一名 → 这就是 RRF 的价值：**共识优先**。

### 4.4 k=60 的作用

k 越大，排名差异被压得越平（第 1 名和第 10 名差距变小）；k 越小，头部越强势。

> 60 是原论文给的默认值，工程上基本不用调。

## 五、Rerank 重排

### 5.1 Bi-Encoder vs Cross-Encoder（必背对比）

|              | Bi-Encoder（向量检索用的）                | Cross-Encoder（Rerank 用的）         |
| ------------ | ----------------------------------------- | ------------------------------------ |
| 输入方式     | query 和 doc **分别**编码，各自出一个向量 | query 和 doc **拼在一起**送进模型    |
| 能否提前算好 | ✅ 可以离线建索引                          | ❌ 必须现场算，无法预存               |
| 速度         | 快（向量比对）                            | 慢（每条候选都要跑一次模型）         |
| 精度         | 一般                                      | 高（能看到 query 与 doc 的词级交互） |

> 结论：Cross-Encoder 精度高但太贵，**只能对粗召回后的小候选集跑**，绝不能全库跑。

### 5.2 标准两段式链路

```
全库（几十万条）
   ↓ 向量 + BM25 双路粗召回，各取 top10
候选集（≤20 条）
   ↓ RRF 融合排序
   ↓ Cross-Encoder 逐对精排
最终 top3 → 交给 LLM
```

## 六、关键参数解释

| 参数            | 取值 | 含义             | 调大 / 调小的影响                            |
| --------------- | ---- | ---------------- | -------------------------------------------- |
| `chunk_size`    | 400  | 每块最大字符数   | 太大→噪声多、检索不准；太小→语义被切碎       |
| `chunk_overlap` | 80   | 相邻块重叠字符数 | 防一句话被切断，通常取 chunk_size 的 10%~20% |
| `top_k`（每路） | 10   | 单路召回条数     | 越大召回越全，但 Rerank 开销线性上升         |
| RRF `k`         | 60   | 排名平滑常数     | 一般不调                                     |
| 最终输出        | 3    | 送 LLM 的片段数  | 多了占上下文、引入噪声；少了可能信息不全     |

> 两路各取 10 而不是直接取 3 的原因：**先宽召回缩小范围，再让 Rerank 做筛选**。粗召回要多，精排要准。

## 七、完整代码

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
from sentence_transformers import CrossEncoder

# 加载环境变量
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

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

    # 加载rerank模型
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    print("\n----------【混合检索 + Rerank 实操测试（jieba中文分词版）】----------")
    query = "可以报销哪些方面的东西"

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

    # 4. Rerank重排，精细打分
    pairs = [(query, doc) for doc in fused_candidates]
    rerank_scores = reranker.predict(pairs)
    reranked = sorted(zip(rerank_scores, fused_candidates), key=lambda x:x[0], reverse=True)

    # 输出最终Top3
    print("\n==== 混合检索+Rerank 最终Top3结果 ====")
    for i, (score, doc) in enumerate(reranked[:3]):
        print(f"结果{i+1} 重排分数:{score:.4f}")
        print(doc)
        print("-"*60)

```

## 八、依赖安装

```
pip install python-dotenv pdfplumber langchain-text-splitters dashscope pinecone jieba rank-bm25 sentence-transformers
```

> ⚠️ 多 Python 环境时，务必用 `D:\software\Python311\python.exe -m pip install xxx`，否则会装到别的解释器里，报 `ModuleNotFoundError`。

## 九、踩坑记录（真实调试过程）

### 坑 1：反复入库导致重复召回

```
每次都跑 index.upsert()  → 相同 chunk 的 id 重复入库
```

**原因**：入库代码没注释，每次启动都重传一遍。 **修法**：入库代码只在文档更新时打开，跑完立刻注释回。更新文档还要先清空 Pinecone 索引，否则新旧内容混在一起。

### 坑 2：BM25 中文完全失效

```
tokenized_chunks = [c.split() for c in chunks]
query_tokens = query.split()
```

**原因**：`split()` 按空格切，中文句子没有空格，整句变成一个「词」，BM25 匹配不到任何东西。 **修法**：`jieba.lcut()` 中文分词 + 停用词过滤。

### 坑 3：CrossEncoder 下载超时（WinError 10060）

```
thrown while requesting HEAD https://huggingface.co/cross-encoder/...
```

**原因**：国内直连 HuggingFace 不通。 **修法**：代码顶部加 `os.environ['HF_ENDPOINT'] = '`[`https://hf-mirror.com`](https://hf-mirror.com)`'`；或提前下载到本地用绝对路径加载；或临时注释掉 Rerank，只跑双路 + RRF。

### 坑 4：Rerank 内存爆掉（os error 1455 / not enough memory）

```
RuntimeError: DefaultCPUAllocator: not enough memory
OSError: 页面文件太小
```

**原因**：10 条长 chunk 一次性批量送进 BERT，CPU 内存 + 虚拟内存扛不住。 **修法**：候选截到 5 条以内 + 逐条 `predict`；或缩小 chunk_size；或直接砍掉 Rerank 只用 RRF。

### 坑 5：中文提问，阿司匹林内容排第一

```
query="报销流程是啥" → 结果1 却是阿司匹林副作用
```

**原因**（三层，层层叠加）：

1. `ms-marco-MiniLM-L-6-v2` 是**英文数据集训练**的 CrossEncoder，对中文「报销」vs「阿司匹林」的区分能力极弱；

1. 原始文档把阿司匹林、奶茶、差旅、考勤混在一起切块，一个 chunk 里塞多种主题；

1. 候选池只有 10 条，坏样本容易挤进前排。

**修法**：优先做**文档按主题拆分**（治本），换中文 Rerank 模型（`bge-reranker-base`），再加关键词硬过滤兜底。

### 坑 6：模型会不会重复下载

**结论**：不会。HuggingFace 缓存是**全局用户级**的（`C:\Users\<用户名>\.cache\huggingface\hub`），换项目、换文件夹、新开终端都读同一份缓存，只有删缓存或换机器才重新下载。

## 十、当前版本 vs 生产级版本的差距

**已具备（符合主流范式）**

- 双路召回（Dense + Sparse），覆盖语义与字面

- RRF 融合，规避量纲不一致问题

- 两段式：宽召回 + 精排

- 中文分词 + 停用词过滤

- 完整链路可运行，能输出带分数的 Top3

**生产环境还需补充**

- 文档预处理阶段按主题隔离，禁止异构内容同块（**收益最大**）

- 换中文 Reranker（bge-reranker-base / bge-reranker-v2-m3）

- 分数阈值过滤，低分直接丢弃，不送 LLM

- 向量检索的 metadata 过滤（按文档类型 / 时间 / 权限预筛）

- Query 改写、多查询扩展（Multi-Query）

- 评估指标：Recall@K / Precision@K / MRR / NDCG

- 缓存层（相同 query 命中缓存）、异步批量、超时重试

> 定位：**课程作业 / 面试 Demo 完全合格**；企业上线必须先解决数据切块质量，再谈模型升级。

## 十一、评估指标（怎么判断检索好不好）

| 指标        | 含义                          | 公式                              | 目标     |
| ----------- | ----------------------------- | --------------------------------- | -------- |
| Recall@K    | 前 K 个结果覆盖了多少相关文档 | 检索到的相关文档数 / 总相关文档数 | 越高越好 |
| Precision@K | 前 K 个里有多少是相关的       | 相关文档数 / K                    | 越高越好 |
| MRR         | 第一个相关文档排名的倒数      | 1 / 第一个相关文档的排名          | 越高越好 |
| NDCG        | 考虑排序位置质量的综合指标    | 折价累计增益                      | 越高越好 |

> Recall 管「有没有捞到」，Precision 管「捞得准不准」，MRR/NDCG 管「排得好不好」。

## 十二、下一步衔接：接入 LLM 生成答案

```
检索 Top3 片段 + 用户问题 → 拼 Prompt → 百炼大模型 → 答案
```

关键原则：

- Prompt 里明确要求「只依据提供的资料回答，资料没有就说不知道」→ 抑制幻觉

- 片段要带来源标注，方便前端展示引用

- 上下文长度要控制，别把 10 条全塞进去

## 附：速查清单

- 向量管语义，BM25 管字面，两路一起才是混合检索

- Embedding 不分词，BM25 必须 jieba 分词

- RRF 只看排名不看分数，公式 `1/(k+rank)`，k=60

- Rerank 用 Cross-Encoder，贵，只能跑小候选集

- 双路各取 top10 → 融合 → 精排 → 输出 top3

- 入库代码跑完必须注释，避免重复 upsert

- 英文 Reranker 别用在中文场景

- 效果不好先查数据切块，再查模型
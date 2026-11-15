- # RAG 查询优化（HyDE 假设性文档嵌入）笔记

> 主题：Hypothetical Document Embedding，让 LLM 先编一段假想答案，用假想答案的向量去检索真实文档
>
> 一句话定位：**问题短、文档长，问句和文档在向量空间里不对称；HyDE 先把问题"补写成一段答案风格的假想文档"，拿这段文档去检索，利用"答案—答案相似度 > 问题—答案相似度"提升召回**

## 一、这份代码在整个 RAG 流程中的位置

```
用户原始问题
    ↓
【HyDE】
1. LLM 凭空生成一段假设性假想文档（允许有幻觉）
2. 用假想文档做向量 + BM25 混合检索（本工程同时用原始问题检索兜底）
3. RRF 融合取 Top-N 真实文档 → 连同原始问题交给 LLM 生成答案
```

> 与前面几种查询增强策略对比

| 策略                     | 核心行为                           | 检索输入                 |
| ------------------------ | ---------------------------------- | ------------------------ |
| Multi Query / RAG-Fusion | 同义改写，多路召回                 | 多条同义问句             |
| Decomposition            | 拆成平级子问题，各自作答再汇总     | 多个子问题               |
| Step-Back                | 向上抽象成更宽泛的回退问题         | 一条高层问句             |
| **HyDE**                 | LLM 生成一段**假想答案文档**再检索 | **陈述句风格的假想文档** |

## 二、传统 RAG 的痛点：查询—文档不对称

- 用户**问题很短**（几个词，疑问句式）；

- 知识库 **chunk 很长**（几百字，陈述句式）；

- 两者句式、长度、语义空间分布差异大，余弦相似度天然偏低，容易召回失败。

示例：

> 问题：`Milvus 是什么？`（短问句向量）
>
> 文档：`Milvus 是一个开源向量数据库，专为 AI 应用设计……`（长陈述向量）

HyDE 的思路：先让 LLM 编一段"答案风格"的文本，它的行文和真实文档更接近，再用它去检索。**答案和答案之间的相似度，远高于问题和答案之间的相似度**！

#### 工作流程图

| 步骤              | 说明                                                         |
| ----------------- | ------------------------------------------------------------ |
| 1. 生成假设文档   | LLM 凭自身参数记忆生成一段假想回答，**允许错误、允许幻觉**，不读知识库 |
| 2. 假想文档向量化 | 对假想文档做 Embedding                                       |
| 3. 向量检索       | 用假想文档向量检索真实知识库；本工程额外用原始问题检索做兜底 |
| 4. 最终推理       | 只把**检索到的真实文档** + 原始问题交给 LLM 作答；**假想文档到此丢弃，不进上下文** |

### 三条铁律

1. **假想文档只做检索 query，绝不能放进上下文当参考资料**，否则幻觉会被当成真实业务内容输出。

1. 假想文档来自 LLM 内部知识，**全程没有读取 PDF / 向量库 / BM25**。

1. 假想文档跑偏时，靠原始问题这一路检索兜底；两路结果合并 RRF 融合。

## 四、代码模块说明

| 函数                        | 职责                                                         |
| --------------------------- | ------------------------------------------------------------ |
| `generate_hypothetical_doc` | 原始问题 → LLM 生成假想文档；`temperature=0.8` 增加多样性；空返回检测 + 重试 + 调用间隔休眠 |
| `hyde_run`                  | 完整链路：生成假想文档 → 假想文档 + 原始问题两路混合检索 → RRF 取 top3 → 真实文档组装上下文 → LLM 作答；生成失败降级为原始问题检索 |
| 底座复用                    | `DashScopeEmbedding`、Pinecone、BM25、`rrf_fuse`、PDF 读取与分块 |

关键实现点：

- 检索两路：`rank_hyde`（假想文档）+ `rank_origin`（原始问题），合并后 RRF；

- 最终 prompt 只含真实 `context` 和 `original_q`，`**hypo_doc**` **不出现**；

- 空返回必须手动判断：DeepSeek 限流/风控时会返回 HTTP 200 但 `content=""`，不抛异常，`except` 抓不到。

## 五、踩坑记录（真实调试过程）

### 坑 1：LLM 静默返回空字符串

日志现象：`[DEBUG LLM原始返回]: ''`，重试两次仍为空，最后触发降级。

**原因**：prompt 带 `假设性文档：输出：` 这类分隔标记，flash 模型易静默拒答；调用间隔太短触发限流，接口返回 200 但内容为空，异常捕获不到。

**修法**：

- prompt 改成直白指令，去掉分隔标记；

- 每次调用前 `time.sleep(1.5)` 拉长间隔；

- `max_tokens` 提到 500；

- 手动 `strip()` 后判断长度是否为 0，空则重试。

### 坑 2：把假想文档拼进上下文

**后果**：编造内容被当成知识库素材，输出错误答案。

**修法**：假想文档只用于检索，上下文只放检索出来的真实 chunk。

### 坑 3：只走假想文档一路检索，没有兜底

假想文档一旦跑偏，召回全是噪声，整条链路失效。

**修法**：原始问题并行检索一路，两路合并 RRF。

### 坑 4：简单问题无脑开 HyDE

短句事实问答直接检索就够好，多一次 LLM 调用纯耗 token、加延迟。

**修法**：复杂/概念类问题才启用，简单问题走普通 RAG。

### 坑 5：生成数量过多

多份假想文档向量求平均会稀释有效语义，一般 3~5 份即可；本工程单份生成。

## 六、适用 / 不适用场景

✅ 适合

- 概念类、制度描述类问题：问句短，知识库是大段陈述句

- 原始问句直接检索语义对不上、召回不足的问题

❌ 不适合

- 简单事实问答（收益低、多耗 token）

- LLM 完全不懂的垂直领域（假想文档全是噪声）

- 需要实时外部数据的场景

## 七、当前版本 vs 生产级版本差距

**已具备**

- 完整 HyDE 链路：生成假想文档 → 两路检索兜底 → 真实文档作答

- 空返回检测、重试、调用间隔休眠

- 生成失败降级为原始问题检索，链路不崩

**生产还需补充**

- 前置判断：自动识别问题是否适合 HyDE

- 假想文档缓存：相似问题复用，减少重复 LLM 调用

- 多份假想文档向量平均（可选，提升鲁棒性）

- 完整日志：原始问题、假想文档、召回文档留痕排查

> 定位：课程 Demo 合格；上线需补策略判断、缓存与降级监控。

## 附：速查清单

- HyDE = LLM 凭空编一段假想答案文档，用它的向量检索真实知识库

- 假想文档允许幻觉，**只做检索输入，绝不进回答上下文**

- 解决"问句短、文档长"的查询—文档语义不对称

- 工程上配原始问题检索两路兜底；生成失败降级回普通 RAG

- 注意静默空返回：HTTP 200 不代表有内容，必须手动校验字符串

- 代价：多一次 LLM 调用，token 与时延上升；LLM 不懂领域时会失效

> 🎤 面试口述：HyDE 针对传统 RAG 问句和文档语义不对称的问题，让大模型先生成一段假想答案文档，用假想文档的向量去检索真实知识库，利用答案—答案相似度更高来提升召回；假想文档只用于检索、不进上下文；缺点是依赖大模型领域知识、增加时延和 token，工程上通常叠加原始问题检索兜底，并在生成失败时降级。

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
from openai import OpenAI

# 加载环境变量
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

# ========== 读取外部prompt文件 ==========
def load_prompt(file_path="prompt.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# ========== 通义百炼 Embedding ==========
class DashScopeEmbedding(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = TextEmbedding.call(model="text-embedding-v3", input=texts)
        if resp.status_code != 200:
            raise Exception(f"embedding失败：{resp.message}")
        return [item["embedding"] for item in resp.output["embeddings"]]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

# ========== LLM问答基础函数 ==========
def llm_generate_answer(question: str, context: str, retry_times=3):
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
            time.sleep(2 * (i+1))
    return "多次重试失败，请检查DeepSeek API Key和网络"

# ========== 向量+BM25混合检索，返回rank数据 ==========
def single_query_hybrid_rank_data(query, index, embedding, bm25, chunks, stop_words, topk=10):
    # 向量检索
    q_vec = embedding.embed_query(query)
    vec_res = index.query(vector=q_vec, top_k=topk, include_metadata=True)
    vec_rank_data = [(m["metadata"]["text"], i+1) for i, m in enumerate(vec_res["matches"])]

    # BM25检索
    raw_tokens = jieba.lcut(query)
    query_tokens = [w for w in raw_tokens if w not in stop_words and w.strip()]
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:topk]
    bm25_rank_data = [(chunks[idx], i+1) for i, idx in enumerate(bm25_top_idx)]

    return vec_rank_data + bm25_rank_data

# ========== PDF读取 ==========
def load_pdf_text(pdf_file_path):
    full_text = ""
    with pdfplumber.open(pdf_file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n\n"
    return full_text

# ========== 文本分块 ==========
def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "！", "？"]
    )
    chunks = splitter.split_text(text)
    print(f"一共切出 {len(chunks)} 个chunk")
    return chunks

# ========== RRF融合 ==========
def rrf_fuse(results_list, k=60):
    score_dict = {}
    for text, rank in results_list:
        score = 1.0 / (k + rank)
        score_dict[text] = score_dict.get(text, 0) + score
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    return [txt for txt, _ in sorted_items]

# ===================== HyDE 模块（改良：两路检索兜底 + 空值检测 +降级） =====================
def generate_hypothetical_doc(original_q: str, retry_times=2):
    """生成假设文档，仅用于检索，允许幻觉，不做回答素材"""
    prompt = f"""
请模拟企业内部制度文档，写一段回答，用来回答下面用户问题。
只输出文档正文，不要标题，不要解释，不要开场白。
允许内容不完全符合事实，重点模仿制度文档的行文风格。

用户问题：{original_q}
"""
    for i in range(retry_times):
        time.sleep(1.5)  # 每次调用强制休眠，避免触发接口限流返回空
        try:
            resp = client.chat.completions.create(
                model="deepseek-flash",
                messages=[{"role":"user","content":prompt}],
                temperature=0.8,
                max_tokens=500
            )
            raw_content = resp.choices[0].message.content
            print(f"[DEBUG LLM原始返回]: {repr(raw_content)}")
            hypo_candidate = raw_content.strip()
            if len(hypo_candidate) == 0:
                print(f"⚠️ 第{i+1}次生成，返回内容为空，继续重试")
                continue
            return hypo_candidate
        except Exception as e:
            print(f"生成假设文档失败，重试 {i+1}: {e}")
            time.sleep(1.5)
    raise Exception("HyDE生成假设文档多次失败，均得到空内容")


def hyde_run(original_q, index, embedding, bm25, chunks, stop_words, topk=10):
    # 异常降级：生成失败直接使用原始问题
    try:
        hypo_doc = generate_hypothetical_doc(original_q)
    except Exception as e:
        print(f"⚠️ HyDE生成假设文档异常:{e}，降级，使用原始问题作为检索query")
        hypo_doc = original_q

    print(f"✅原始问题：{original_q}")
    print("🎭 HyDE【完整假设文档】：")
    print(hypo_doc)
    print("-"*60)

    # 两路检索兜底：假设文档 + 原始问题
    rank_hyde = single_query_hybrid_rank_data(hypo_doc, index, embedding, bm25, chunks, stop_words, topk=topk)
    rank_origin = single_query_hybrid_rank_data(original_q, index, embedding, bm25, chunks, stop_words, topk=topk)
    all_rank = rank_hyde + rank_origin

    fused = rrf_fuse(all_rank)
    top3 = fused[:3]

    context_parts = [f"【文档{idx+1}】\n{d}" for idx, d in enumerate(top3)]
    context = "\n\n".join(context_parts)

    # 注意：只传入真实检索文档，假设文档不进上下文！
    final_prompt = f"""
你是公司员工知识库问答助手，严格依据提供的文档回答，禁止编造内容。
参考文档：
{context}
用户原始问题：{original_q}

请条理清晰输出答案。
"""
    resp = client.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role":"user","content":final_prompt}],
        temperature=0.3
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    embedding = DashScopeEmbedding()
    index_name = "aspirin-demo"
    pdf_path = r"d:\code\agent\test.pdf"

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

    # ========== 入库（首次运行取消注释，跑完立刻注释！）==========
    # raw_text = load_pdf_text(pdf_path)
    # chunks = split_text(raw_text)
    # vecs = embedding.embed_documents(chunks)
    # ids = [f"id-{i}" for i in range(len(chunks))]
    # metadata_list = [{"text": c} for c in chunks]
    # to_upsert = list(zip(ids, vecs, metadata_list))
    # index.upsert(to_upsert)
    # print("✅ 入库完成")

    # BM25初始化
    raw_text = load_pdf_text(pdf_path)
    chunks = split_text(raw_text)
    stop_words = {"有", "哪些", "？", "。", "，", "的", "和", "是"}
    tokenized_chunks = []
    for c in chunks:
        words = jieba.lcut(c)
        words = [w for w in words if w not in stop_words and w.strip()]
        tokenized_chunks.append(words)
    bm25 = BM25Okapi(tokenized_chunks)

    original_query = "公司年休假有哪些关键的注意事项？"

    print("\n----------【HyDE假设文档检索模式】----------")
    final_ans = hyde_run(original_query, index, embedding, bm25, chunks, stop_words)
    print("\n🎯 HyDE最终回答：\n", final_ans)

```
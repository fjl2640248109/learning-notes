- # RAG 查询优化（Decomposition 问题分解）笔记

> 主题：把复杂复合问题拆成多个独立子问题，各自检索、各自作答，最后综合成完整回答
>
> 一句话定位：**Decomposition = 分而治之：一个大问题 → N 个子问题 → N 套完整 RAG → 1 次归纳汇总**

## 一、在整条 RAG 流程中的位置

```
复杂原始问题 →【LLM 分解为子问题】→ 每个子问题各跑一遍 RAG → N 个子答案 →【LLM 综合】→ 最终回答
                                                      ↑
                            子问题内部：向量+BM25混合检索 → RRF融合 → Top3 → 生成子答案
```

> Decomposition **真的把问题拆成了不同的信息需求**。

## 二、为什么需要问题分解

用户的复杂提问往往是**复合型**，一句话里塞了好几个信息点：

> "阿司匹林的用法、副作用和禁忌人群分别是什么"

这句话实际包含 3 个独立诉求。不做分解会出三种问题：

| 问题       | 表现                                                         |
| ---------- | ------------------------------------------------------------ |
| 检索不精准 | 一整句拿去做向量检索，语义被"平均"掉，召回一堆沾边但不针对的 chunk |
| 信息遗漏   | 某一路需求没被任何 chunk 覆盖，模型直接少答一块              |
| 回答混乱   | 上下文里塞满三类内容，模型归纳时容易串味、丢要点             |

分解之后：

- 每个子问题短、聚焦 → 检索命中率明显提高

- 每个信息点都被单独照顾 → 覆盖全

- 汇总时有明确的"提纲" → 答案结构清晰

## 三、这份代码的完整执行流程（按顺序）

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

# ===================== Decomposition 问题分解模块 =====================
def decompose_question(original_q: str, max_sub=5, retry_times=2):
    """
    把原始复杂问题拆解为子问题列表
    :param original_q: 用户原始问题
    :param max_sub: 最多拆解多少个子问题
    :param retry_times: JSON解析失败重试次数
    :return: list[str] 子问题列表
    """
    client = OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com"
    )
    decompose_prompt = f"""
你是问题分解助手。
将用户复杂问题拆解为{max_sub}个以内，互相独立、可以单独检索回答的子问题。
不要同义改写，要拆分不同信息维度的真实子问题。
禁止多余说明，严格输出JSON数组。
输出示例：["子问题1","子问题2","子问题3"]

用户原始问题：{original_q}
"""
    for i in range(retry_times):
        try:
            resp = client.chat.completions.create(
                model="deepseek-flash",
                messages=[{"role": "user", "content": decompose_prompt}],
                temperature=0
            )
            content = resp.choices[0].message.content
            start = content.find("[")
            end = content.rfind("]") + 1
            sub_list = json.loads(content[start:end])
            # 简单去重
            sub_list = list(dict.fromkeys(sub_list))
            return sub_list
        except Exception as e:
            print(f"分解子问题失败，重试 {i+1}：{e}")
            time.sleep(1.5)
    raise Exception("多次尝试，问题分解失败")


def get_sub_rag_answer(sub_q, index, embedding, bm25, chunks, stop_words, topk=10):
    """
    单个子问题执行完整RAG链路：混合检索 → RRF融合 → 组装上下文 → 生成该子问题的子答案
    """
    rank_data = single_query_hybrid_rank_data(sub_q, index, embedding, bm25, chunks, stop_words, topk=topk)
    fused_candidates = rrf_fuse(rank_data)
    top3 = fused_candidates[:3]

    # 组装子问题的上下文
    context_parts = []
    for idx, doc_text in enumerate(top3):
        context_parts.append(f"【文档{idx+1}】\n{doc_text}")
    sub_context = "\n\n".join(context_parts)

    sub_ans = llm_generate_answer(question=sub_q, context=sub_context)
    return {"sub_question": sub_q, "sub_answer": sub_ans}


def synthesize_all_sub_answer(original_q, sub_qa_list, retry_times=2):
    """综合全部子问题、子答案，输出最终完整回答"""
    client = OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com"
    )
    synthesis_prompt = f"""
你是专业问答助手。
基于【用户原始问题】以及下面一组子问题和对应的子答案，综合整理一份完整、逻辑连贯的回答。
不要简单拼接，需要归纳整理，贴合用户原始提问。

用户原始问题：{original_q}

子问题与子答案：
{json.dumps(sub_qa_list, ensure_ascii=False, indent=2)}

输出最终回答：
"""
    for i in range(retry_times):
        try:
            resp = client.chat.completions.create(
                model="deepseek-flash",
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.3
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"综合答案异常，重试：{e}")
            time.sleep(2)
    return "子答案综合生成失败"
# ============================================================================


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


    # ========== 模式开关 ==========
    RUN_DECOMPOSITION = True   # True 运行Decomposition；False运行原来Multi‑Query
    original_query = "如何训练一个GPT模型并部署到生产环境？"


    if not RUN_DECOMPOSITION:
        # ----------原有逻辑 Multi‑Query + RAG‑Fusion----------
        print("\n----------【Multi Query 多查询 + 混合检索 实操测试】----------")
        multi_q = generate_multi_queries(original_query, num=4)
        print("✅ 生成的Multi Query列表：")
        for i, q in enumerate(multi_q):
            print(f"  Q{i+1}: {q}")

        total_rank_data = []
        for q in multi_q:
            print(f"\n--- 正在检索：【{q}】 ---")
            one_rank = single_query_hybrid_rank_data(q, index, embedding, bm25, chunks, stop_words, topk=10)
            total_rank_data.extend(one_rank)

        fused_candidates = rrf_fuse(total_rank_data)
        print(f"\nRRF融合后候选数量：{len(fused_candidates)}")
        top3 = fused_candidates[:3]

        context_parts = []
        for idx, doc_text in enumerate(top3):
            context_parts.append(f"【文档{idx+1}】\n{doc_text}")
        context = "\n\n".join(context_parts)

        answer = llm_generate_answer(question=original_query, context=context)

        print("\n==================== 🤖 LLM最终回答 ====================")
        print(answer)

    else:
        # ---------- Decomposition 问题分解模式----------
        print("\n----------【Decomposition 问题分解模式】----------")
        print(f"用户原始问题：{original_query}")

        # 1、问题分解
        sub_questions = decompose_question(original_query, max_sub=4)
        print(f"\n✅拆解得到子问题列表：{sub_questions}")

        sub_qa_results = []
        for sq in sub_questions:
            print(f"\n👉处理子问题：{sq}")
            sub_qa = get_sub_rag_answer(sq, index, embedding, bm25, chunks, stop_words, topk=10)
            print(f"子答案：{sub_qa['sub_answer']}")
            sub_qa_results.append(sub_qa)

        # 2、综合所有子答案
        final_raw = synthesize_all_sub_answer(original_query, sub_qa_results)

        print("\n==================== 🎯 Decomposition最终综合回答 ====================")
        print(final_raw)

```

**Decomposition 分支（**`**RUN_DECOMPOSITION = True**`**）**

1. `decompose_question(original_query, max_sub=4)` → **第 1 次 LLM 调用**，拿到子问题列表

1. 进入 `for sq in sub_questions` 循环，每个子问题调用 `get_sub_rag_answer`：

- `single_query_hybrid_rank_data` → 向量检索 top10 + BM25 top10

- `rrf_fuse` → 两路融合排序

- 取 top3 拼成子上下文

- `llm_generate_answer(question=sq)` → **每个子问题各 1 次 LLM 调用**，产出子答案

1. `synthesize_all_sub_answer(original_query, sub_qa_list)` → **最后 1 次 LLM 调用**，把全部子问答归纳成一份完整答案

> 一次请求的 LLM 调用量 = **1（分解）+ N（子答案）+ 1（汇总）**。
>
> 4 个子问题就是 **6 次**，这是 Decomposition 最需要警惕的成本点。

## 四、三个新函数各管什么

| 函数                        | 职责                      | 关键设计                                                     |
| --------------------------- | ------------------------- | ------------------------------------------------------------ |
| `decompose_question`        | 把大问题拆成子问题        | `temperature=0` 求稳定；截取 `[...]` 再解析；`dict.fromkeys` 去重；失败重试 2 次 |
| `get_sub_rag_answer`        | 子问题跑完整 RAG 拿子答案 | 完全复用已有混合检索 + RRF + `llm_generate_answer`，不重复造轮子 |
| `synthesize_all_sub_answer` | 全部子答案 → 最终回答     | prompt 里必须同时带**原始问题**，否则汇总会跑偏成"复述子答案" |

## 五、三种分解策略：谁是代码管，谁是 LLM 管

这是最容易混淆的一节，必须分清**策略范式**和**具体内容**：

| 策略                  | 子问题依赖关系         | 谁决定拆分内容          | 谁决定执行流程                     | 现有代码能否直接支持 |
| --------------------- | ---------------------- | ----------------------- | ---------------------------------- | -------------------- |
| **并行 Parallel**     | 互相独立，无先后       | LLM 生成                | 代码 for 循环，各自独立跑          | ✅ **就是现在这份**   |
| **顺序 Sequential**   | 后一步依赖前一步的结果 | LLM 拆步骤              | **代码必须把上一步答案喂给下一步** | ❌ 需要改造           |
| **层次 Hierarchical** | 树状多级               | LLM 拆第一层 + 每层再拆 | **代码需要嵌套循环**               | ❌ 需要改造           |

### 三条结论

1. **并行分解**：一句 prompt 约束 + 现有循环就够，最省事，也是员工知识库最常见场景

1. **顺序分解**：光有 prompt 一定错——LLM 就算拆出有序步骤，代码不传递上下文，依赖链直接断

1. **层次分解**：必须两层循环（先拆大类，再对每个大类二次拆分），一次自由输出会退化成扁平列表

> 想"一套代码自动切换三种策略"是不成立的：每种策略都要配套**独立的执行流程**，只能在上一层加一个策略路由做分支调度。

### 顺序分解为什么必须改代码

举例："完整报销流程怎么操作"

- 子问题1：报销需要准备什么材料 → 拿到材料清单

- 子问题2：材料齐了，提交入口和审批人是谁 → **检索时必须带上子答案1的信息**

- 子问题3：审批通过后多久打款 → **依赖前面两步**

如果用现在的并行循环去跑，子问题2、3 会丢掉前序信息，各自孤立检索，答案必然缺环。

## 六、Decomposition 对比 Multi Query（面试必问）

| 维度           | Multi Query              | Decomposition                     |
| -------------- | ------------------------ | --------------------------------- |
| 改的是什么     | 说法（同义改写）         | 结构（拆成新问题）                |
| 意图是否改变   | ❌ 不变                   | ✅ 变成多个不同意图                |
| 是否生成子答案 | ❌ 不生成，只负责捞文档   | ✅ 每个子问题独立生成              |
| LLM 调用次数   | 1（改写）+ 1（回答）     | 1（分解）+ N（子答案）+ 1（汇总） |
| 解决什么       | 措辞差异导致的**漏召回** | 复合问题导致的**漏信息、答不全**  |
| 适用问题       | 简短、单一、问法模糊     | 复合、对比、多维度、多步骤        |

## 七、踩坑记录

### 坑 1：简单问题也上分解

"什么是 RAG？"这种单一问题拆开毫无收益，只多烧 4~5 次 API 调用。

**判断标准**：问题里是否包含 ≥2 个独立信息需求（"……和……分别是什么"、"对比 A 和 B"）才算适合。

### 坑 2：并行代码跑顺序问题

当前代码结构只支持**并行分解**。把 main 里换成"如何训练一个GPT模型并部署到生产环境"这类**强顺序依赖**的问题，拆出的步骤仍会各自孤立检索，答案接不上。

**修法**：要么换成顺序型问题（如员工制度里"……分别是什么"），要么补一套顺序执行逻辑。

## 八、当前版本 vs 生产级差距

**已具备**

- 完整链路：分解 → 子问题独立 RAG → 综合，且全部复用已有混合检索与 RRF

- 子问题数量、温度、重试都有约束

- 汇总阶段回扣原始问题

**生产还需补**

- **子问题串行执行 → 改并发**（当前 4 个子问题依次跑，时延叠加，用线程池可大幅压缩）

- 分解结果质量校验：过滤过短、重复、与原问题无关的子问题

- 子答案为空或失败时的降级策略（不能把空答案塞进汇总）

- 缓存：相同子问题不重复检索

- 顺序 / 层次策略的独立执行流程（当前仅支持并行）

- 评估：对比"直接 RAG" vs "Decomposition" 的答案完整度、覆盖度

- 汇总输出格式约束（如【回答】+【重要提醒】），当前为自由文本

> 定位：**学习 Demo 完全合格，三种策略已能讲清边界；上线前至少要补并发与降级。**

## 附：速查清单

- Decomposition 拆的是**问题结构**，不是问法；子问题意图各异

- 一次请求 = 1 次分解 + N 次子答案 + 1 次汇总，成本随 N 线性增长

- 只有复合型问题才值得拆；单一事实问答不要拆

- 子问题用 `temperature=0`，追求稳定可复现

- 子问题要去重，`max_sub` 要设上限

- 汇总必须带 `original_query`，否则答非所问

- **并行**：prompt 约束即可；**顺序**：代码必须传递上一轮答案；**层次**：代码必须嵌套循环

- 光改 prompt 不能切换三种策略，每种策略都要独立执行流程

- 现有代码是串行循环，生产要改并发

- 区分口诀：Multi Query 治说法，RAG-Fusion 治排序，Decomposition 治问太宽
# RAG 路由 + 混合检索 + Text2SQL 笔记

> 主题：一套可运行的 RAG 问答系统，先由大模型判断问题该走哪条数据源，再分别走向量知识库、结构化 SQL、Web 搜索或知识图谱。 一句话定位：**路由先选路，检索再取数，最后由大模型基于取到的内容作答**。

## 一、在 RAG 流程中的位置

```
用户提问 → 【路由分类】 → 判断数据源
              ├─ vectorstore ：PDF/制度文档 → 向量 + BM25 混合检索 → 大模型作答
              ├─ sql_database：员工统计问题 → Text2SQL 生成 SQL → 查 MySQL → 大模型整理
              ├─ web_search   ：实时/外部信息（占位，待接搜索 API）
              ├─ graph_db     ：实体关系/图谱查询（占位，待接 Neo4j）
              └─ fallback_human：无法判断 → 转人工
```

核心思想：**不让所有问题都进向量库**，按问题类型分流到最合适的数据源，提高准确率、减少幻觉。

## 二、整体模块拆解

| 模块       | 对应函数                                               | 作用                                                         |
| ---------- | ------------------------------------------------------ | ------------------------------------------------------------ |
| 环境与配置 | `load_dotenv` / `MYSQL_CFG`                            | 读 API Key、配置 MySQL 连接                                  |
| Embedding  | `DashScopeEmbedding`                                   | 封装通义百炼向量模型，统一 `embed_documents / embed_query` 接口 |
| PDF 读取   | `load_pdf_text`                                        | 用 pdfplumber 逐页提取全文文本                               |
| 文本分块   | `split_text`                                           | 递归分块 `RecursiveCharacterTextSplitter`，带重叠            |
| 混合检索   | `single_query_hybrid_rank_data`                        | 向量检索 + BM25 关键词检索，两路都返回「文本 + 排名」        |
| RRF 融合   | `rrf_fuse`                                             | 用 RRF 公式把两路排名结果融合重排                            |
| 向量问答   | `normal_retrieve_run`                                  | 取融合后的 Top-K 片段，拼进 prompt 交大模型作答              |
| 路由       | `route_question`                                       | 大模型输出纯 JSON，判断数据源并给理由                        |
| Text2SQL   | `generate_sql` / `safe_exec_sql` / `format_sql_answer` | 生成 SQL → 黑名单校验 → 执行 → 基于结果整理答案              |
| 扩展占位   | `web_search_run` / `graph_db_run`                      | 预留搜索、图谱两条链路                                       |
| 总调度     | `pipeline`                                             | 按路由结果分发到对应分支                                     |

## 三、关键技术点

1. ### 路由设计

- 让大模型只输出一行纯 JSON：`{"datasource": "...", "reasoning": "..."}`。

- 代码里用正则剥掉 ```json 标记、提取花括号内容再 `json.loads`。

- 解析失败时兜底默认走 `vectorstore`，避免系统崩在路由上。

1. ### Text2SQL 安全约束

- **只允许 SELECT**，黑名单拦截 `delete / drop / alter / update / insert / create`。

- **禁止编造**：答案整理 prompt 里强约束「只能用查询返回的数据，不能猜数字」。

- 连接用完在 `finally` 里关闭，避免连接泄漏。

## 四、两条主链路怎么跑

### 链路 A：走向量知识库（问制度、考勤、年假）

```
问题 → 路由判定 vectorstore
     → 向量检索 + BM25 检索 → RRF 融合取 Top3
     → 片段拼进 prompt → DeepSeek 基于文档作答（禁止编造）
```

### 链路 B：走 Text2SQL（问人数、薪资、部门统计）

```
问题 → 路由判定 sql_database
     → 大模型生成 MySQL SELECT 语句
     → 关键词黑名单校验 → pymysql 连 rag_demo 执行
     → 字段 + 原始数据交大模型，整理成自然语言回答
```

## 五、踩坑记录

### 坑 1：MySQL 中文报 latin-1 编码错

报错 `'latin-1' codec can't encode characters`，是连接字符集问题。在配置里加上 `charset="utf8mb4"` 和 `use_unicode=True`。

### 坑 2：路由 JSON 带 markdown 标记

大模型常返回 ````json ... ````，需用正则去掉代码块标记，再提取 `{...}`，否则 `json.loads` 失败。

### 坑 3：PDF 重复读取

加载一次全文存为 `raw_text`，分块和 BM25 共用同一个 `chunks`，不要在两处各读一遍 PDF。

### 坑 4：SQL 结果为空 / 执行失败

`safe_exec_sql` 异常时返回 `None`，下游 `format_sql_answer` 要先判空，否则取 `sql_result['cols']` 直接报错。

## 六、当前版本 vs 生产级

**已具备**

- 多数据源路由 + 容错兜底

- 向量 / BM25 混合检索 + RRF 融合

- Text2SQL 黑名单防护 + 防幻觉约束

- 分块、入库、检索、作答完整串联

**生产还需补**

- `web_search` 接 Tavily / Serper 等真实搜索 API

- `graph_db` 接 Neo4j，做实体抽取 + Cypher 生成

- 检索后加 Rerank 重排，进一步提升 Top-K 精度

- 给 chunk 附 metadata（来源文件、页码）便于溯源

- API Key、数据库密码走环境变量，不硬写在代码里

## 附：完整代码

```
import os
import time
import jieba
import json
import logging
import pdfplumber
import pymysql
import re
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

# ========== MySQL 配置 ==========
MYSQL_CFG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "database": "rag_demo",
    "charset": "utf8mb4",
    "use_unicode": True
}

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

# ========== 普通混合检索链路 ==========
def normal_retrieve_run(original_q, index, embedding, bm25, chunks, stop_words, topk=3):
    print("----------【普通向量+BM25混合检索模式】----------")
    rank_raw = single_query_hybrid_rank_data(original_q, index, embedding, bm25, chunks, stop_words, topk)
    fused_chunks = rrf_fuse(rank_raw)
    selected_chunks = fused_chunks[:topk]

    context_parts = [f"【文档片段{i+1}】\n{d}" for i, d in enumerate(selected_chunks)]
    context = "\n\n".join(context_parts)

    final_prompt = """你是公司员工知识库问答助手，严格依据下面参考文档回答，禁止编造内容。
参考文档：
{context}
用户问题：{original_q}
请条理清晰给出回答：""".format(context=context, original_q=original_q)
    resp = client.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.3
    )
    return resp.choices[0].message.content

# ===================== 【路由 + Text2SQL 模块】 =====================
def route_question(question: str):
    prompt = f"""
你是RAG路由分类器。
数据源说明：
vectorstore：员工制度PDF、考勤、请假、年假这类非结构化文档。
sql_database：员工人数、年龄、薪资、部门这类结构化统计查询。
web_search：实时信息、行业趋势、新闻、外部政策，本地知识库和数据库没有的数据。
graph_db：实体关系、关联分析、知识图谱路径查询类问题。
fallback_human：无法判断，交给人工。

严格要求：只输出一行纯JSON，不要```、不要解释、不要换行。
字段：datasource、reasoning
示例：{{"datasource":"vectorstore","reasoning":"问题是年假制度，属于文档知识库"}}

用户问题: {question}
"""
    resp = client.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role":"user", "content": prompt}],
        temperature=0
    )
    content = resp.choices[0].message.content
    try:
        content = re.sub(r"```(json)?", "", content)
        json_str = re.search(r"\{.*?\}", content, re.DOTALL).group(0)
        obj = json.loads(json_str)
        ds = obj.get("datasource", "vectorstore")
        reason = obj.get("reasoning", "无")
        return ds, reason
    except Exception as e:
        print(f"[路由解析错误] 模型原始返回内容：{content}")
        return "vectorstore", f"解析路由结果失败，err:{str(e)}，默认走向量知识库"


def generate_sql(question):
    prompt = f"""
你是SQL生成专家。MySQL数据库，表employee，字段：id, name, age, department, salary, entry_date
规则：只输出SELECT，禁止DELETE/DROP/ALTER/UPDATE/INSERT/CREATE，mysql语法，不要额外文字
用户问题：{question}
SQL：
"""
    resp = client.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()


def safe_exec_sql(sql):
    forbid_keywords = ["delete","drop","alter","update","insert","create"]
    low_sql = sql.lower()
    for kw in forbid_keywords:
        if kw in low_sql:
            return "【拒绝执行】高危SQL"
    conn = None
    try:
        conn = pymysql.connect(**MYSQL_CFG)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        col_names = [i[0] for i in cur.description]
        return {"cols":col_names, "data":rows}
    except Exception as e:
        logging.error(f"sql err: {sql} {e}")
        print(f"[SQL执行报错] {e}")
        return None
    finally:
        if conn:
            conn.close()


def format_sql_answer(user_q, sql_result):
    if sql_result is None:
        return "SQL查询失败"
    if isinstance(sql_result, str):
        return sql_result

    prompt = f"""
【强制规则：只允许使用下面提供的数据，绝对不能编造数字、不能猜测、不能自行推断！】
用户问题：{user_q}
返回字段：{sql_result['cols']}
原始查询数据：{sql_result['data']}

严格基于上面的数据直接回答，不要额外发挥。
"""
    resp = client.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role":"user","content":prompt}],
        temperature=0.0
    )
    return resp.choices[0].message.content

# ===================== 新增：Web Search、GraphDB 占位函数 =====================
def web_search_run(question):
    print("----------【Web搜索模式】----------")
    return f"[Web搜索] 问题：{question}，搜索能力待接入"

def graph_db_run(question):
    print("----------【Graph DB知识图谱模式】----------")
    return f"[GraphDB] 问题：{question}，图谱查询能力待接入"

# ===================== 总入口：路由调度 =====================
def pipeline(question, index, embedding, bm25, chunks, stop_words):
    datasource, reason = route_question(question)
    print(f"\n📍路由决策：{datasource}，理由：{reason}")
    if datasource == "vectorstore":
        return normal_retrieve_run(question, index, embedding, bm25, chunks, stop_words)
    elif datasource == "sql_database":
        print("----------【Text2SQL结构化查询模式】----------")
        sql = generate_sql(question)
        print(f"📝生成SQL: {sql}")
        res = safe_exec_sql(sql)
        return format_sql_answer(question, res)
    elif datasource == "web_search":
        return web_search_run(question)
    elif datasource == "graph_db":
        return graph_db_run(question)
    else:
        return "已转入人工处理，请联系管理员。"


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

    # 只加载一次PDF，复用给chunks和BM25
    raw_text = load_pdf_text(pdf_path)
    chunks = split_text(raw_text)

    # ========== 入库（首次运行取消注释，跑完立刻注释！）==========
    # vecs = embedding.embed_documents(chunks)
    # ids = [f"id-{i}" for i in range(len(chunks))]
    # metadata_list = [{"text": c} for c in chunks]
    # to_upsert = list(zip(ids, vecs, metadata_list))
    # index.upsert(to_upsert)
    # print("✅ 入库完成")

    # BM25初始化
    stop_words = {"有", "哪些", "？", "。", "，", "的", "和", "是"}
    tokenized_chunks = []
    for c in chunks:
        words = jieba.lcut(c)
        words = [w for w in words if w not in stop_words and w.strip()]
        tokenized_chunks.append(words)
    bm25 = BM25Okapi(tokenized_chunks)

    # 测试问题
    original_query = "统计年龄大于35岁的员工有几个"

    final_ans = pipeline(original_query, index, embedding, bm25, chunks, stop_words)
    print("\n🎯 最终回答：\n", final_ans)
```
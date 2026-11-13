# RAG 向量化与向量库（Embedding + Pinecone）笔记

> 主题：把分好的 chunk 交给通义百炼生成向量，存入 Pinecone，并完成交互式语义检索 一句话定位：**Embedding 负责把文字变成「数字指纹」，Pinecone 负责存指纹 + 按语义相似度检索**

## 一、在 RAG 流程中的位置

```
文档上传 → 数据清洗 → 文本分块(Chunking) → 【向量化 Embedding → 存入向量库】 → 检索召回 → 交给大模型生成答案
```

- **Embedding（百炼** `**text-embedding-v3**`**）**：文字 → 固定长度的浮点数向量（语义编码）

- **Pinecone**：存「向量 + 原文元数据」，做相似度检索，本身不理解文字、不生成答案

## 二、核心概念

1. ### 向量与维度

- 一段文字经过 Embedding 模型，输出一串浮点数，就是**向量**；语义越接近，向量整体越相似。

- **维度**：向量里数字的个数，由模型决定。`text-embedding-v3` 固定输出 **1024 维**。

- 维度不是文字越长越多：一句话和一大段文字，只要用同一个模型，输出维度都一样。

1. ### 一条 Pinecone 记录的结构

| 字段       | 内容                                  | 作用                                         |
| ---------- | ------------------------------------- | -------------------------------------------- |
| `_id`      | 入库时分配的唯一编号                  | 标识这条 chunk，**属于文档片段，不属于提问** |
| `vector`   | 1024 个浮点数                         | 相似度比对只用它                             |
| `metadata` | 原始 chunk 文本（默认 key 为 `text`） | 召回后取出来展示、拼 prompt，不参与向量比对  |

1. ### 相似度排序是「检索时临时算」的

- 库里存的向量本身**没有固定顺序**，排序是每次提问时实时计算的产物。

- 入库一次，向量永久保存在 Pinecone；**提问的向量是临时传过去算相似度的，用完即结束，不会存入库**。

## 三、代码结构拆解

1. ### 环境变量

```
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")   # 百炼 Key
pinecone_api_key = os.getenv("PINECONE_API_KEY")     # Pinecone Key
```

1. ### 封装 Embedding 适配 LangChain 接口

LangChain 要求实现两个方法：`embed_documents`（批量入库向量化）、`embed_query`（提问向量化）。

```
from langchain_core.embeddings import Embeddings
from dashscope import TextEmbedding


class DashScopeEmbedding(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = TextEmbedding.call(model="text-embedding-v3", input=texts)
        if resp.status_code != 200:
            raise Exception(f"embedding失败：{resp.message}")
        return [item["embedding"] for item in resp.output["embeddings"]]


    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
```

- `embed_documents`：输入 chunk 列表，输出和列表一一对应的向量列表。

- `embed_query`：只把当前这一句问题转向量。

1. ### 读取 PDF + 递归分块

```
import pdfplumber

def load_pdf_text(pdf_file_path):
    full_text = ""
    with pdfplumber.open(pdf_file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n\n"
    return full_text

def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "！", "？"]
    )
    return splitter.split_text(text)
```

> `pdfplumber` 只支持**文字型 PDF**；扫描图片版 PDF 需要先 OCR。

1. ### 创建 / 连接 Pinecone 索引

```
pc = Pinecone(api_key=pinecone_api_key)
if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1024,                                  # 必须和 v3 维度一致
        metric="cosine",                                # 余弦相似度
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )


index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embedding)
```

1. ### 入库（只跑一次）

```
raw_text = load_pdf_text(pdf_path)
chunks = split_text(raw_text)
vector_store.add_texts(texts=chunks)
```

`add_texts` 内部流程：chunk 文本 → 调百炼生成向量 → 组装成 `id + vector + metadata(text)` → 上传 Pinecone。

1. ### 交互式检索

```
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # k = 召回 TopK 条

while True:
    query = input("👉 请输入你的问题：")
    if query.strip().lower() == "exit":
        break
    res = retriever.invoke(query)
    for doc in res:
        print(doc.page_content)
```

- `k`：最多返回相似度最高的 K 条；库里不足 K 条时按实际数量返回。

- 每次循环独立发起一次检索：问题转向量 → Pinecone 比对 → 重新排序返回。

  ### 完整代码

  ```
  import os
  import time
  import pdfplumber
  from dotenv import load_dotenv
  from langchain_text_splitters import RecursiveCharacterTextSplitter
  from langchain_core.embeddings import Embeddings
  import dashscope
  from dashscope import TextEmbedding
  from langchain_pinecone import PineconeVectorStore
  from pinecone import Pinecone, ServerlessSpec
  
  # 加载环境变量
  load_dotenv()
  dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
  pinecone_api_key = os.getenv("PINECONE_API_KEY")
  
  # ========== 封装通义百炼 Embedding ==========
  class DashScopeEmbedding(Embeddings):
      def embed_documents(self, texts: list[str]) -> list[list[float]]:
          """批量向量化文档chunk"""
          resp = TextEmbedding.call(
              model="text-embedding-v3",
              input=texts
          )
          print(f"request_id: {resp.request_id}")
          if resp.status_code != 200:
              raise Exception(f"embedding失败：{resp.message}")
          embeddings = [item["embedding"] for item in resp.output["embeddings"]]
          print("✅ 向量维度：", len(embeddings[0]))
          print("✅ 向量样例：", embeddings[0][:5])
          return embeddings
  
      def embed_query(self, text: str) -> list[float]:
          """向量化用户query"""
          return self.embed_documents([text])[0]
  
  # ========== 读取PDF文本 ==========
  def load_pdf_text(pdf_file_path):
      """读取PDF全部文字，返回字符串"""
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
  
  # ========== 主流程 ==========
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
  
      vector_store = PineconeVectorStore(index=index, embedding=embedding)
  
      # ===================== 在这里改你的PDF路径！=====================
      pdf_path = r"d:\code\agent\test.pdf"
      # =================================================================
  
      # ① 入库：读取PDF、分块、上传（入库成功之后注释这三行！）
      raw_text = load_pdf_text(pdf_path)
      chunks = split_text(raw_text)
      vector_store.add_texts(texts=chunks)
      print("✅ PDF全部向量上传到Pinecone完成！")
  
      # ② 循环交互式检索
      retriever = vector_store.as_retriever(search_kwargs={"k": 3})
      print("\n==== 知识库检索已就绪，输入 exit 退出 ====")
      while True:
          query = input("\n👉 请输入你的问题：")
          if query.strip().lower() == "exit":
              print("程序退出")
              break
          if not query.strip():
              print("问题不能为空，请重新输入！")
              continue
  
          res = retriever.invoke(query)
          print("\n==== 召回结果 ====")
          for doc in res:
              print(doc.page_content)
              print("-" * 50)
  
  ```

## 四、踩坑记录

### 坑 1：索引维度和模型维度不匹配

- `text-embedding-v3` 是 **1024 维**，老索引用 v2 的 **1536 维**，上传/检索直接报错。

- 修法：删掉旧索引，按模型维度重建；换模型时必须同步改 `dimension`。

### 坑 2：重复运行导致向量叠加重复

- `add_texts` 默认是**追加写入**，不会清空旧数据；重复跑入库，相同 chunk 会生成多条不同 `_id` 的记录，召回出现重复文本。

- 修法：入库成功后**注释掉入库三行**，之后只跑检索；换 PDF 前先清空索引或删索引重建。

### 坑 3：误以为网页搜索面板能输入中文问题

- Pinecone 控制台的 `Search by ID` 是拿指定 chunk 的向量去搜相似记录，**不能输入自然语言问题**；点击列表条目会自动填充该条 `_id`，搜索基准被固定，结果自然不变。

- 网页面板不会自动同步 Python 代码发出的检索请求，判断结果以**终端打印为准**。

### 坑 4：把提问存进了库里

- 入库只存 chunk；提问向量是临时用于检索的，不会写入索引，也不会生成新的 `_id`。

### 坑 5：用量统计延迟

- 百炼后台的调用记录、Token 消耗有几分钟延迟；排查单次请求用代码里打印的 `request_id` 去日志里搜。

## 五、当前版本 vs 生产级

**已具备**

- 百炼 v3 Embedding 接入 LangChain 接口

- PDF 解析 + 递归分块 + 向量化入库完整串联

- 索引维度、相似度算法配置正确

- 循环交互式检索，可退出

**生产还需补**

- 入库前前置去重（相同 chunk 只保留一条）

- 给 chunk 附 metadata：来源文件、页码，便于溯源

- 召回后加 **Rerank 重排**：先多召回（k=5），重排模型二次打分，过滤无关内容

- 检索结果按相似度分数打印并设置阈值过滤

- 接入大模型：把召回 chunk 拼进 prompt，生成通顺答案

- 扫描件 OCR、表格/公式单独处理

## 六、依赖安装

```
pip install pdfplumber python-dotenv langchain-text-splitters \
    langchain-core langchain-pinecone pinecone dashscope
```

## 附：速查清单

- 链路：chunk → 百炼 v3 向量(1024 维) → Pinecone 存向量+原文 → 问题向量检索 TopK

- 记录三要素：`_id`（chunk 编号）、`vector`（比对用）、`metadata`（原文）

- 排序：检索时实时算，库里不存排序；提问向量不入库

- 入库只跑一次，跑完注释；重复跑会叠加重复数据

- 维度必须对齐：v3 = 1024，v2 = 1536

- `k` 控制召回条数；网页 `Search by ID` 不能当中文提问调试台
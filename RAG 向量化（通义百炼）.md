- # RAG 向量化与 Chroma 向量库笔记

  > 主题：文档分块完成后，调用 Embedding 模型转为向量，存入 Chroma 向量库，执行向量检索召回 chunk 
  >
  > 一句话定位：**分块文本 → Embedding 转向量 → 向量库持久化存储 → query 向量相似度检索，按 chunk 整块召回** 
  >
  > 学习状态：分块已跑通，本部分是下一步要动手实现的内容
  
  ## 一、在 RAG 流程中的位置

  ```undefined
  文档上传 → 数据清洗 → 文本分块 → 【向量化 Embedding】 → 存向量库 → 向量检索召回 → Prompt 组装 → 交给大模型生成
  ```

  向量化：把人类文本转换成多维浮点数向量，用来表达文本语义；检索依靠向量距离判断语义相似度。
  
  ## 二、向量化 + 向量库为什么要做
  
  1. Embedding 模型不能接收超长文本，只能接收小块 chunk。
  2. 向量可以表达语义，可以做语义检索，不局限关键词匹配。
  3. 向量库持久保存向量与原文，不用每次程序启动重新调用 Embedding 接口。
  4. 检索最小单元是 chunk，整块召回，不会自动剔除 chunk 内部的无关句子。

  ## 三、核心组件说明
  
  ### 1. Embedding 模型（通义 text-embedding-v3）
  
  - 输入：文本字符串列表（多个 chunk）
  - 输出：每段文本对应 1024 维向量

  > ⚠️ 注意：入库、查询必须使用**同一个 Embedding 模型**；Chroma 集合会记录绑定的 embedding 函数，不一致会直接抛冲突异常。
  
  ### 2. Chroma PersistentClient 持久化客户端

  ```python
  client = chromadb.PersistentClient(path="./chroma_db")
  ```

  - 存储位置：本地文件夹 `chroma_db`
  - sqlite 文件：保存 chunk 原文、id、元数据
  - HNSW 索引文件：向量近似最近邻索引

  > 删除 `chroma_db` 文件夹 = 清空整个向量库。

  ### 3. Collection 集合
  
  相当于一张数据表，存放一批文档的向量、文本、id。
  
  | 方法                  | 作用                                                     |
  | :-------------------- | :------------------------------------------------------- |
  | `create_collection()` | 创建集合，可指定距离算法                                 |
  | `collection.add()`    | 写入 chunk 文本、id，内部自动调用 embedding 生成向量入库 |
  | `collection.query()`  | 传入问题文本，自动生成 query 向量，做相似度检索          |

  ### 4. 距离度量：cosine 余弦距离
  
  代码配置：`metadata={"hnsw:space": "cosine"}`

  ```undefined
  余弦距离 = 1 - 余弦相似度
  ```
  
  - 范围 `[0, 2]`；**数值越小，语义越相似**
  - 0：完全相同；0.1~0.3 相关性较好；大于 0.7 基本无关

  > HNSW 是近似索引，检索结果是近似结果，不是暴力穷举精确计算。
  
  ## 四、完整执行流程
  
  1. PDF 解析清洗，得到干净长文本。
  2. `RecursiveCharacterTextSplitter` 递归字符分块，输出 chunk 列表。
  3. 创建 Chroma 持久化 client，删除旧集合避免残留数据。
  4. `create_collection` 创建集合，绑定 embedding 函数，设置 cosine 距离。
  5. `collection.add(documents=chunks, ids=ids)`：
     1. 循环每个 chunk 调用 dashscope embedding 接口生成向量
     2. 向量 + 原文持久化保存到 `chroma_db`
  6. 用户 query 问题，调用 `collection.query(query_texts=[query], n_results=N)`：
     1. 问题文本生成 query 向量
     2. HNSW 索引计算距离，按距离从小到大排序，返回 top-N 个完整 chunk
  7. 返回结果：`documents` 召回的文本片段，`distances` 对应余弦距离。

  > **关键点**：召回以 chunk 为最小单元整块返回，不会拆分 chunk 内部句子；如果切分边界混入无关内容，噪音会跟着一起返回。

  ## 五、关键参数
  
  **n_results：召回块数**
  
  - Demo 常用 2；生产一般粗召回 top 5~10，后续交给 rerank 再筛选
  - 不能无限多召回：受 LLM 上下文窗口、token 成本、噪音幻觉风险限制
  
  **chunk_size / chunk_overlap（分块参数，直接影响向量质量）**
  
  - 中文：`chunk_size=300~500` 字符，overlap 取块大小 10%~20%
  - chunk 过大：单块混入多个主题，向量语义混杂；chunk 过小：语义残缺，召回效果差
  
  ## 六、简化核心代码片段
  
  ```python
  # 1. 持久化客户端
  client = chromadb.PersistentClient(path="./chroma_db")
  
  # 2. 创建集合
  collection = client.create_collection(
      name="demo",
      embedding_function=embedding_func,
      metadata={"hnsw:space": "cosine"}
  )
  
  # 3. 写入 chunk
  collection.add(documents=chunks, ids=[f"chunk_{i}" for i in range(len(chunks))])
  
  # 4. 向量检索召回
  res = collection.query(query_texts=["你的问题"], n_results=2)
  # res["documents"] 召回文本；res["distances"] 余弦距离
  
  ```
  
  ## 全部代码
  
  ```python
  import re
  import os
  import chromadb
  import dashscope
  from http import HTTPStatus
  from typing import Dict, Any
  from chromadb import Documents, EmbeddingFunction, Embeddings
  from chromadb.utils.embedding_functions import register_embedding_function
  from langchain_text_splitters import RecursiveCharacterTextSplitter
  import pdfplumber
  
  # ========== 通义百炼Embedding类（适配新版chroma） ==========
  @register_embedding_function
  class DashScopeEmbeddingFunction(EmbeddingFunction):
      def __init__(self):
          # dashscope.base_http_api_url = "https://ws-o17cyqvf wfqarv9g.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
          dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
  
      def __call__(self, input: Documents) -> Embeddings:
          resp = dashscope.TextEmbedding.call(
              model="text-embedding-v3",
              input=input
          )
          if resp.status_code == HTTPStatus.OK:
              embeddings = [item["embedding"] for item in resp.output["embeddings"]]
              return embeddings
          else:
              raise Exception(f"向量接口报错：{resp.code}, {resp.message}")
  
      @staticmethod
      def name() -> str:
          return "dashscope-text-embedding-v3"
  
      def get_config(self) -> Dict[str, Any]:
          return {}
  
      @staticmethod
      def build_from_config(config: Dict[str, Any]) -> "DashScopeEmbeddingFunction":
          return DashScopeEmbeddingFunction()
  
  # ---------------------- 清洗函数 ----------------------
  def remove_headers_footers(text: str) -> str:
      page_en = re.compile(r'page\s+\d+\s+of\s+\d+', re.IGNORECASE)
      page_cn = re.compile(r'第\s*\d+\s*页\s*/\s*共\s*\d+\s*页')
      text = page_en.sub("", text)
      text = page_cn.sub("", text)
      return text
  
  def clean_invisible_chars(text: str) -> str:
      return re.sub(r'[\r\f\u200b\u200c\u200d]', '', text)
  
  # ---------------------- PDF读取 ----------------------
  def extract_text_from_pdf(pdf_path):
      all_text = ""
      with pdfplumber.open(pdf_path) as pdf:
          for page in pdf.pages:
              page_text = page.extract_text()
              if page_text:
                  all_text += page_text + "\n"
      return all_text
  
  # ---------------------- 预处理 + 分块 ----------------------
  def preprocess_document(pdf_path, chunk_size=300, chunk_overlap=50):
      raw_text = extract_text_from_pdf(pdf_path)
      text = remove_headers_footers(raw_text)
      text = clean_invisible_chars(text)
      splitter = RecursiveCharacterTextSplitter(
          chunk_size=chunk_size,
          chunk_overlap=chunk_overlap,
          separators=["\n\n", "\n", "。", "！", "？", "，", " "]
      )
      chunks = splitter.split_text(text)
      return chunks
  
  
  if __name__ == "__main__":
      # 【保持注释！方案2，终端设置环境变量】
      # os.environ["DASHSCOPE_API_KEY"] = "sk-xxx"
  
      pdf_file = r"D:\code\agent\test.pdf"
      persist_path = "./chroma_db"
      collection_name = "aspirin_demo"
  
      # 实例化embedding
      embedding_func = DashScopeEmbeddingFunction()
  
      client = chromadb.PersistentClient(path=persist_path)
      try:
          client.delete_collection(collection_name)
      except:
          pass
  
      collection = client.create_collection(
          name=collection_name,
          embedding_function=embedding_func,
          metadata={"hnsw:space": "cosine"}
      )
  
      chunks = preprocess_document(pdf_file, chunk_size=300, chunk_overlap=50)
      print(f"一共切出 {len(chunks)} 块\n")
  
      ids = [f"chunk_{i}" for i in range(len(chunks))]
      collection.add(
          documents=chunks,
          ids=ids
      )
      print("✅ 全部chunk入库完成！")
  
      # 查询测试
      query = "阿司匹林有什么副作用？"
      res = collection.query(
          query_texts=[query],
          n_results=2
      )
      print(f"\n查询问题：{query}")
      for idx, doc in enumerate(res["documents"][0]):
          print(f"\n==== 结果 {idx+1} ====")
          print(doc)
          print(f"距离：{res['distances'][0][idx]}")
  
  
  ```
  
  
  
  ## 七、踩坑记录
  
  ### 坑 1：Embedding 函数冲突报错
  
  报错：`An embedding function already exists in the collection configuration`
  
  - 原因：集合已经持久保存了 embedding 配置，`get_collection` 时又传入不一样的 embedding 实例
  - 解决：读取已有集合时不传 `embedding_function`，chroma 自动读取持久化配置；或者删除集合重建
  
  ### 坑 2：chunk 内部混入无关内容（阿司匹林 + 奶茶同一 chunk）
  
  - 原因：递归字符分块只看符号，不理解语义；主题切换刚好落在同一个块内
  - 现象：整块向量整体语义偏向有效内容，距离很小被召回，但块内部混杂噪音
  - 缓解方案：
    - 扩充 `separators`，增加文档主题分隔标记优先切分
    - 粗召回多取几条，后续增加 Rerank 重排过滤
    - 进阶：使用语义分块 / 结构化分块
  
  ### 坑 3：不要用记事本直接打开 chroma.sqlite3
  
  sqlite3 是二进制数据库，记事本打开全乱码，修改会直接损坏向量库。
  
  ## 八、当前版本 vs 生产级
  
  **已具备**
  
  - PDF 解析-清洗-递归分块-向量化-向量持久化-向量检索完整链路
  - 使用余弦距离做语义召回，可打印距离观察相关性
  - 可删除集合重置向量库
  
  **生产还需补充**
  
  - 召回后流程：Rerank 重排、距离阈值过滤、上下文压缩、去重
  - metadata 存入来源、页码信息，实现回答溯源
  - token 计数，而不是简单 `len()` 字符
  - 表格、公式单独处理
  - 向量库异常捕获；限制 embedding 批量调用大小
  
  ## 九、下一步学习
  
  向量检索只是粗召回；召回之后还要经过 rerank 重排、prompt 组装，送入 LLM 大模型生成最终回答，完成完整 RAG 链路。
  
  ## 附：速查清单
  
  - Chroma 向量与文本保存在 `chroma_db` 文件夹；删除文件夹即清空库
  - cosine 余弦距离：数值越小语义越相似
  - 检索最小单元是 chunk，整块召回，不会自动删除块内部无关句子
  - 入库与查询必须使用同一个 Embedding 模型
  - 线上一般粗召回多块，再经过后处理筛选，再交给大模型
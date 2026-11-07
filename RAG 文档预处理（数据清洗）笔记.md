- # RAG 文档预处理（数据清洗）笔记

  > 主题：把杂乱文档（PDF / 原始文本）清洗成适合做检索增强生成（RAG）的干净文本 一句话定位：**清洗 = 只删噪声 + 保留语义边界，不做格式化表演**

  ## 一、这份代码在整个 RAG 流程中的位置

  ```
  文档上传 → 【数据清洗】 → 文本分块(Chunking) → 向量化(Embedding) → 存入向量库 → 检索 → 交给大模型生成答案
  ```

  清洗是**最上游**的一步，质量直接决定后续分块和召回效果。

  **重要认知**：清洗不是越干净越好。

  - 删掉的东西：页眉、页脚、页码、不可见字符、多余空格 → 这些是**噪声**

  - 不能删的东西：段落边界、句子结构、专有名词大小写 → 这些是**语义信息**

  ## 二、行业主流清洗流程（7 步，顺序不能乱）

  | 步骤                | 做什么                                 | 常用手段                                                     |
  | ------------------- | -------------------------------------- | ------------------------------------------------------------ |
  | 1. 文档解析         | 把各种格式读出纯文本                   | PDF→pdfplumber / PyMuPDF；图片型PDF→OCR；Word→python-docx；网页→Trafilatura / BeautifulSoup |
  | 2. 固定噪声移除     | 删页码、页眉、页脚、水印、重复版权声明 | 正则匹配模板化文本                                           |
  | 3. 不可见脏字符清理 | 删 \r、\f、零宽字符                    | 正则字符类                                                   |
  | 4. 行内空格压缩     | 多个空格 / Tab → 单个空格              | re.sub(r'[ \t]+', ' ', text)                                 |
  | 5. 段落归一化       | 连续空行 → 单个换行，**保留段落**      | re.sub(r'(\n\s*)+', '\n', text)                              |
  | 6. 短碎片过滤       | 删掉无意义残行（单符号、1~3 个字符）   | 按行拆分 + 长度阈值                                          |
  | 7. 输出干净文本     | 交给分块环节                           | strip() 收尾                                                 |

  ### 为什么第 2 步必须放在第 4、5 步之前？

  这是最容易踩的坑：

  > 页码被删掉后，它原本前后的两个换行会**相邻叠加**； 如果先压缩空白、再删页码，就会留下「双空格」或「空行残留」。

  顺序口诀：**先删噪声，再规整空白。**

  ## 三、两种输出策略，按场景选

  ### 策略 A：保留段落（长文档 RAG 推荐）

  适用于论文、手册、产品文档、合同。

  ```
  raw_text = re.sub(r'[ \t]+', ' ', raw_text)    # 只压缩行内空格
  raw_text = re.sub(r'(\n\s*)+', '\n', raw_text) # 连续换行+空白 → 单个换行
  ```

  结果：每段一行，段落边界完整。

  ### 策略 B：全部压成一行（短文本检索用）

  适用于 FAQ、标题库、关键词索引。

  ```
  raw_text = re.sub(r'\s+', ' ', raw_text)  # 换行、空格、Tab 全变单个空格
  ```

  结果：整段变一整行，段落信息丢失（**长文档不要这么干**）。

  ## 四、核心正则逐条解释

  | 正则                     | 含义                                         | 作用                                      |
  | ------------------------ | -------------------------------------------- | ----------------------------------------- |
  | r'page\s+\d+\s+of\s+\d+' | page + 空白 + 数字 + 空白 + of + 空白 + 数字 | 匹配英文页码 Page 2 of 12                 |
  | re.IGNORECASE            | 忽略大小写                                   | PAGE 3 OF 12 也能匹配                     |
  | r'[ \t]+'                | 一个或多个**空格或 Tab**                     | 只压行内，不动换行                        |
  | r'(\n\s*)+'              | 「换行 + 任意空白」这一组**重复一次以上**    | 把 \n\n、\n   \n、\n\n\n 全部压成 1 个 \n |
  | r'\s+'                   | 任意空白（含换行）                           | 会抹平所有段落，谨慎使用                  |
  | sub("", text)            | 把匹配到的内容替换为空                       | 相当于删除                                |

  > 关键区别：\n+ 只能匹配**纯换行**，换行中间夹了空格就失效 → 用 (\n\s*)+。

  ## 五、踩坑记录（真实调试过程）

  ### 坑 1：变量写错，删页码白做了

  ```
  text = remove_headers_footers(raw_text)      # 结果存进 text
  raw_text = re.sub(r'[ \t]+', ' ', raw_text)  # 却在用没删页码的 raw_text！
  ```

  **原因**：处理链中途换变量名，两条分支各走各的，页码根本没删。 **修法**：统一用一个变量 raw_text 往下传，每步覆盖它。

  ### 坑 2：删页码后出现「双空格」

  ```
  Large Language Model  Retrieval-Augmented ...   ← 中间两个空格
  ```

  **原因**：先压缩空白 → 再删页码 → 页码位置两侧空格拼到一起。 **修法**：先删页码，再压缩空白。

  ### 坑 3：删页码后残留空行

  删掉页码留下多个连续 \n，用 \n+ 没清干净（因为换行中间夹着空格）。 **修法**：re.sub(r'(\n\s*)+', '\n', raw_text)。

  ### 坑 4：误把段落全压成一行

  用 \s+ → 空格 会连段落一起抹平。 **修法**：区分「行内空白」和「段落换行」，长文档用策略 A。

  ## 六、最终整理版代码

  ```
  import re
  
  def remove_headers_footers(text: str) -> str:
      """去除页码、页眉页脚噪声（中英文页码）"""
      page_en = re.compile(r'page\s+\d+\s+of\s+\d+', re.IGNORECASE)   # Page 1 of 10
      page_cn = re.compile(r'第\s*\d+\s*页\s*/\s*共\s*\d+\s*页')        # 第 1 页 / 共 10 页
      text = page_en.sub("", text)
      text = page_cn.sub("", text)
      return text
  
  def clean_invisible_chars(text: str) -> str:
      """清理不可见脏字符：回车、换页符、零宽字符"""
      return re.sub(r'[\r\f\u200b\u200c\u200d]', '', text)
  
  def extract_text_from_pdf(pdf_path: str) -> str:
      """从 PDF 提取文本（依赖 pdfplumber）"""
      import pdfplumber
      text = ""
      with pdfplumber.open(pdf_path) as pdf:
          for page in pdf.pages:
              page_text = page.extract_text()
              if page_text:
                  text += page_text + "\n"
      return text
  
  def preprocess_document(doc, doc_type: str) -> str:
      """
      文档预处理主函数
      :param doc: 文件路径 / 原始文本
      :param doc_type: "pdf" / "raw_text"
      """
      # 1. 文档解析：拿到原始长文本
      if doc_type == 'pdf':
          raw_text = extract_text_from_pdf(doc)
      elif doc_type == 'raw_text':
          raw_text = doc
      else:
          raise ValueError("doc_type 只能是 pdf / raw_text")
      # 2. 先删固定噪声（顺序优先！RAG清洗经典要点：先删噪声，再处理空白）
      raw_text = remove_headers_footers(raw_text)
      # 3. 清理不可见字符
      raw_text = clean_invisible_chars(raw_text)
      # 4. 行内多个空格 / Tab → 单个空格
      raw_text = re.sub(r'[ \t]+', ' ', raw_text)
      # 5. 段落归一化：任意「换行+空白」连续块 → 单个换行，保留段落边界
      raw_text = re.sub(r'(\n\s*)+', '\n', raw_text)
      # 6. 过滤无意义短碎片行（长度 ≤ 4 丢弃）
      valid_lines = [ln.strip() for ln in raw_text.split("\n")]
      raw_text = "\n".join(ln for ln in valid_lines if len(ln) > 4)
      # 7. 收尾
      text = raw_text.strip()
      # text = text.lower()  # RAG 一般不要转小写，会丢失专有名词信息
      return text
  
  
  # ========== 测试入口（二选一，注释切换） ==========
  if __name__ == "__main__":
      # 方案A：字符串测试（用来验证清洗逻辑）
      test_str = """
      Large Language Model
      
      Page 2 of 12
      
      Retrieval-Augmented Generation (RAG) combines retrieval
      and large language model generation.
      
      Page 3 of 12
      Vector database is used to store embeddings.
  
  
      检索增强生成（RAG）
  
      第 2 页 / 共 10 页
  
      RAG 可以把外部文档交给大模型，减少幻觉。
      """
      result = preprocess_document(test_str, doc_type="raw_text")
  
      # 方案B：解析PDF，使用时打开下面注释，注释掉方案A
      # pdf_path = r"C:\Users\lenovo\Desktop\俸佳璐简历.pdf"
      # result = preprocess_document(pdf_path, doc_type="pdf")
  
      # print("清洗结果：")
      # print(result)
  
      # 可选：把清洗结果保存为txt文件
      with open(r"D:\code\agent\clean_result.txt", "w", encoding="utf-8") as f:
       f.write(result)
      print("\n✅ 已保存至 clean_result.txt")
  ```

  ## 七、测试用例与预期输出

  ### 用例 1：英文页码 + 多空行

  输入：

  ```
  Large Language Model
  
  
  Page 2 of 12
  
  
  Retrieval-Augmented Generation (RAG) combines retrieval
  and large language model generation.
  
  
  Page 3 of 12
  Vector database is used to store embeddings.
  ```

  输出：

  ```
  Large Language Model
  Retrieval-Augmented Generation (RAG) combines retrieval and large language model generation.
  Vector database is used to store embeddings.
  ```

  ### 用例 2：中文页码（检验中文规则）

  输入：

  ```
  检索增强生成（RAG）
  
  
  第 2 页 / 共 10 页
  
  
  RAG 可以把外部文档交给大模型，减少幻觉。
  ```

  输出：

  ```
  检索增强生成（RAG）
  RAG 可以把外部文档交给大模型，减少幻觉。
  ```

  ### 用例 3：大写页码（检验 IGNORECASE）

  输入：PAGE 5 OF 8 → 输出中被完整删除。

  ### 用例 4：无页码纯文本

  应只做空白规整，内容完整保留。

  ## 八、如何真正解析 PDF

  把测试区换成：

  ```
  if __name__ == "__main__":
      pdf_path = "test.pdf"          # PDF 与 py 文件放同一目录即可
      result = preprocess_document(pdf_path, doc_type="pdf")
      print(result)
  
  
      # 可选：把清洗结果存成 txt
      with open("clean_result.txt", "w", encoding="utf-8") as f:
          f.write(result)
  ```

  **Windows 路径写法**：

  ```
  pdf_path = r"C:\code\agent\test.pdf"   # 原始字符串，推荐
  pdf_path = "C:/code/agent/test.pdf"    # 正斜杠也可以
  ```

  **依赖安装**：

  ```
  pip install pdfplumber
  pip install beautifulsoup4   # 后面处理 HTML 时用
  ```

  **常见报错**：

  | 报错                            | 原因              | 解决                       |
  | ------------------------------- | ----------------- | -------------------------- |
  | FileNotFoundError               | 路径错            | 检查 PDF 位置 / 用绝对路径 |
  | PasswordError                   | PDF 加密          | 先解密                     |
  | 输出为空                        | 扫描版（图片）PDF | 需要 OCR（PaddleOCR 等）   |
  | ModuleNotFoundError: pdfplumber | 依赖没装          | pip install pdfplumber     |

  ## 九、当前版本 vs 生产级版本的差距

  **已具备（符合主流范式）**

  - 顺序正确：先删噪声 → 再规整空白

  - 保留段落边界，未抹平语义结构

  - 区分「行内空格」和「段落换行」

  - 中英文页码规则

  - 不可见字符清理

  - 短碎片过滤

  **生产环境还需补充**

  - 页眉标语 / 版本号 / 版权文字的识别（正则难覆盖，需规则集或模型）

  - 表格处理（转 Markdown 或丢弃）

  - 公式处理（LaTeX 保留原始标记）

  - 图片 OCR 分支

  - 跨页重复文本去重

  - 全角半角 / 繁简统一

  - try-except 异常兜底、日志记录

  > 定位：**课程作业 / Demo 完全合格**；企业上线需补上述模块。

  ## 十、下一步衔接：分块（Chunking）

  清洗完之后紧接着是分块，常见策略：

  | 策略         | 说明                             | 适用                 |
  | ------------ | -------------------------------- | -------------------- |
  | 固定长度分块 | 按字符数/Token 数硬切            | 简单快速，易切断语义 |
  | 递归分块     | 按段落 → 句子 → 字符逐级降级切分 | 通用推荐             |
  | 语义分块     | 按语义相似度切分                 | 质量最好，成本高     |
  | 结构化分块   | 按标题层级切分                   | 手册、说明书         |

  关键参数：chunk_size（块大小）、chunk_overlap（块间重叠，避免信息断裂）。

  > 段落边界之所以不能丢：递归分块的第一刀就是按 \n 切段落。段落没了，分块质量直接崩。

  ## 附：速查清单

  - 清洗顺序：解析 → 删噪声 → 去脏字符 → 压行内空格 → 归一换行 → 过碎片 → strip

  - 处理链只用一个变量，别中途换名

  - 长文档保留段落，短文本才全压一行

  - .lower() 默认不要开

  - 正则只治模板噪声，复杂页眉得靠规则集
# RAG 文档分块（Chunking）笔记

> 主题：把清洗干净的整篇长文本，切成大小合适、语义尽量完整的小段 一句话定位：**分块 = 按语义边界切，而不是按字符数硬剁**

## 一、在 RAG 流程中的位置

```
文档上传 → 数据清洗 → 【文本分块】 → 向量化 Embedding → 存入向量库 → 检索 → 交给大模型生成
```

分块是清洗之后、向量化之前的**中间环节**，切得好不好直接决定召回准不准。

## 二、为什么要分块

1. **模型有长度上限**：embedding 模型、大模型都不能一次吃整本书。

1. **语义会被稀释**：整篇文档压成一个向量，语义混杂，检索时很难精准命中某一小段。

1. **降低幻觉**：检索只取回相关的几个小块，送进 prompt 的无关内容少，模型瞎编的概率就低。

> 类比：问「林黛玉是谁」，你不会把 500 页《红楼梦》整本递过去，而是翻到讲她的那几页。

## 三、四种主流策略对比

本笔记只是对滑动窗口的练习，其他策略后续学习

| 策略              | 怎么切                       | 优点                         | 缺点                        | 适用                          |
| ----------------- | ---------------------------- | ---------------------------- | --------------------------- | ----------------------------- |
| 固定长度分块      | 按字符数硬切                 | 最简单、最快                 | 容易把句子/知识点从中间劈断 | 快速原型、简单文本            |
| 滑动窗口分块      | 固定长度 + 重叠              | 重叠能保住跨块上下文         | 依然不看语义边界            | 入门、结构混乱的文本          |
| **递归分块**      | 段落 → 句子 → 字符，逐级降级 | 优先在语义断点切，长度还可控 | 稍复杂                      | **绝大多数 RAG 项目（主流）** |
| 语义 / 结构化分块 | 按语义相似度或标题层级切     | 精度最高                     | 成本高、依赖文档结构        | 高质量知识库、手册、论文      |

## 四、两个核心参数

- **chunk_size 块大小**

太小 → 单块语义残缺，检索回来一堆碎片；

太大 → 一块里混进大量无关内容，检索不精准。

中文推荐 **200~500 tokens**（约 150~400 汉字）。

- **chunk_overlap 重叠长度**

相邻两块重复保留的一段，防止知识点正好卡在两块的接缝上。

推荐 **块大小的 10%~20%**（300 块用 50 重叠，合适）。

> ⚠️ **最大的坑：字符 ≠ token。** 中文 1 个汉字 ≈ 1.3~1.5 token，换行 \n 也算 1 个字符。代码里写的是 len()，所以参数填的是**字符数**，不要直接当 token 用。

## 五、滑动窗口原理（核心公式）

```
step = chunk_size - overlap
```

步长 = 块长 − 重叠。窗口每次只往后挪 step，这样两块之间天然留出 overlap 个字符的重合。

以 chunk_size=300, overlap=50 为例，step=250：

```
块0：[0 ..................... 300]
块1：        [250 ..................... 550]
块2：                [500 ..................... 800]
                 ↑重叠区 250~300、500~550
```

- i 取值：0, 250, 500, 750 …

- 最后一块不足 300 字符时，切片自动取剩余全部，不会报错

对比：普通固定分块 step = chunk_size，块与块完全不重合，一句话被切断就彻底丢失上下文。

## 六、整理版代码

```
import re

# ---------- 分块：滑动窗口 ----------
def chunk_by_sliding_window(text: str, chunk_size: int = 300, overlap: int = 50):
    """滑动窗口分块，带重叠"""
    chunks = []
    step = chunk_size - overlap          # 每轮向后移动的距离
    for i in range(0, len(text), step):
        one_chunk = text[i:i + chunk_size]
        chunks.append(one_chunk)
    return chunks

# ---------- 主流程：解析 → 清洗 → 分块 ----------
def preprocess_document(doc, doc_type: str, chunk_size: int = 300, overlap: int = 50):
    if doc_type == 'pdf':
        raw_text = extract_text_from_pdf(doc)
    else:
        raise ValueError("当前仅支持 pdf")

    raw_text = remove_headers_footers(raw_text)     # 删页码页眉
    raw_text = clean_invisible_chars(raw_text)      # 清不可见字符
    raw_text = re.sub(r'[ \t]+', ' ', raw_text)     # 行内空格压缩
    raw_text = re.sub(r'(\n\s*)+', '\n', raw_text)  # 段落归一
    valid_lines = [ln.strip() for ln in raw_text.split("\n")]
    raw_text = "\n".join(ln for ln in valid_lines if len(ln) > 4)  # 过滤短碎片
    clean_text = raw_text.strip()

    chunk_list = chunk_by_sliding_window(clean_text, chunk_size, overlap)
    return clean_text, chunk_list                   # 返回：全文 + 分块列表



if __name__ == "__main__":
    pdf_path = r"C:\Users\lenovo\Desktop\俸佳璐简历.pdf"
    clean_result, chunk_list = preprocess_document(pdf_path, "pdf", 300, 50)


    with open(r"D:\code\agent\clean_result.txt", "w", encoding="utf-8") as f:
        f.write(clean_result)


    with open(r"D:\code\agent\chunk_result.txt", "w", encoding="utf-8") as f:
        f.write(f"一共切出 {len(chunk_list)} 块\n\n")
        for idx, chunk in enumerate(chunk_list):
            f.write(f"======== Chunk {idx} ========\n")
            f.write(chunk + "\n")
            f.write(f"本块字符长度：{len(chunk)}\n")
            f.write("-" * 40 + "\n")


    print(f"一共切出 {len(chunk_list)} 块")
```

**返回值说明**：chunk_list 是 **list**，每个元素是 **str**（一小段文本）。后面做向量化就是逐个遍历这个 list。

## 七、踩坑记录

### 坑 1：把字符数当 token

chunk_size=300 指的是 300 个字符，中文实际约 400~450 token。填参数前先想清楚用的是哪种单位。

### 坑 2：overlap 太大，块数暴涨

chunk_size=50, overlap=10 时 step 只有 40，一份简历能切出几十块，内容高度重复。 overlap 一般不超过块长的 20%，否则等于把同一段文本反复入库。

### 坑 3：被短碎片过滤吃掉内容

len(ln) > 4 会把标题、单字、编号行一起删掉。清洗阶段觉得没问题，但分块时发现原文变短了、语义缺块 —— 这个阈值要按文档类型调。

### 坑 4：块边界断句

纯字符切法不看句号、段落。检查 chunk_result.txt 时经常看到半句话开头结尾，这就是纯字符分块的固有缺陷，所以才要上递归分块。

## 八、当前版本 vs 生产级

**已具备**

- 带重叠，保住跨块上下文

- 参数可配置，块长/重叠都在调用处控制

- 与清洗流程完整串联，输出可直接喂给 embedding

**生产还需补**

- 换成递归分块（按 \n\n → \n → 。 → 空格 → 字符逐级切）

- token 计数用 tiktoken / 模型自带 tokenizer，而不是 len()

- 表格、公式单独抽出来处理，别混在正文里切

- 按标题层级做结构化分块（手册、说明书类文档）

- 给每个 chunk 附 metadata（来源文件、页码），检索后好溯源

## 九、下一步：Embedding

```
for chunk in chunk_list:
    vector = embedding_model.encode(chunk)   # 每个块一个向量
    # 向量 + 原文一起写入 FAISS / Chroma
```

## 附：速查清单

- 一句话原理：窗口大小 chunk_size，每次挪 chunk_size - overlap，留出重叠

- 参数：中文 200~500 token，overlap 取 10%~20%

- 单位：代码里是**字符**，不是 token

- 检查手段：打开 chunk_result.txt，看块开头结尾是不是半句话

- 想上主流方案：换 RecursiveCharacterTextSplitter
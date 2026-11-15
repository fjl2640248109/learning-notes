- # RAG 查询优化（RAG-Fusion 多查询融合）笔记

> 主题：对多条查询的检索结果做倒数排序融合（RRF），产出一份全新的全局排序
>
> 一句话定位：**RAG-Fusion = Multi-Query（多路召回） + RRF（按排名加权求和），把"堆结果"升级成"算排序"**

## 一、在整条 RAG 流程中的位置

```
用户原始问题 →【LLM生成多条同义查询】→ 每条query独立检索 →【RRF融合重排】→ Top-K 上下文 → LLM 生成答案
                              ↑ 1.1.5 Multi Query              ↑ 1.1.6 RAG-Fusion
```

和上一节的关系一句话说清：

| 环节                 | 谁负责                        | 产出                       |
| -------------------- | ----------------------------- | -------------------------- |
| Multi Query（1.1.5） | 生成多条改写 query + 多路检索 | **多组互相独立的候选列表** |
| RAG-Fusion（1.1.6）  | 对这些列表做 RRF 融合         | **一份统一的全局排序**     |

> Multi Query 只保证"查得广"，RAG-Fusion 负责"排得对"。

## 二、核心：RRF 到底在算什么

```
RRF(doc) = Σ  1 / (k + rank)
```

| 符号   | 含义                                                         | 本代码取值                 |
| ------ | ------------------------------------------------------------ | -------------------------- |
| `rank` | 该文档在**某一个检索列表**里的名次，**从 1 开始**            | 第1名 rank=1，第2名 rank=2 |
| `k`    | 平滑常数，削弱尾部名次的贡献，防止"第1名通吃"                | 60                         |
| `Σ`    | 该文档在**所有列表**中出现的每一次都累加；没出现的那一路不加分 | —                          |

三条铁律：

1. **只用排名，不用分数**：向量返回的 0.96 / 0.92 一律丢弃，只看第几名

1. **排名越前，单次贡献越大**：第1名 1/61，第10名 1/70，差距逐级衰减

1. **多次命中会累加**：被多条 query 都检索到的文档，分数叠加

> 为什么这样做：不同检索器（向量 / BM25 / 不同 embedding 模型）分数尺度完全不同，**没法直接相加**；换成"名次"这一种统一语言，就能公平比较。

## 三、代码结构（4 个函数各管一段）

| 函数                     | 扮演的角色                                     | 真实项目里替换成                  |
| ------------------------ | ---------------------------------------------- | --------------------------------- |
| `generate_multi_queries` | 模拟 LLM 生成同义查询（硬编码字典）            | 真调 DeepSeek 生成 3~5 条 query   |
| `mock_vector_search`     | 模拟向量库检索，返回 `(文档文本, 相似度)` 列表 | Pinecone `index.query` / 混合检索 |
| `reciprocal_rank_fusion` | **核心**：RRF 打分 + 全局降序重排              | 不替换，这就是算法本体            |
| `__main__`               | 串联流程 + 取 Top3                             | —                                 |

### 核心函数逐行

```
fused_scores = {}
for _, doc_list in search_results_dict.items():      # 遍历每一条query的检索结果
    for rank, (doc_text, _) in enumerate(doc_list, start=1):   # rank从1开始，注意丢弃原始分数
        if doc_text not in fused_scores:
            fused_scores[doc_text] = 0
        fused_scores[doc_text] += 1.0 / (k + rank)     # RRF 累加
sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
```

三个设计要点：

- `enumerate(..., start=1)`：rank 必须从 1 起，从 0 起会让 `1/(k+0)` 凭空多出高分

- `(doc_text, _)` 里的下划线：**原始 score 被显式丢弃**，这是 RRF 的关键动作

- 用 `doc_text` 当字典 key：文本相同的文档自动合并 → **顺手完成去重**，不需再写一个 dedupe 函数

### 完整代码

```
import json

def generate_multi_queries(original_query: str, num=3):
    """模拟LLM生成多条同义查询（这里硬编码模拟大模型输出，省去API调用）"""
    prompt_map = {
        "Python异步编程的优势是什么？": [
            "Python异步编程有什么好处？",
            "asyncio怎样提升Python编程效率？",
            "为什么使用Python异步IO？"
        ]
    }
    return prompt_map.get(original_query, [original_query])


def reciprocal_rank_fusion(search_results_dict, k=60):
    """RRF倒数排序融合 核心实现
    search_results_dict: dict, key:query, value: list[(doc_text, raw_score)]
    """
    fused_scores = {}
    for _, doc_list in search_results_dict.items():
        # rank从1开始
        for rank, (doc_text, _) in enumerate(doc_list, start=1):
            if doc_text not in fused_scores:
                fused_scores[doc_text] = 0
            fused_scores[doc_text] += 1.0 / (k + rank)
    # 降序排序
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs


def mock_vector_search(query, top_k=4):
    """模拟向量数据库检索，返回[(文档文本,相似度分数)]"""
    mock_lib = {
        "Python异步编程的优势是什么？": [
            ("DocA：异步IO提升并发能力", 0.96),
            ("DocB：asyncio适合IO密集型任务", 0.92),
            ("DocC：异步避免线程开销", 0.85),
            ("DocD：GIL锁对多线程限制", 0.78)
        ],
        "Python异步编程有什么好处？": [
            ("DocB：asyncio适合IO密集型任务", 0.95),
            ("DocE：异步提升接口吞吐量", 0.88),
            ("DocC：异步避免线程开销", 0.84),
            ("DocF：协程切换开销很小", 0.75)
        ],
        "asyncio怎样提升Python编程效率？": [
            ("DocC：异步避免线程开销", 0.93),
            ("DocB：asyncio适合IO密集型任务", 0.90),
            ("DocA：异步IO提升并发能力", 0.82),
            ("DocF：协程切换开销很小", 0.74)
        ],
        "为什么使用Python异步IO？": [
            ("DocA：异步IO提升并发能力", 0.94),
            ("DocB：asyncio适合IO密集型任务", 0.91),
            ("DocE：异步提升接口吞吐量", 0.81),
            ("DocD：GIL锁对多线程限制", 0.72)
        ]
    }
    return mock_lib.get(query, [])[:top_k]


if __name__ == "__main__":
    original_user_query = "Python异步编程的优势是什么？"
    print(f"用户原始问题：{original_user_query}\n")

    # 1.生成多条同义查询
    multi_queries = generate_multi_queries(original_user_query, num=3)
    print(f"生成的多查询：{multi_queries}\n")

    # 2.每条query执行模拟检索
    query_result_map = {}
    for q in multi_queries:
        res = mock_vector_search(q, top_k=4)
        query_result_map[q] = res
        print(f"【{q}】检索返回：{[d[0] for d in res]}")

    # 3.RRF融合
    fused_result = reciprocal_rank_fusion(query_result_map, k=60)
    print("\n==== RAG‑Fusion RRF融合后综合排序 ====")
    for doc_text, score in fused_result:
        print(f"score={score:.4f} | {doc_text}")

    # 4.取top3上下文，注意：传给LLM回答仍然使用【原始问题】
    top3_context = [item[0] for item in fused_result[:3]]
    print("\n==== 送入LLM的上下文top3 ====")
    for c in top3_context:
        print("-", c)

    print("\n> 给大模型提问使用：", original_user_query)

```

## 四、跑这组数据的真实结果

**三路检索返回（按名次）**

| 查询                               | rank1 | rank2 | rank3 | rank4 |
| ---------------------------------- | ----- | ----- | ----- | ----- |
| Q1 Python异步编程有什么好处？      | DocB  | DocE  | DocC  | DocF  |
| Q2 asyncio怎样提升Python编程效率？ | DocC  | DocB  | DocA  | DocF  |
| Q3 为什么使用Python异步IO？        | DocA  | DocB  | DocE  | DocD  |

**RRF 得分明细（k=60）**

| 文档     | 出现次数 | 命中的名次       | 计算过程           | 总分       |
| -------- | -------- | ---------------- | ------------------ | ---------- |
| **DocB** | 3 次     | Q1:1、Q2:2、Q3:2 | 1/61 + 1/62 + 1/62 | **0.0487** |
| **DocC** | 2 次     | Q1:3、Q2:1       | 1/63 + 1/61        | **0.0323** |
| **DocA** | 2 次     | Q2:3、Q3:1       | 1/63 + 1/61        | **0.0323** |
| **DocE** | 2 次     | Q1:2、Q3:3       | 1/62 + 1/63        | **0.0320** |
| **DocF** | 2 次     | Q1:4、Q2:4       | 1/64 + 1/64        | 0.0313     |
| **DocD** | 1 次     | Q3:4             | 1/64               | 0.0156     |

**最终排序**

```
DocB (0.0487) > DocC (0.0323) > DocA (0.0323) > DocE (0.0320) > DocF (0.0313) > DocD (0.0156)
```

### 从这组结果能读出三个结论

1. **DocB 三路全中且都在前二 → 稳居第一**

说明"被多条 query 反复命中靠前位置"才是融合的真正赢家。

1. **原始相似度最高的 DocA（0.96）并没有排第一**

RRF 完全无视原始分数，只看名次 —— 这就是它与普通加权打分最大的区别。

1. **DocF 出现 2 次却输给只出现 1 次的 DocD 之外所有文档？不，反过来看：**

DocF 两次都是第 4 名（0.0313），DocD 一次第 4 名（0.0156），**频率奖励确实生效**；

但对比 DocA/DocC（两次，含一次第1名，0.0323），**位置质量比出现次数更值钱**。

> 结论口诀：**排名靠前 > 出现次数多 > 原始相似度高**。三者里 RRF 只认前两个。

## 五、RAG-Fusion 对比"简单拼接"

| 维度       | 普通 Multi Query（extend 堆列表） | RAG-Fusion（RRF 融合）           |
| ---------- | --------------------------------- | -------------------------------- |
| 排序依据   | 各查询的原始相似度，尺度不统一    | 名次加权累加，尺度统一           |
| 去重       | 需要单独写 dedupe                 | 字典 key 天然去重                |
| 结果多样性 | 容易被某一路结果霸榜              | 多路均衡，共同命中的文档自然上前 |
| 鲁棒性     | 某条 query 分数虚高就会污染全局   | 单条差 query 的影响被稀释        |
| 依赖       | 需要可信的相似度分数              | **不依赖分数，只看名次**         |

## 六、适用 / 不适用

✅ 适合

- 用户问题短、含糊、多角度，需要多路召回

- 要融合**多种检索来源**（向量 + BM25，甚至不同 embedding 模型的结果）

- 对召回率、结果多样性要求高

❌ 不适合

- 简单事实查询：多轮 LLM + 多次检索的代价远大于收益

- 强实时场景：时延会明显上升

- 资源受限环境

## 七、踩坑记录

### 坑 1：rank 从 0 开始

`enumerate(doc_list)` 默认从 0 起 → 第 1 名变成 `1/(60+0)`，分数虚高且顺序全乱。

**修法**：`enumerate(doc_list, start=1)`。

### 坑 2：把 RRF 分数当业务相似度

RRF 分数（0.015~0.048 这个量级）**只有相对排序意义，绝对值没有业务含义**，不要拿它去做阈值过滤（比如"低于 0.5 丢弃"必然全丢）。

### 坑 3：k 取值随意

k 常见 50~100，默认 60。k 太小 → 头部名次差距被放大，第 1 名几乎通吃；k 太大 → 各名次趋同，排名信息被抹平。

### 坑 4：用它替代 Rerank

RRF 是**用名次做粗排融合**，它无法判断"query 与 chunk 的语义到底多相关"。噪声真正要靠 Rerank（CrossEncoder 逐对打分）来压。

## 八、当前版本 vs 生产级

**已具备**

- RRF 公式、rank 起点、k 常数全部正确

- 天然去重（字典 key 为文本）

- 原始分数被显式丢弃，规避了多路尺度不一致

- 最终回答锚定原始问题

**补齐后就是生产可用**

- `generate_multi_queries` / `mock_vector_search` 换成真实 LLM 与 Pinecone 调用

- 融合后加一层 Rerank 精排

- 文档改用稳定 ID 做 key（当前用文本，改一个字就成两篇）

- 记录每一路检索的贡献占比，便于调 num 和 k

- 加评估：对比单 query vs RAG-Fusion 的 Recall@K

> 定位：**这一版是纯算法 Demo，4 个函数替换掉 2 个就能接进你已有的工程代码。**

## 附：速查清单

- RAG-Fusion = Multi Query 多路召回 + RRF倒数排序融合

- 公式 `Σ 1/(k+rank)`，rank **从 1 开始**，k 默认 60

- **只认名次，不认原始相似度分数**

- 排名靠前 > 出现次数多 > 原始分数高

- 字典 key 用文本或文档 ID，天然去重

- RRF 分数只用于相对排序，不能做阈值

- 融合后仍可能噪声多 → 需要 Rerank 兜底

- 生成答案永远用 `original_query`

- 代价：1 次 LLM 生成 query + N 次检索，高并发慎用
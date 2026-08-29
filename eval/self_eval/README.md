# Search self-eval

这套评测把当前 `zeta-engine` 当作只读被测对象，不修改 `src/zeta_engine`。
`freeze` 会把源码复制到 `frozen_src`；后续评测从快照导入代码，因此工作树继续变化也
不会改变被测实现。语料和索引不复制，但会逐文件校验 SHA-256。

三层测试：

1. 从页面标题自动生成精确查找题，标题相同的 URL 都算正确；
2. 从页面标题类别自动生成自然搜索题，由人从脚本给出的候选中选择一个或多个正确 URL；
3. 从第二层自动派生简称、关键词和去标点挑战题，沿用人工选择的答案。

## 运行流程

```bash
.venv/bin/python eval/self_eval/self_eval.py freeze
.venv/bin/python eval/self_eval/self_eval.py generate
```

只编辑 `selections.jsonl`，每行格式如下。允许多个答案：

```json
{"id":"l2-0001","relevant_urls":["http://example/a.htm","http://example/b.htm"]}
```

然后装配测试集并评测：

```bash
.venv/bin/python eval/self_eval/self_eval.py build
.venv/bin/python eval/self_eval/self_eval.py evaluate
```

在确认本机 PyTorch 支持 MPS 时可追加 `--device mps`。

默认比较 BM25F、Dense 和 Hybrid，输出第一、第五、二十名召回率、MRR@20、
平均/p50/p95 时延，以及第三层相对第二层的成功保持率。若冻结的源码、语料或
索引发生变化，脚本会拒绝继续运行；需要评测新版本时显式重新执行 `freeze`。

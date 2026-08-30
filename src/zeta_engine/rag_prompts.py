"""Prompt builders for answer generation, control, and verification."""

from zeta_engine.rag_config import AGENT_MAX_QUERIES


def build_prompt(
    query: str,
    context: str,
    *,
    include_citations: bool = False,
    verification_feedback: str = "",
) -> str:
    """Build the final-answer prompt used after evidence gathering."""

    output_instruction = (
        "只输出最终答案，并用证据ID标注依据。"
        if include_citations
        else "只输出最终答案；不输出思考过程、前言、总结或引用标签。"
    )
    feedback = verification_feedback or "（无）"
    return f"""请根据检索材料直接回答问题。
材料可能不完整；优先给出有用答案，必要时简短说明不确定之处。
可以根据材料做直接推理和计算，不要求答案必须在原文中逐字出现。
多个材料描述相似但不同的事件时，优先采用标题、措辞和限定条件与问题最匹配的材料，
不要因为另一条材料较新就自动替换，也不要混合不同事件；仍无法区分时采用相关性优先级更高的材料。
{output_instruction}
如有上次校验反馈，请据此修正：{feedback}

问题：{query}

<检索材料>
{context}
</检索材料>

最终答案："""


def build_agent_prompt(
    query: str,
    context: str,
    searched_queries: list[str],
    *,
    remaining_cycles: int,
    verification_feedback: str = "",
    collection_available: bool = False,
    collection_queries: list[str] | None = None,
) -> str:
    """Build the controller prompt for answer/search/calculation decisions."""

    searched = "\n".join(f"- {item}" for item in searched_queries) or "- （无）"
    inspected = "\n".join(
        f"- {item}" for item in (collection_queries or [])
    ) or "- （无）"
    feedback = verification_feedback or "（无）"
    budget = (
        "不得再返回 search；必须根据已有材料返回 answer。"
        if remaining_cycles == 0
        else (
            f"当前还可发起 {remaining_cycles} 轮搜索。先完成回答所必需且尚未执行的工具操作，"
            "随后优先 answer；只有核心信息完全缺失时才 search。"
        )
    )
    collection_rule = (
        """2. 先判断答案是否依赖完整名单、表格或重复记录的计数、筛选、交集，或者依赖这些集合计算比例、平均值、上下界。
   如果依赖且尚未执行对应集合查询，优先返回 collection，让工具扫描完整资源并保留分组边界；
   不要手工枚举长列表，也不要把当前 Top-K 当作完整集合。
3. 计算派生量时，操作数必须来自问题和同一口径的材料。不得凭常识引入二者都未限定的新总体；
   有多种解释时，优先选择不增加新实体、能够用现有完整证据闭合计算的解释，并在答案中简短说明口径。"""
        if collection_available
        else ""
    )
    collection_action = (
        """- collection：
  {"action":"collection","query":"保留原问题全部范围、筛选条件和计算目标的集合查询","slots":[]}
"""
        if collection_available
        else ""
    )
    return f"""你负责决定现在直接回答、继续检索、扫描完整集合，还是执行一次确定性计算。
尽量给出答案；材料不完整时可以给出部分答案或注明不确定性。
只输出一个 JSON 对象。

选择规则：
1. 相关性优先级最高的材料能够直接、完整回答时，采用它；除非问题明确给出的条件只匹配另一条记录。
   不得自行增加问题没有给出的版本、阶段或日期限定来改选低优先级记录。
{collection_rule}
4. 完整证据已经给出计算输入且希望避免算错时，返回 calculate。
   问题询问最低、最高或其他边界，而同口径材料已经给出数量约束时，应计算该边界；不要改为索要最终实现值。
5. 材料能支持一个有用答案，或能通过直接推理得到答案时，返回 answer。
6. 只有回答所需的核心事实没有出现时，才返回 search；不要重复已经搜索过的查询。
7. 多条材料描述相似但不同的记录时，只使用问题明确给出的标题、措辞、时间、对象、版本或阶段限定。
   问题没有给出区分条件时，不得自行增加版本、阶段或日期；在都能直接回答的记录中采用相关性优先级更高的材料。

动作格式：
- answer:
  {{"action":"answer","answer":"最终答案","claims":[],"slots":[]}}
- search（最多 {AGENT_MAX_QUERIES} 个）:
  {{"action":"search","queries":["缺失的核心事实"],"slots":[]}}
{collection_action}
- calculate:
  {{"action":"calculate","calculations":[{{"name":"结果名","operator":"count","values":["逐项输入"],"evidence_ids":[]}}],"slots":[]}}
  派生量示例：{{"name":"比例","operator":"ratio","items":[{{"value":3,"evidence_ids":["ev_x"]}},{{"value":8,"evidence_ids":["ev_y"]}}]}}
  其他可用 operator：sum、min、max、sort、classify、add、subtract、multiply、divide、ratio、intersection、union、difference、complement。

claims 和 slots 可以为空；能确认来源时再填写 evidence_ids。

{budget}

原始问题：{query}

已搜索查询：
{searched}

已执行集合查询：
{inspected}

上次校验反馈：{feedback}

<检索材料>
{context}
</检索材料>"""

def build_verifier_prompt(query: str, context: str, answer: str) -> str:
    """Build the evidence-and-derivation audit prompt."""

    return f"""请检查候选答案是否基本回答了问题，以及是否与检索材料明显矛盾。
不要因为材料不完整、答案包含合理推断或措辞不够精确而拒绝。
只输出一个 JSON 对象：
{{
  "valid": true,
  "requirements": [
    {{"description": "问题要求的一项事实", "satisfied": true, "evidence_ids": []}}
  ],
  "claims": [
    {{"statement": "候选答案的一项关键断言", "status": "supported", "evidence_ids": [], "calculation_ids": []}}
  ],
  "conflicts": [],
  "issues": [],
  "queries": []
}}

判定规则：
1. 先检查候选答案的来源选择。相关性优先级最高的材料能直接、完整回答时，应采用它；
   候选答案如果自行增加问题没有给出的版本、阶段或日期，并因此改用冲突的低优先级记录，必须 valid=false。
2. 把用户明确要求的每个输出分别列入 requirements；任何一项没有回答时 valid=false。
3. claims 覆盖候选答案的主要断言。允许基于材料进行直接推理、概括和计算，
   不要求结论在原文中逐字出现。
4. 问题要求总数、全部项目或跨多个对象汇总时，当前 Top-K 中出现的记录不等于完整集合；
   证据不能说明覆盖完整范围时 valid=false，并生成针对遗漏对象或范围的补充查询。
   如果确定性集合工具已经给出完整扫描、分组边界和计数，应将其视为完整集合证据。
5. 答案与材料明显矛盾、计算明显错误或漏掉明确要求时 valid=false。
   比例、平均值和上下界等派生量应按问题措辞和已有集合确定操作数，不得预设材料必须提供某个未被问题指定的口径。
   不得用常识中的另一种指标定义替换候选答案明确说明且由同口径证据支持的定义。
   边界问题已有同口径数量约束时，不得仅因缺少最终实现值而判为材料不足。
6. 多条材料描述相似但不同的事件时，候选答案应来自与问题标题、措辞和限定最匹配的记录，
   不能仅因另一条材料较新而替换。问题未给版本、阶段或日期限定时，候选答案不得自行增加这些限定，
   并应采用能直接回答问题且相关性优先级更高的材料。
7. valid=false 且缺少核心事实时，可生成最多 {AGENT_MAX_QUERIES} 个新查询；否则 queries 为空。
8. valid=true 时，requirements 均 satisfied、claims 均 supported、conflicts 均 resolved，issues 为空。

问题：{query}

候选答案：{answer}

<检索材料>
{context}
</检索材料>"""

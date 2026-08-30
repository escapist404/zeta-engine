import unittest

from zeta_engine.collection_ops import (
    aggregate,
    calculate,
    checkpoint,
    declared_record_count,
    filter_records,
    infer_record_kind,
    join,
    locate,
    measure,
    project,
    scan,
    segment,
    semantic_map,
    set_op,
    sort_records,
    validate,
)


class CollectionOpsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scanned = scan(
            "source-a",
            title="项目汇总",
            url="https://example.test/projects",
            text="",
            content_html=(
                "<main><p>本年度共3项项目。</p>"
                "<p>项目名称：甲计划</p><p>负责人：张三</p><p>状态：通过</p>"
                "<p>项目名称：乙计划</p><p>负责人：李四</p>"
                "<p>项目名称：丙计划</p><p>负责人：张三</p><p>状态：通过</p>"
                "</main>"
            ),
        )
        self.collection = segment(self.scanned)

    def test_scan_segment_project_and_measure_are_domain_independent(self) -> None:
        self.assertTrue(self.scanned.complete)
        self.assertEqual(self.collection.start_field, "项目名称")
        self.assertEqual(len(self.collection.records), 3)
        self.assertEqual(
            project(self.collection.records[1], ["项目名称", "状态"]),
            {"项目名称": ("乙计划",), "状态": ()},
        )

        kind = infer_record_kind(self.collection.start_field)
        declared = declared_record_count(self.scanned, kind=kind)
        coverage = measure(
            self.scanned,
            self.collection,
            declared_records=declared,
        )
        self.assertEqual(kind, "项目")
        self.assertEqual(declared, 3)
        self.assertTrue(coverage.record_complete)
        self.assertEqual(coverage.parsed_records, 3)

    def test_filter_aggregate_and_set_operations_are_deterministic(self) -> None:
        selected = filter_records(
            self.collection.records,
            field="负责人",
            operator="contains",
            value="张三",
        )

        self.assertEqual(aggregate(selected, operator="count"), 2)
        self.assertEqual(
            set_op(
                ["检索", "智能体"],
                ["智能体", "多模态"],
                operator="intersection",
            ),
            ("智能体",),
        )

    def test_locate_checkpoint_sort_join_calculate_and_validate_compose(self) -> None:
        located = locate(
            "项目",
            lambda _query, _limit: ["2024项目", "2025项目", "2026项目"],
            constraints=[lambda item: "2024" not in item],
        )
        self.assertEqual(located, ("2025项目", "2026项目"))

        first_page = checkpoint(self.scanned, limit=3)
        second_page = checkpoint(
            self.scanned,
            cursor=first_page.next_cursor or 0,
            limit=20,
        )
        self.assertFalse(first_page.complete)
        self.assertTrue(second_page.complete)
        self.assertEqual(
            len(first_page.blocks) + len(second_page.blocks),
            len(self.scanned.blocks),
        )

        ordered = sort_records(
            self.collection.records,
            key=lambda record: {
                "甲计划": 1,
                "乙计划": 2,
                "丙计划": 3,
            }[record.values("项目名称")[0]],
            reverse=True,
        )
        self.assertEqual(ordered[0].values("项目名称"), ("丙计划",))
        joined = join(
            self.collection.records,
            [("张三", "技术部"), ("李四", "运营部")],
            left_key=lambda record: record.values("负责人")[0],
            right_key=lambda item: item[0],
        )
        self.assertEqual(len(joined), 3)
        self.assertEqual(calculate(3, operator="divide", right=2), 1.5)
        assignments = semantic_map(
            self.collection.records,
            item_id=lambda record: record.record_id,
            mapper=lambda record: (
                "重点项目" if record.values("负责人") == ("张三",) else "普通项目",
            ),
        )
        self.assertEqual(assignments["source-a:record:1"], ("重点项目",))

        report = validate(
            self.collection.records,
            required_fields=["项目名称", "状态"],
        )
        self.assertFalse(report.valid)
        self.assertEqual(report.issues[0].record_id, "source-a:record:2")


if __name__ == "__main__":
    unittest.main()

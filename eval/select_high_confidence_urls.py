import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/day5_query_target_urls.jsonl"
TARGET = ROOT / "outputs/day5_query_target_urls_high_confidence.jsonl"

# 1-based source line -> URLs retained after the 50-agent audit.
# Deliberately capped at three direct-answer URLs per query.
SELECTED = {
    1: (
        "http://www.phys.ruc.edu.cn/",
        "http://www.phys.ruc.edu.cn/xygkx/xyjj.htm",
    ),
    2: (
        "http://www.phys.ruc.edu.cn/kxyj/yjtd/yzyfzwl_lzxxylzjs/tdjs.htm",
        "http://www.phys.ruc.edu.cn/kxyj/yjtd/yzyfzwl_lzxxylzjs.htm",
    ),
    3: ("http://www.phys.ruc.edu.cn/rcpy/bksjy/pyfa.htm",),
    4: (
        "http://www.phys.ruc.edu.cn/szdw/jzry/ayjfxpx/clwl/lc.htm",
        "http://www.phys.ruc.edu.cn/szdw/jzry/ayjfxpx/clwl/css.htm",
        "http://www.phys.ruc.edu.cn/szdw/jzry/ayjfxpx/jswlffjyy/yr.htm",
    ),
    5: (
        "http://www.phys.ruc.edu.cn/rcpy/bksjy/pyfa.htm",
        "http://www.phys.ruc.edu.cn/rcpy/bksjy/jpkc.htm",
        "http://www.phys.ruc.edu.cn/rcpy/bksjy/jxms.htm",
    ),
    7: ("http://www.phys.ruc.edu.cn/gjjl1/xshy.htm",),
    8: ("http://www.phys.ruc.edu.cn/info/1033/2762.htm",),
    9: (
        "http://www.phys.ruc.edu.cn/szdw/jzry/ayjfxpx/njtwxsy/wjc.htm",
        "http://www.phys.ruc.edu.cn/info/1160/1502.htm",
        "http://www.phys.ruc.edu.cn/info/1033/2797.htm",
    ),
    10: ("http://www.phys.ruc.edu.cn/info/1146/1813.htm",),
    11: ("http://www.phys.ruc.edu.cn/info/1033/2870.htm",),
    12: (
        "http://www.phys.ruc.edu.cn/info/1033/2894.htm",
        "http://www.phys.ruc.edu.cn/info/1550/2895.htm",
    ),
    13: (
        "http://psy.ruc.edu.cn/",
        "http://psy.ruc.edu.cn/yxgk2/index.htm",
        "http://psy.ruc.edu.cn/yxgk2/xkfx1/index.htm",
    ),
    15: (
        "http://psy.ruc.edu.cn/shfw1/kcpxb/eb6dccf76fa3457e95af32ab9616cf39.htm",
        "http://psy.ruc.edu.cn/shfw1/kcpxb/8bd07b1db903468ab36d79832439f704.htm",
        "http://psy.ruc.edu.cn/shfw1/kcpxb/a38f62902b6f4ca1949eba40b97d0163.htm",
    ),
    16: ("http://psy.ruc.edu.cn/szdw2/index.htm",),
    18: (
        "http://psy.ruc.edu.cn/yxgk2/xkfx1/index.htm",
        "http://psy.ruc.edu.cn/rcpy/index.htm",
        "http://psy.ruc.edu.cn/rcpy/bs/70cd2792c93c47fabc0846fcbf3e3d90.htm",
    ),
    19: (
        "http://psy.ruc.edu.cn/zszp/index.htm",
        "http://psy.ruc.edu.cn/zszp/sbzs/index.htm",
        "http://psy.ruc.edu.cn/zszp/sbzs/4387e0b25c964e7c8ef49c0e161c269d.htm",
    ),
    21: ("http://guoxue.ruc.edu.cn/",),
    27: ("http://guoxue.ruc.edu.cn/rcpy/kc/index.htm",),
    28: (
        "http://guoxue.ruc.edu.cn/szdw/hygdxx/8dc8a456b99c428c9997b38282bf28e2.htm",
        "http://guoxue.ruc.edu.cn/szdw/hygdxx/ef8559535f7a489d81c990232c9628c6.htm",
        "http://guoxue.ruc.edu.cn/szdw/hygdxx/49f6e443d4714e24978091e0b0cdf6d8.htm",
    ),
    29: ("http://guoxue.ruc.edu.cn/rcpy/kc/index.htm",),
    30: (
        "http://guoxue.ruc.edu.cn/rcpy/zs/index.htm",
        "http://guoxue.ruc.edu.cn/rcpy/zs/49e340beb7af430e99599406e5b46acf.htm",
        "http://guoxue.ruc.edu.cn/rcpy/zs/a329665a40a54880b0f28e69d072625a.htm",
    ),
    31: (
        "http://guoxue.ruc.edu.cn/wzsy/xsjz/index1.htm",
        "http://guoxue.ruc.edu.cn/wzsy/xsjz/197f8e7589c94f339af09e4f2c82a997.htm",
        "http://guoxue.ruc.edu.cn/wzsy/xsjz/a074ea7acfb941a4be99138ec1a4e908.htm",
    ),
    32: (
        "http://guoxue.ruc.edu.cn/wzsy/tzgg/cdd3857faaa44bb0bd0680aaf7cf6ea4.htm",
        "http://guoxue.ruc.edu.cn/wzsy/tzgg/2bbb520328d64ca691e116beae0189df.htm",
        "http://guoxue.ruc.edu.cn/wzsy/tzgg/914deb247d7c456291d206181737c619.htm",
    ),
    33: ("http://guoxue.ruc.edu.cn/xygk/xzks/index.htm",),
    34: (
        "http://guoxue.ruc.edu.cn/ky/xsxm/index.htm",
        "http://guoxue.ruc.edu.cn/ky/jsxm/index.htm",
        "http://guoxue.ruc.edu.cn/ky/kygz/73e57ed1423d43fe8c356c92f382e01e.htm",
    ),
    37: ("https://envi.ruc.edu.cn/xygk/xyjj/index.htm",),
    40: (
        "https://envi.ruc.edu.cn/jszy/pj/3966fa40ecdd41beb2e7cec9c11cda3f.htm",
        "https://envi.ruc.edu.cn/jszy/lh/78f0e86862ff45aa8888084abdb9f7fe.htm",
        "https://envi.ruc.edu.cn/jszy/wk/b4f0792e601a4865ae69645f8c2b959f.htm",
    ),
    41: (
        "https://envi.ruc.edu.cn/kxyj/kycg/index.htm",
        "https://envi.ruc.edu.cn/kxyj/kycg/4d927d4c80704d17b756cf8cf4575fb0.htm",
        "https://envi.ruc.edu.cn/kxyj/kycg/90a51347ba1c46d4bb297cbede212179.htm",
    ),
    42: ("https://envi.ruc.edu.cn/kxyj/kyxm/158a469e36c6408d99ac1bc96d849ac5.htm",),
    45: (
        "https://envi.ruc.edu.cn/tzgg/c4409f7103cd429384e7ebb5beb4e423.htm",
        "https://envi.ruc.edu.cn/tzgg/5e5dad8a138948298156976c62984fbb.htm",
        "https://envi.ruc.edu.cn/tzgg/d5bd3e4172d24918baaf8919a51b2421.htm",
    ),
    49: ("https://envi.ruc.edu.cn/tzgg/9dbe4248d07d4373a72712137dcd282a.htm",),
    50: (
        "https://envi.ruc.edu.cn/jszy/pj/3966fa40ecdd41beb2e7cec9c11cda3f.htm",
        "https://envi.ruc.edu.cn/jszy/lh/78f0e86862ff45aa8888084abdb9f7fe.htm",
        "https://envi.ruc.edu.cn/jszy/wk/b4f0792e601a4865ae69645f8c2b959f.htm",
    ),
    51: ("http://ai.ruc.edu.cn/",),
    52: ("http://ai.ruc.edu.cn/newslist/lecture/20250702007.html",),
    53: (
        "http://ai.ruc.edu.cn/overview/intro/index.htm",
        "http://ai.ruc.edu.cn/newslist/newsdetail/20200304001.html",
    ),
    54: ("http://ai.ruc.edu.cn/academicfaculty/index.htm",),
    55: (
        "http://ai.ruc.edu.cn/newslist/lecture/20250703003.html",
        "http://ai.ruc.edu.cn/newslist/lecture/20250619002.html",
    ),
    56: (
        "http://ai.ruc.edu.cn/academicfaculty/rczp/index.htm",
        "http://ai.ruc.edu.cn/academicfaculty/rczp/20250516001.html",
        "http://ai.ruc.edu.cn/academicfaculty/rczp/20250516002.html",
    ),
    57: (
        "http://ai.ruc.edu.cn/english/gsaiabout/introduction/index.htm",
        "http://ai.ruc.edu.cn/english/index.htm",
        "http://ai.ruc.edu.cn/english/academic/index.htm",
    ),
    58: (
        "http://ai.ruc.edu.cn/research/science/0351c1fb247f4343ad28ce4290edec94.htm",
        "http://ai.ruc.edu.cn/research/science/0e6b1a8e44164413a878842d481f454c.htm",
        "http://ai.ruc.edu.cn/research/science/20210414.html",
    ),
    60: (
        "http://ai.ruc.edu.cn/newslist/jcsp/20190927002.html",
        "http://ai.ruc.edu.cn/newslist/jcsp/20230615100.html",
        "http://ai.ruc.edu.cn/newslist/jcsp/7c13a9addaee424dbf68ce5d5b2fdf78.htm",
    ),
    61: (
        "http://ai.ruc.edu.cn/student/undergraduate/index.htm",
        "http://ai.ruc.edu.cn/student/master/index.htm",
        "http://ai.ruc.edu.cn/student/doctor/index.htm",
    ),
    63: (
        "http://ai.ruc.edu.cn/newslist/newsdetail/20250824002.html",
        "http://ai.ruc.edu.cn/newslist/newsdetail/20190119001.html",
        "http://ai.ruc.edu.cn/newslist/newsdetail/20210420001.html",
    ),
    64: (
        "http://ai.ruc.edu.cn/newslist/newsdetail/20260401004.html",
        "http://ai.ruc.edu.cn/newslist/notice/20260410100.html",
    ),
    65: (
        "http://ai.ruc.edu.cn/newslist/lecture/20251219002.html",
        "http://ai.ruc.edu.cn/newslist/lecture/20260326003.html",
        "http://ai.ruc.edu.cn/newslist/newsdetail/20260401001.html",
    ),
    66: (
        "http://ai.ruc.edu.cn/newslist/lecture/20251223008.html",
        "http://ai.ruc.edu.cn/newslist/lecture/20250319003.html",
        "http://ai.ruc.edu.cn/newslist/lecture/20250623003.html",
    ),
    67: (
        "http://ai.ruc.edu.cn/newslist/lecture/20250618004.html",
        "http://ai.ruc.edu.cn/newslist/lecture/20251223008.html",
        "http://ai.ruc.edu.cn/research/science/20260516001.html",
    ),
    68: ("http://ai.ruc.edu.cn/newslist/lecture/20261223009.html",),
    69: (
        "http://ai.ruc.edu.cn/newslist/lecture/index.htm",
        "http://ai.ruc.edu.cn/newslist/lecture/20240521100.html",
        "http://ai.ruc.edu.cn/newslist/lecture/20250619001.html",
    ),
    70: (
        "http://ai.ruc.edu.cn/student/undergraduate/index.htm",
        "http://ai.ruc.edu.cn/overview/intro/index.htm",
    ),
    71: (
        "http://ai.ruc.edu.cn/newslist/newsdetail/20260601100.html",
        "http://ai.ruc.edu.cn/newslist/newsdetail/20260612003.html",
    ),
    72: (
        "http://ai.ruc.edu.cn/newslist/newsdetail/20230703003.html",
        "http://ai.ruc.edu.cn/newslist/jcsp/20230615100.html",
        "http://ai.ruc.edu.cn/newslist/newsdetail/20240613004.html",
    ),
    73: (
        "http://ai.ruc.edu.cn/mag/202001/zxsl/jydt/gngx/e3d4fa14aa234ee1bcc0828e70ce11aa.htm",
        "http://ai.ruc.edu.cn/newslist/newsdetail/20260612003.html",
        "http://ai.ruc.edu.cn/newslist/newsdetail/20260714101.html",
    ),
    74: ("http://ai.ruc.edu.cn/newslist/newsdetail/20260714100.html",),
    75: ("http://ai.ruc.edu.cn/newslist/newsdetail/20260714103.html",),
    77: ("http://ai.ruc.edu.cn/newslist/newsdetail/20260722100.html",),
    78: ("http://ai.ruc.edu.cn/newslist/newsdetail/20260724.html",),
    79: ("http://ai.ruc.edu.cn/newslist/newsdetail/20260727100.html",),
    80: (
        "http://ai.ruc.edu.cn/newslist/newsdetail/20260825100.html",
        "http://ai.ruc.edu.cn/newslist/newsdetail/120721216d0146d5953b9206a612ae6f.htm",
    ),
    81: ("http://ai.ruc.edu.cn/newslist/newsdetail/index.htm",),
    82: (
        "http://ai.ruc.edu.cn/student/master/index.htm",
        "http://ai.ruc.edu.cn/newslist/notice/20260318100.html",
        "http://ai.ruc.edu.cn/newslist/notice/20260228001.html",
    ),
    83: ("http://ai.ruc.edu.cn/newslist/notice/20260112100.html",),
    84: ("http://ai.ruc.edu.cn/newslist/notice/20260306101.html",),
    85: (
        "http://ai.ruc.edu.cn/newslist/notice/20260318100.html",
        "http://ai.ruc.edu.cn/newslist/notice/20240320100.html",
        "http://ai.ruc.edu.cn/newslist/notice/20210319001.html",
    ),
    89: (
        "https://gsai.ruc.edu.cn/addons/teacher/index/info.html?user_id=30&ln=cn",
        "https://gsai.ruc.edu.cn/yankailin",
    ),
    91: (
        "https://gsai.ruc.edu.cn/XiaoZHOU",
        "https://gsai.ruc.edu.cn/addons/teacher/index/info.html?user_id=41&ln=cn",
    ),
    93: (
        "https://gsai.ruc.edu.cn/bingsu",
        "https://gsai.ruc.edu.cn/addons/teacher/index/info.html?user_id=40&ln=cn",
    ),
    94: ("https://gsai.ruc.edu.cn/caozhao",),
    95: (
        "https://gsai.ruc.edu.cn/chenxu",
        "https://gsai.ruc.edu.cn/addons/teacher/index/info.html?user_id=19&ln=cn",
    ),
    96: (
        "https://gsai.ruc.edu.cn/chongxuan",
        "https://gsai.ruc.edu.cn/addons/teacher/index/info.html?user_id=24&ln=cn",
    ),
    98: ("https://gsai.ruc.edu.cn/english/XiaoZHOU",),
    100: ("https://gsai.ruc.edu.cn/english/bingsu",),
}


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    output = []
    for line_number, urls in SELECTED.items():
        assert 1 <= len(urls) <= 3
        source = rows[line_number - 1]
        missing = set(urls) - set(source["target_urls"])
        assert not missing, f"line {line_number} missing URLs: {sorted(missing)}"
        output.append({"query": source["query"], "target_urls": list(urls)})

    assert len({row["query"] for row in output}) == len(output)
    TARGET.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output)
    )
    print(f"wrote {len(output)} queries and {sum(len(r['target_urls']) for r in output)} URLs to {TARGET}")


if __name__ == "__main__":
    main()

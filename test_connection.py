import sqlite3
from pathlib import Path

Path('data').mkdir(exist_ok=True)

db = sqlite3.connect("data/zeta.db")

db.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    url TEXT UNIQUE,
    title TEXT,
    content TEXT
)
""")

from zeta_engine import storage
from zeta_engine.storage import Document

document = [
    Document(
        url="https://news.ruc.edu.cn/2086967844864720898.html", 
        title="倒计时10天！通州校区想和你提前见一面！", 
        text="盛夏八月，万物竞秀。中国人民大学双主校区生机勃发，待启新章，10天后，3000余名2026级本科新生，8月底，6000余名2026级研究生新生，即将来校报到", 
        fetched_at="2026-08-24T10:00:00"
    ), 
    Document(
        url="https://news.ruc.edu.cn/2084246331740848129.html", 
        title="中国人民大学“重走光辉校史路”实践团赴延安开展实践活动",
        text="7月20日至24日，中国人民大学“重走光辉校史路”实践团赴延安，参加延河高校人才培养联盟“到延安去——重走来时路·奋进新征程”青年红色筑梦实践活动。此次实践既是一场回望历史、寻根铸魂的红色研学，也是一堂深入基层、感知时代的实践课程。中国人民大学青年学子将弘扬伟大延安精神，以严谨求实的学术态度、深入基层的实践作风和服务人民的青春担当，用脚步丈量祖国大地，用眼睛发现中国精神，用耳朵倾听人民呼声，用内心感应时代脉搏，把个人理想融入党和国家事业，在强国建设、民族复兴的新征程上书写挺膺担当的青春答卷。", 
        fetched_at="2026-0824T10:00:25"
    ), 
    Document(
        url="https://news.ruc.edu.cn/2079483069547610113.html", 
        title="中国人民大学青年学者李崇轩获颁2026WAIC云帆奖", 
        text="7月17日至20日，2026世界人工智能大会暨人工智能全球治理高级别会议在上海举行。中华人民共和国主席习近平出席大会开幕式，并发表题为《携手构建公正合理的全球人工智能治理体系》的主旨讲话，为促进全球人工智能发展注入新动力。7月17日至20日，2026世界人工智能大会暨人工智能全球治理高级别会议在上海举行。中华人民共和国主席习近平出席大会开幕式，并发表题为《携手构建公正合理的全球人工智能治理体系》的主旨讲话，为促进全球人工智能发展注入新动力。此次获评云帆奖“璀璨明星”，既是对李崇轩科研创新能力的高度认可，也展现了中国人民大学在人工智能领域青年人才培养和前沿科研探索方面取得的积极成效。",
        fetched_at="2026-0824T10:00:54"
    )
]

db.commit()


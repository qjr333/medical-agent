import os
import re
import json
import hashlib
import tempfile
import numpy as np

from collections import Counter
from functools import wraps
from threading import RLock


# =========================================================
# 1. Sentence Transformers
# =========================================================

# Import the optional, heavyweight dependency only when a query needs it.
SentenceTransformer = None


# =========================================================
# 2. 项目路径
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


KNOWLEDGE_DIR = os.path.join(
    BASE_DIR,
    "knowledge"
)


META_DIR = os.path.join(
    BASE_DIR,
    "knowledge_meta"
)


SOURCE_REGISTRY_PATH = os.path.join(
    META_DIR,
    "source_registry.json"
)


# =========================================================
# 3. Embedding配置
# =========================================================

EMBEDDING_MODEL_NAME = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "BAAI/bge-small-zh-v1.5"
)


MIN_SEMANTIC_SCORE = float(
    os.getenv(
        "RAG_MIN_SCORE",
        "0.20"
    )
)


DISABLE_EMBEDDING = (
    os.getenv(
        "RAG_DISABLE_EMBEDDING",
        "0"
    )
    ==
    "1"
)


RAG_CACHE_DIR = os.getenv(
    "RAG_CACHE_DIR",
    os.path.join(BASE_DIR, "data", "rag_cache")
)

INDEX_CACHE_PATH = os.path.join(RAG_CACHE_DIR, "knowledge_embeddings.npz")
CACHE_SCHEMA_VERSION = 1


# =========================================================
# 4. 全局状态
# =========================================================

embedding_model = None

embedding_available = False

embedding_error = None


source_registry = {}

knowledge_blocks = []

block_embeddings = None

_index_lock = RLock()
_cache_metadata = None
_embedding_load_attempted = False
_embedding_model_name = None
_index_build_attempted = False


def _with_index_lock(function):
    """Keep lazy initialization and retrieval consistent with index reloads."""
    @wraps(function)
    def locked(*args, **kwargs):
        with _index_lock:
            return function(*args, **kwargs)
    return locked


# =========================================================
# 5. Topic关键词
# =========================================================

TOPIC_KEYWORDS = {

    # -----------------------------------------------------
    # 家庭血压
    # -----------------------------------------------------

    "home_bp_monitoring": [

        "家庭血压",
        "家庭自测",
        "血压计",
        "测血压",
        "量血压",
        "复测血压",
        "重新测血压",
        "袖带",
        "血压记录",
        "上传血压",
        "血压数据",
        "早晨血压",
        "晚上血压"
    ],


    # -----------------------------------------------------
    # 随访流程
    # -----------------------------------------------------

    "followup_workflow": [

        "随访",
        "回访",
        "主动随访",
        "慢病管理",
        "长期管理",
        "随访流程",
        "随访内容"
    ],


    # -----------------------------------------------------
    # 服药依从性
    # -----------------------------------------------------

    "medication_adherence": [

        "漏服",
        "漏吃",
        "忘记吃药",
        "忘记服药",
        "没吃药",
        "没有吃药",
        "服药不规律",
        "间断服药",
        "依从性",
        "规律服药",
        "不服药"
    ],


    # -----------------------------------------------------
    # 用药安全
    # -----------------------------------------------------

    "medication_safety": [

        "多吃一颗",
        "多吃一片",
        "补吃",
        "补服",
        "加药",
        "减药",
        "停药",
        "换药",
        "调整药量",
        "药物剂量",
        "副作用",
        "不良反应",
        "药物相互作用",
        "多重用药",
        "降压药"
    ],


    # -----------------------------------------------------
    # 老年特殊管理
    # -----------------------------------------------------

    "elderly_management": [

        "老年",
        "老人",
        "高龄",
        "衰弱",
        "认知",
        "记忆力",
        "跌倒",
        "体位性",
        "独居"
    ],


    # -----------------------------------------------------
    # 红旗症状 / 转诊
    # -----------------------------------------------------

    "red_flags_referral": [

        "急诊",
        "急救",
        "危险",
        "转诊",
        "胸痛",
        "胸口痛",
        "呼吸困难",
        "喘不上气",
        "晕厥",
        "晕倒",
        "意识不清",
        "肢体无力",
        "说话不清",
        "高血压急症"
    ],


    # -----------------------------------------------------
    # 就医建议
    # -----------------------------------------------------

    "visit_guidance": [

        "就医",
        "就诊",
        "去医院",
        "看医生",
        "挂什么科",
        "复诊",
        "复查",
        "就医准备"
    ],


    # -----------------------------------------------------
    # 生活方式
    # -----------------------------------------------------

    "lifestyle_management": [

        "饮食",
        "吃什么",
        "盐",
        "低盐",
        "运动",
        "锻炼",
        "体重",
        "减重",
        "肥胖",
        "饮酒",
        "喝酒",
        "吸烟",
        "戒烟",
        "睡眠",
        "营养"
    ]
}


# =========================================================
# 6. 自然语言正则模式
#
# 比普通关键词更重要
# =========================================================

TOPIC_PATTERNS = {

    # -----------------------------------------------------
    # 家庭血压
    # -----------------------------------------------------

    "home_bp_monitoring": [

        r"(在家|家庭).{0,10}(测|量).{0,6}血压",

        r"(怎么|如何|正确).{0,10}(测|量).{0,6}血压",

        r"(早上|早晨|晨起|晚上|睡前).{0,10}(测|量).{0,6}血压",

        r"血压.{0,8}(测几次|量几次|怎么测|怎么量)",

        r"(血压计|袖带).{0,10}(使用|选择|测量)",

        r"(复测|重新测).{0,8}血压"
    ],


    # -----------------------------------------------------
    # 随访
    # -----------------------------------------------------

    "followup_workflow": [

        r"(多久|什么时候|何时).{0,8}(随访|回访)",

        r"(主动|定期|长期).{0,8}(随访|管理)",

        r"随访.{0,10}(内容|流程|怎么做|做什么)"
    ],


    # -----------------------------------------------------
    # 服药依从性
    # -----------------------------------------------------

    "medication_adherence": [

        r"(忘记|忘了|漏了|漏服|漏吃|没|没有|未).{0,10}(吃|服).{0,10}(药|降压药)",

        r"(药|降压药).{0,10}(忘记|忘了|漏服|漏吃|没吃|未服)",

        r"(服药|吃药).{0,10}(不规律|间断|断断续续|依从性)",

        r"(经常|偶尔|有时).{0,10}(忘|漏).{0,10}(吃药|服药|降压药)",

        r"(自行|自己).{0,6}(停药|断药)"
    ],


    # -----------------------------------------------------
    # 用药安全
    # -----------------------------------------------------

    "medication_safety": [

        r"(能不能|可以不可以|要不要|是否).{0,12}(多吃|加服|补吃|补服|少吃|停药|减药|换药)",

        r"(多吃|加服|补吃|补服).{0,8}(一颗|一片|药)",

        r"(血压高|血压升高).{0,12}(加药|多吃|补吃|补服)",

        r"(头晕|不舒服|副作用).{0,12}(停药|减药|少吃)",

        r"(药物|降压药).{0,10}(副作用|不良反应|相互作用)"
    ],


    # -----------------------------------------------------
    # 老年
    # -----------------------------------------------------

    "elderly_management": [

        r"(老人|老年|高龄)",

        # 65-99岁
        r"(?:6[5-9]|[7-9][0-9])岁",

        # >=100岁
        r"[1-9][0-9]{2}岁",

        r"(站起来|起身|站立).{0,10}(头晕|眼黑|晕)",

        r"(衰弱|跌倒|认知|记忆力|独居)"
    ],


    # -----------------------------------------------------
    # 红旗
    # -----------------------------------------------------

    "red_flags_referral": [

        r"(胸痛|胸口痛|胸口特别痛|胸口很痛)",

        r"(呼吸困难|喘不上气|无法呼吸)",

        r"(晕厥|晕倒|晕过去)",

        r"(意识不清|意识异常|昏迷)",

        r"(肢体无力|手脚无力|偏瘫)",

        r"(说话不清|言语不清)",

        r"(急诊|急救|转诊)"
    ],


    # -----------------------------------------------------
    # 就医
    # -----------------------------------------------------

    "visit_guidance": [

        r"(要不要|是否|需要).{0,10}(去医院|就医|就诊|复诊)",

        r"(挂什么科|看什么科)",

        r"(什么时候|何时).{0,10}(去医院|看医生|复诊)",

        r"(去医院|就诊).{0,10}(准备|带什么)"
    ],


    # -----------------------------------------------------
    # 生活方式
    # -----------------------------------------------------

    "lifestyle_management": [

        r"高血压.{0,10}(吃什么|怎么吃|饮食)",

        r"(盐|钠).{0,10}(多少|摄入|限制|少吃)",

        r"(运动|锻炼).{0,10}(怎么|多久|多少|适合)",

        r"(减重|体重|肥胖)",

        r"(喝酒|饮酒|抽烟|吸烟|戒烟)"
    ]
}


# =========================================================
# 7. Pattern权重
#
# 目的：
# 老年身份不能覆盖真正的医学意图
# =========================================================

TOPIC_PATTERN_WEIGHTS = {

    "home_bp_monitoring":
        3.0,

    "followup_workflow":
        3.0,

    "medication_adherence":
        4.0,

    "medication_safety":
        5.0,

    "elderly_management":
        1.5,

    "red_flags_referral":
        8.0,

    "visit_guidance":
        4.0,

    "lifestyle_management":
        3.0
}


# =========================================================
# 8. Metadata值解析
# =========================================================

def parse_value(
    value: str
):

    value = value.strip()


    if not value:
        return None


    # bool

    if value.lower() == "true":
        return True


    if value.lower() == "false":
        return False


    # null

    if value.lower() in [
        "null",
        "none"
    ]:

        return None


    # JSON

    if (
        value.startswith("[")
        or
        value.startswith("{")
    ):

        try:

            return json.loads(
                value
            )

        except Exception:

            pass


    # number

    try:

        if "." in value:

            return float(
                value
            )


        return int(
            value
        )


    except Exception:

        pass


    # string

    return value.strip(
        "\"'"
    )


# =========================================================
# 9. 加载Source Registry
# =========================================================

def load_source_registry():

    if not os.path.exists(
        SOURCE_REGISTRY_PATH
    ):

        print(
            "[RAG] 未找到 source_registry.json"
        )

        return {}


    try:

        with open(
            SOURCE_REGISTRY_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )


        sources = data.get(
            "sources",
            []
        )


        registry = {}


        for source in sources:

            source_id = source.get(
                "source_id"
            )


            if source_id:

                registry[
                    source_id
                ] = source


        print(
            f"[RAG] Source Registry：{len(registry)} 个来源"
        )


        return registry


    except Exception as error:

        print(
            "[RAG] Source Registry读取失败"
        )


        return {}


# =========================================================
# 10. 解析Markdown知识块
# =========================================================

def parse_markdown_blocks(
    filepath: str,
    filename: str
):

    try:

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()


    except Exception as error:

        print(
            f"[RAG] 无法读取知识来源：{filename}"
        )

        return []


    # -----------------------------------------------------
    # 根据 block_id 找原子知识块
    # -----------------------------------------------------

    matches = list(
        re.finditer(
            r"(?m)^block_id\s*:\s*(.+?)\s*$",
            text
        )
    )


    blocks = []


    # -----------------------------------------------------
    # 如果是普通旧Markdown
    # -----------------------------------------------------

    if not matches:

        topic_name = (
            filename
            .replace(
                ".md",
                ""
            )
        )


        return [

            {

                "block_id":
                    filename,

                "source_file":
                    filename,

                "source_id":
                    None,

                "source_title":
                    filename,

                "year":
                    None,

                "source_pages":
                    None,

                "chapter":
                    None,

                "topic":
                    topic_name,

                "subtopic":
                    None,

                "population":
                    [],

                "authority":
                    None,

                "priority":
                    50,

                "safety_layer_candidate":
                    False,

                "content":
                    text.strip(),

                "search_text":
                    text.strip()
            }

        ]


    # -----------------------------------------------------
    # 解析所有block
    # -----------------------------------------------------

    for index, match in enumerate(
        matches
    ):

        start = match.start()


        if (
            index
            <
            len(matches) - 1
        ):

            end = (
                matches[
                    index + 1
                ].start()
            )


        else:

            end = len(
                text
            )


        segment = (
            text[
                start:end
            ]
            .strip()
        )


        lines = (
            segment
            .splitlines()
        )


        metadata = {}


        content_start = None


        # -------------------------------------------------
        # Metadata
        # -------------------------------------------------

        for line_index, line in enumerate(
            lines
        ):

            stripped = (
                line
                .strip()
            )


            # Markdown正文标题开始

            if stripped.startswith(
                "#"
            ):

                content_start = (
                    line_index
                )

                break


            if stripped in [
                "",
                "---"
            ]:

                continue


            if ":" in stripped:

                key, value = (
                    stripped.split(
                        ":",
                        1
                    )
                )


                metadata[
                    key.strip()
                ] = parse_value(
                    value
                )


        # -------------------------------------------------
        # Content
        # -------------------------------------------------

        if (
            content_start
            is not None
        ):

            content = "\n".join(
                lines[
                    content_start:
                ]
            ).strip()


        else:

            content_lines = []

            metadata_finished = False


            for line in lines:

                stripped = line.strip()


                if metadata_finished:

                    content_lines.append(
                        line
                    )

                    continue


                if stripped in [
                    "",
                    "---"
                ]:

                    continue


                if ":" not in stripped:

                    metadata_finished = True

                    content_lines.append(
                        line
                    )


            content = "\n".join(
                content_lines
            ).strip()


        # -------------------------------------------------
        # 基础字段
        # -------------------------------------------------

        block_id = (
            metadata.get(
                "block_id"
            )
            or
            match.group(1)
            .strip()
        )


        source_id = (
            metadata.get(
                "source_id"
            )
        )


        registry_source = (
            source_registry.get(
                source_id,
                {}
            )
        )


        source_title = (

            metadata.get(
                "source_title"
            )

            or

            registry_source.get(
                "title"
            )

            or

            filename
        )


        year = (

            metadata.get(
                "year"
            )

            or

            registry_source.get(
                "year"
            )
        )


        priority = (

            metadata.get(
                "priority"
            )

            or

            registry_source.get(
                "base_priority"
            )

            or

            50
        )


        population = (

            metadata.get(
                "population"
            )

            or

            registry_source.get(
                "population"
            )

            or

            []
        )


        if isinstance(
            population,
            str
        ):

            population = [
                population
            ]


        topic = metadata.get(
            "topic"
        )


        subtopic = metadata.get(
            "subtopic"
        )


        block = {

            "block_id":
                block_id,

            "source_file":
                filename,

            "source_id":
                source_id,

            "source_title":
                source_title,

            "year":
                year,

            "source_pages":
                metadata.get(
                    "source_pages"
                ),

            "chapter":
                metadata.get(
                    "chapter"
                ),

            "topic":
                topic,

            "subtopic":
                subtopic,

            "population":
                population,

            "authority":
                (
                    metadata.get(
                        "authority"
                    )
                    or
                    registry_source.get(
                        "document_type"
                    )
                ),

            "priority":
                priority,

            "safety_layer_candidate":
                bool(
                    metadata.get(
                        "safety_layer_candidate",
                        False
                    )
                ),

            "content":
                content
        }


        # -------------------------------------------------
        # 给Embedding检索的文本
        # -------------------------------------------------

        block[
            "search_text"
        ] = (
            f"主题：{topic}\n"
            f"子主题：{subtopic}\n"
            f"适用人群：{population}\n"
            f"指南来源：{source_title}\n"
            f"年份：{year}\n\n"
            f"{content}"
        )


        blocks.append(
            block
        )


    return blocks


# =========================================================
# 11. 加载全部知识文件
# =========================================================

def load_knowledge_base():

    os.makedirs(
        KNOWLEDGE_DIR,
        exist_ok=True
    )


    all_blocks = []


    markdown_files = []


    for filename in sorted(
        os.listdir(
            KNOWLEDGE_DIR
        )
    ):

        if not filename.lower().endswith(
            ".md"
        ):

            continue


        markdown_files.append(
            filename
        )


        filepath = os.path.join(
            KNOWLEDGE_DIR,
            filename
        )


        blocks = (
            parse_markdown_blocks(
                filepath,
                filename
            )
        )


        all_blocks.extend(
            blocks
        )


    print(
        f"[RAG] Markdown文件：{len(markdown_files)} 个来源"
    )


    print(
        f"[RAG] 共解析 {len(all_blocks)} 个原子知识块"
    )


    return all_blocks


# =========================================================
# 12. Embedding模型
# =========================================================

@_with_index_lock
def load_embedding_model():
    global SentenceTransformer
    global embedding_model, embedding_available, embedding_error
    global _embedding_load_attempted, _embedding_model_name

    if DISABLE_EMBEDDING:
        embedding_available = False
        embedding_error = "RAG_DISABLE_EMBEDDING=1"
        return

    # A changed model configuration must never reuse an old model instance.
    if _embedding_model_name != EMBEDDING_MODEL_NAME:
        embedding_model = None
        embedding_available = False
        embedding_error = None
        _embedding_load_attempted = False
        _embedding_model_name = EMBEDDING_MODEL_NAME

    if embedding_error == "RAG_DISABLE_EMBEDDING=1":
        embedding_available = embedding_model is not None
        embedding_error = None
        if embedding_model is None:
            _embedding_load_attempted = False

    if _embedding_load_attempted:
        return

    _embedding_load_attempted = True
    try:
        if SentenceTransformer is None:
            from sentence_transformers import SentenceTransformer as model_class
            SentenceTransformer = model_class

        print("[RAG] 正在加载Embedding模型")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embedding_available = True
        embedding_error = None
        print("[RAG] Embedding模型加载完成")
    except Exception as error:
        embedding_available = False
        embedding_error = str(error)
        print("[RAG] Embedding模型加载失败，使用关键词检索")


# =========================================================
# 13. Document Embedding
# =========================================================

def encode_documents(
    texts
):

    if hasattr(
        embedding_model,
        "encode_document"
    ):

        return embedding_model.encode_document(

            texts,

            normalize_embeddings=True
        )


    return embedding_model.encode(

        texts,

        normalize_embeddings=True
    )


# =========================================================
# 14. Query Embedding
# =========================================================

def encode_query(
    query
):

    if hasattr(
        embedding_model,
        "encode_query"
    ):

        result = embedding_model.encode_query(

            [query],

            normalize_embeddings=True
        )


    else:

        result = embedding_model.encode(

            [query],

            normalize_embeddings=True
        )


    return result[0]


# =========================================================
# 15. 关键词检索Fallback
# =========================================================

def normalize_text(
    text: str
):

    text = (
        text
        .lower()
    )


    text = re.sub(
        r"\s+",
        "",
        text
    )


    text = re.sub(
        r"[^\w\u4e00-\u9fff]",
        "",
        text
    )


    return text


def make_ngrams(
    text: str,
    n: int = 2
):

    text = normalize_text(
        text
    )


    if not text:

        return set()


    if len(
        text
    ) <= n:

        return {
            text
        }


    return {

        text[
            index:
            index + n
        ]

        for index
        in range(
            len(text) - n + 1
        )
    }


def lexical_score(
    query: str,
    document: str
):

    query_tokens = (
        make_ngrams(
            query
        )
    )


    document_tokens = (
        make_ngrams(
            document
        )
    )


    if (
        not query_tokens
        or
        not document_tokens
    ):

        return 0.0


    intersection = len(

        query_tokens
        &
        document_tokens
    )


    denominator = (

        len(
            query_tokens
        )

        *

        len(
            document_tokens
        )

    ) ** 0.5


    if denominator == 0:

        return 0.0


    return (
        intersection
        /
        denominator
    )


# =========================================================
# 16. Topic Router
# =========================================================

def detect_topics(
    query: str,
    patient_state: dict = None,
    topic_hint: str = None
):

    scores = Counter()


    query_lower = (
        query
        .lower()
        .strip()
    )


    # -----------------------------------------------------
    # Action Selector明确给出Topic
    # -----------------------------------------------------

    if topic_hint:

        scores[
            topic_hint
        ] += 10


    # -----------------------------------------------------
    # 普通关键词
    # -----------------------------------------------------

    for topic, keywords in (
        TOPIC_KEYWORDS.items()
    ):

        for keyword in keywords:

            if (
                keyword.lower()
                in query_lower
            ):

                scores[
                    topic
                ] += 1


    # -----------------------------------------------------
    # 正则语义模式
    # -----------------------------------------------------

    for topic, patterns in (
        TOPIC_PATTERNS.items()
    ):

        weight = (
            TOPIC_PATTERN_WEIGHTS.get(
                topic,
                3.0
            )
        )


        for pattern in patterns:

            if re.search(
                pattern,
                query_lower
            ):

                scores[
                    topic
                ] += weight


    # -----------------------------------------------------
    # Patient State
    # -----------------------------------------------------

    if patient_state:

        age = patient_state.get(
            "age"
        )


        # 年龄只轻微影响，
        # 不允许覆盖主医学意图

        if (
            isinstance(
                age,
                int
            )
            and
            age >= 65
        ):

            scores[
                "elderly_management"
            ] += 1


        medication_status = (
            patient_state.get(
                "medication_status"
            )
        )


        if medication_status in [
            "missed",
            "irregular"
        ]:

            scores[
                "medication_adherence"
            ] += 3


        red_flags = (
            patient_state.get(
                "red_flags",
                []
            )
        )


        if red_flags:

            # Safety优先
            scores[
                "red_flags_referral"
            ] += 20


    if not scores:

        return []


    ranked = sorted(

        scores.items(),

        key=lambda item:
            item[1],

        reverse=True
    )


    routed_topics = [

        topic

        for topic, score
        in ranked[:4]

        if score > 0
    ]


    return routed_topics


# =========================================================
# 17. 人群匹配Bonus
# =========================================================

def population_match_bonus(
    block: dict,
    patient_state: dict = None
):

    if not patient_state:

        return 0.0


    bonus = 0.0


    age = patient_state.get(
        "age"
    )


    population = block.get(
        "population",
        []
    )


    if isinstance(
        population,
        str
    ):

        population = [
            population
        ]


    source_id = block.get(
        "source_id"
    )


    source = (
        source_registry.get(
            source_id,
            {}
        )
    )


    # -----------------------------------------------------
    # 老年患者
    # -----------------------------------------------------

    if (
        isinstance(
            age,
            int
        )
        and
        age >= 65
    ):

        elderly_markers = [

            "elderly",
            "age>=65",
            "老年",
            "老年人"
        ]


        if any(
            marker
            in population
            for marker
            in elderly_markers
        ):

            bonus += 0.10


        population_boost = (
            source.get(
                "population_boost",
                {}
            )
        )


        if age >= 80:

            raw_bonus = (
                population_boost.get(
                    "age>=80",
                    0
                )
            )


        else:

            raw_bonus = (
                population_boost.get(
                    "age>=65",
                    0
                )
            )


        try:

            bonus += min(
                float(
                    raw_bonus
                )
                /
                200,

                0.10
            )


        except Exception:

            pass


    return bonus


# =========================================================
# 18. Topic匹配Bonus
# =========================================================

def topic_match_bonus(
    block: dict,
    routed_topics: list
):

    if not routed_topics:

        return 0.0


    block_topic = (
        block.get(
            "topic"
        )
    )


    bonus = 0.0


    # -----------------------------------------------------
    # 主主题
    # -----------------------------------------------------

    if (
        block_topic
        in routed_topics
    ):

        position = (
            routed_topics.index(
                block_topic
            )
        )


        if position == 0:

            bonus += 0.18


        elif position == 1:

            bonus += 0.12


        elif position == 2:

            bonus += 0.08


        else:

            bonus += 0.05


    # -----------------------------------------------------
    # Source Registry Topic Boost
    # -----------------------------------------------------

    source = (
        source_registry.get(
            block.get(
                "source_id"
            ),
            {}
        )
    )


    topic_boost = (
        source.get(
            "topic_boost",
            {}
        )
    )


    for routed_topic in routed_topics:

        raw_bonus = (
            topic_boost.get(
                routed_topic,
                0
            )
        )


        try:

            bonus += min(

                float(
                    raw_bonus
                )
                /
                300,

                0.06
            )


        except Exception:

            pass


    return bonus


# =========================================================
# 19. 来源权威性Bonus
# =========================================================

def source_priority_bonus(
    block: dict
):

    priority = (
        block.get(
            "priority",
            50
        )
    )


    try:

        priority = float(
            priority
        )


    except Exception:

        priority = 50


    # priority 70以下几乎不加
    # priority 100约 +0.06

    bonus = (

        max(
            0,
            priority - 70
        )

        /
        30

        *
        0.06
    )


    return min(
        bonus,
        0.06
    )


# =========================================================
# 20. 重建索引
# =========================================================

def _file_digest(filepath):
    try:
        with open(filepath, "rb") as file:
            return hashlib.sha256(file.read()).hexdigest()
    except OSError as error:
        # Missing/unreadable files cannot match a previously readable snapshot.
        return "unreadable:" + type(error).__name__


def _make_cache_metadata():
    filenames = sorted(os.listdir(KNOWLEDGE_DIR)) if os.path.isdir(KNOWLEDGE_DIR) else []
    files = [
        [filename, _file_digest(os.path.join(KNOWLEDGE_DIR, filename))]
        for filename in filenames
        if filename.lower().endswith(".md")
    ]
    configuration = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "normalize_embeddings": True,
    }
    snapshot = {
        **configuration,
        "files": files,
        "source_registry": _file_digest(SOURCE_REGISTRY_PATH),
        # Include the parsed text/order to avoid mismatches if files change
        # between parsing and hashing, or parsing rules change later.
        "blocks": [
            [block["block_id"], block["source_file"], block["search_text"]]
            for block in knowledge_blocks
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {**configuration, "fingerprint": fingerprint,
            "block_count": len(knowledge_blocks)}


def _valid_embeddings(embeddings):
    if (embeddings.ndim != 2 or embeddings.shape[0] != len(knowledge_blocks)
            or embeddings.shape[1] <= 0 or embeddings.dtype != np.float32
            or not np.isfinite(embeddings).all()):
        return False
    # Retrieval uses the dot product of normalized document/query vectors.
    return bool(np.allclose(np.linalg.norm(embeddings, axis=1), 1.0,
                            rtol=1e-3, atol=1e-3))


def _read_index_cache():
    try:
        with np.load(INDEX_CACHE_PATH, allow_pickle=False) as cache:
            metadata_array = cache["metadata"]
            if metadata_array.ndim != 0 or metadata_array.dtype.kind != "U":
                return None
            metadata = json.loads(metadata_array.item())
            embeddings = cache["embeddings"]
            if not _valid_embeddings(embeddings):
                return None
            expected = {**_cache_metadata,
                        "embedding_dimension": embeddings.shape[1]}
            if metadata != expected:
                return None
            print(f"[RAG] 已读取磁盘向量索引：{len(knowledge_blocks)} 个知识块")
            return embeddings
    except FileNotFoundError:
        return None
    except Exception as error:
        # A stale/corrupt/unreadable cache is a cache miss, never an API error.
        print("[RAG] 向量缓存不可用，首次检索时重新构建")
        return None


def _write_index_cache(embeddings):
    temporary_path = None
    try:
        cache_dir = os.path.dirname(INDEX_CACHE_PATH) or "."
        os.makedirs(cache_dir, exist_ok=True)
        metadata = {**_cache_metadata,
                    "embedding_dimension": embeddings.shape[1]}
        with tempfile.NamedTemporaryFile(mode="wb", dir=cache_dir,
                                         prefix=".rag-index-", delete=False) as file:
            temporary_path = file.name
            np.savez_compressed(
                file, embeddings=embeddings,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True))
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, INDEX_CACHE_PATH)
        temporary_path = None
        print(f"[RAG] 知识块向量已缓存：{len(embeddings)} 个知识块")
    except Exception as error:
        # The in-memory index still works when the cache directory is unwritable.
        print("[RAG] 向量缓存写入失败，保留内存索引")
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


@_with_index_lock
def rebuild_index():
    """Reload source files and reuse matching vectors without loading a model."""
    global source_registry, knowledge_blocks, block_embeddings
    global _cache_metadata, _index_build_attempted
    global embedding_model, embedding_available, embedding_error
    global _embedding_model_name, _embedding_load_attempted

    if (_embedding_model_name is not None
            and _embedding_model_name != EMBEDDING_MODEL_NAME):
        embedding_model = None
        embedding_available = False
        embedding_error = None
        _embedding_model_name = None
        _embedding_load_attempted = False

    # An explicit reload may recover from a transient import/inference failure.
    # Keep a healthy, already loaded model; otherwise retry lazily on next query.
    if not DISABLE_EMBEDDING and embedding_error is not None:
        embedding_available = embedding_model is not None
        embedding_error = None
        _embedding_load_attempted = embedding_model is not None
    elif DISABLE_EMBEDDING:
        embedding_available = False
        embedding_error = "RAG_DISABLE_EMBEDDING=1"

    source_registry = load_source_registry()
    knowledge_blocks = load_knowledge_base()
    _cache_metadata = _make_cache_metadata()
    _index_build_attempted = False
    block_embeddings = None
    if not knowledge_blocks:
        print("[RAG] 当前知识库没有知识块")
    elif DISABLE_EMBEDDING:
        print("[RAG] 当前使用关键词检索模式")
    else:
        block_embeddings = _read_index_cache()
        if block_embeddings is None:
            print("[RAG] 无有效向量缓存，首次检索时建立索引")


def _ensure_embedding_index():
    global block_embeddings, embedding_available, embedding_error
    global _cache_metadata, _index_build_attempted

    if DISABLE_EMBEDDING:
        return False

    # Configuration changes during an explicit reload need the same invalidation
    # as a fresh process, before attempting to use cached or in-memory vectors.
    if (_cache_metadata is None
            or _cache_metadata["embedding_model"] != EMBEDDING_MODEL_NAME):
        _cache_metadata = _make_cache_metadata()
        _index_build_attempted = False
        block_embeddings = _read_index_cache()

    load_embedding_model()
    if not embedding_available:
        return False

    try:
        if block_embeddings is not None:
            dimension_getter = getattr(embedding_model,
                                       "get_sentence_embedding_dimension", None)
            dimension = dimension_getter() if dimension_getter else None
            if dimension is None or dimension == block_embeddings.shape[1]:
                return True
            # A dimension mismatch is a damaged cache; rebuild it once.
            block_embeddings = None

        if _index_build_attempted:
            return False
        _index_build_attempted = True
        print("[RAG] 正在生成知识块向量...")
        embeddings = np.asarray(encode_documents(
            [block["search_text"] for block in knowledge_blocks]
        ), dtype=np.float32)
        if not _valid_embeddings(embeddings):
            raise ValueError("Embedding模型返回无效的知识块向量")
        block_embeddings = embeddings
        _write_index_cache(embeddings)
        print(f"[RAG] Metadata-aware索引完成，知识块数量：{len(knowledge_blocks)}")
        return True
    except Exception as error:
        embedding_available = False
        embedding_error = str(error)
        print("[RAG] 知识块向量生成失败，使用关键词检索")
        return False


# =========================================================
# 21. Metadata-aware Retrieval
# =========================================================

@_with_index_lock
def retrieve_medical_guidance(
    query: str,
    patient_state: dict = None,
    topic_hint: str = None,
    top_k: int = 4,
    max_per_source: int = 2
):

    query = (
        query
        or ""
    ).strip()


    if not query:

        return []


    if not knowledge_blocks:

        return []


    # =====================================================
    # A. Topic Router
    # =====================================================

    routed_topics = (
        detect_topics(

            query,

            patient_state=
                patient_state,

            topic_hint=
                topic_hint
        )
    )


    print(
        "\n[RAG] =============================="
    )

    # Patient queries and model exception details may contain personal data.
    # Public diagnostics report counts and knowledge source identifiers only.
    print(f"[RAG] Routed Topics：{len(routed_topics)} 个主题")


    # =====================================================
    # B. 语义检索
    # =====================================================

    using_embeddings = _ensure_embedding_index()
    if using_embeddings:
        try:
            query_embedding = np.asarray(encode_query(query), dtype=np.float32)
            if (query_embedding.ndim != 1
                    or query_embedding.shape[0] != block_embeddings.shape[1]
                    or not np.isfinite(query_embedding).all()):
                raise ValueError("Embedding模型返回无效的查询向量")
            semantic_scores = block_embeddings @ query_embedding
        except Exception as error:
            # Preserve keyword/topic routing if model inference is unavailable.
            global embedding_available, embedding_error
            embedding_available = False
            embedding_error = str(error)
            using_embeddings = False
            print("[RAG] 查询向量生成失败，使用关键词检索")

    if not using_embeddings:
        semantic_scores = np.asarray([
            lexical_score(query, block["search_text"])
            for block in knowledge_blocks
        ], dtype=np.float32)


    # =====================================================
    # C. Metadata Rerank
    # =====================================================

    candidates = []


    for index, block in enumerate(
        knowledge_blocks
    ):

        semantic_score = float(
            semantic_scores[
                index
            ]
        )


        block_topic = (
            block.get(
                "topic"
            )
        )

        # Metadata only reranks relevant evidence. In keyword fallback mode,
        # source authority, patient population or a topic hint must not create
        # a match when the query shares no keywords with a knowledge block.
        if not using_embeddings and semantic_score <= 0:
            continue


        # -------------------------------------------------
        # 极低相关内容过滤
        #
        # 如果Topic已经明确命中，
        # 允许Topic把它救回来
        # -------------------------------------------------

        if (
            using_embeddings
            and
            semantic_score
            <
            MIN_SEMANTIC_SCORE
            and
            block_topic
            not in routed_topics
        ):

            continue


        # -------------------------------------------------
        # Bonus
        # -------------------------------------------------

        topic_bonus = (
            topic_match_bonus(

                block,

                routed_topics
            )
        )


        population_bonus = (
            population_match_bonus(

                block,

                patient_state
            )
        )


        priority_bonus = (
            source_priority_bonus(
                block
            )
        )


        # -------------------------------------------------
        # Final
        # -------------------------------------------------

        final_score = (

            semantic_score

            +

            topic_bonus

            +

            population_bonus

            +

            priority_bonus
        )


        reasons = []


        if topic_bonus > 0:

            reasons.append(
                "主题匹配"
            )


        if population_bonus > 0:

            reasons.append(
                "适用人群匹配"
            )


        if priority_bonus > 0:

            reasons.append(
                "权威来源加权"
            )


        if not reasons:

            reasons.append(
                "语义相关"
            )


        candidate = dict(
            block
        )


        candidate[
            "semantic_score"
        ] = round(
            semantic_score,
            4
        )


        candidate[
            "topic_bonus"
        ] = round(
            topic_bonus,
            4
        )


        candidate[
            "population_bonus"
        ] = round(
            population_bonus,
            4
        )


        candidate[
            "priority_bonus"
        ] = round(
            priority_bonus,
            4
        )


        candidate[
            "score"
        ] = round(
            final_score,
            4
        )


        candidate[
            "routed_topics"
        ] = routed_topics


        candidate[
            "why_selected"
        ] = reasons


        candidates.append(
            candidate
        )


    # =====================================================
    # D. 排序
    # =====================================================

    candidates.sort(

        key=lambda item:
            item[
                "score"
            ],

        reverse=True
    )


    # =====================================================
    # E. 结果选择
    #
    # 主Topic -> 第二Topic -> 第三Topic
    # -> 综合分数补齐
    # =====================================================

    selected = []

    selected_block_ids = set()

    source_counter = Counter()


    # -----------------------------------------------------
    # Helper
    # -----------------------------------------------------

    def try_add_candidate(
        candidate
    ):

        block_id = (
            candidate.get(
                "block_id"
            )
        )


        source_id = (

            candidate.get(
                "source_id"
            )

            or

            candidate.get(
                "source_file"
            )
        )


        if (
            block_id
            in
            selected_block_ids
        ):

            return False


        if (
            source_counter[
                source_id
            ]
            >=
            max_per_source
        ):

            return False


        selected.append(
            candidate
        )


        selected_block_ids.add(
            block_id
        )


        source_counter[
            source_id
        ] += 1


        return True


    # -----------------------------------------------------
    # 主主题最多取2个
    # -----------------------------------------------------

    if routed_topics:

        primary_topic = (
            routed_topics[0]
        )


        primary_candidates = [

            candidate

            for candidate
            in candidates

            if candidate.get(
                "topic"
            )
            ==
            primary_topic
        ]


        primary_added = 0


        for candidate in (
            primary_candidates
        ):

            if (
                try_add_candidate(
                    candidate
                )
            ):

                primary_added += 1


            if (
                primary_added >= 2
                or
                len(selected)
                >=
                top_k
            ):

                break


    # -----------------------------------------------------
    # 第二主题至少尝试1个
    # -----------------------------------------------------

    if (
        len(selected)
        <
        top_k
        and
        len(routed_topics)
        >= 2
    ):

        second_topic = (
            routed_topics[1]
        )


        for candidate in candidates:

            if (
                candidate.get(
                    "topic"
                )
                !=
                second_topic
            ):

                continue


            if (
                try_add_candidate(
                    candidate
                )
            ):

                break


    # -----------------------------------------------------
    # 第三主题至少尝试1个
    # -----------------------------------------------------

    if (
        len(selected)
        <
        top_k
        and
        len(routed_topics)
        >= 3
    ):

        third_topic = (
            routed_topics[2]
        )


        for candidate in candidates:

            if (
                candidate.get(
                    "topic"
                )
                !=
                third_topic
            ):

                continue


            if (
                try_add_candidate(
                    candidate
                )
            ):

                break


    # -----------------------------------------------------
    # 第四主题
    # -----------------------------------------------------

    if (
        len(selected)
        <
        top_k
        and
        len(routed_topics)
        >= 4
    ):

        fourth_topic = (
            routed_topics[3]
        )


        for candidate in candidates:

            if (
                candidate.get(
                    "topic"
                )
                !=
                fourth_topic
            ):

                continue


            if (
                try_add_candidate(
                    candidate
                )
            ):

                break


    # -----------------------------------------------------
    # 剩余位置
    # -----------------------------------------------------

    for candidate in candidates:

        if (
            len(selected)
            >=
            top_k
        ):

            break


        try_add_candidate(
            candidate
        )


    # =====================================================
    # F. 返回结果
    # =====================================================

    results = []


    for item in selected:

        result = {

            "block_id":
                item.get(
                    "block_id"
                ),

            "topic":
                item.get(
                    "topic"
                ),

            "subtopic":
                item.get(
                    "subtopic"
                ),

            "source_id":
                item.get(
                    "source_id"
                ),

            "source_title":
                item.get(
                    "source_title"
                ),

            "source_file":
                item.get(
                    "source_file"
                ),

            "year":
                item.get(
                    "year"
                ),

            "source_pages":
                item.get(
                    "source_pages"
                ),

            "chapter":
                item.get(
                    "chapter"
                ),

            "population":
                item.get(
                    "population"
                ),

            "authority":
                item.get(
                    "authority"
                ),

            "priority":
                item.get(
                    "priority"
                ),

            "safety_layer_candidate":
                item.get(
                    "safety_layer_candidate",
                    False
                ),

            "semantic_score":
                item.get(
                    "semantic_score"
                ),

            "topic_bonus":
                item.get(
                    "topic_bonus"
                ),

            "population_bonus":
                item.get(
                    "population_bonus"
                ),

            "priority_bonus":
                item.get(
                    "priority_bonus"
                ),

            "score":
                item.get(
                    "score"
                ),

            "why_selected":
                item.get(
                    "why_selected"
                ),

            "routed_topics":
                routed_topics,

            "text":
                item.get(
                    "content"
                )
        }


        results.append(
            result
        )


    # -----------------------------------------------------
    # Debug输出
    # -----------------------------------------------------

    print(f"[RAG] Selected：{len(results)} 个知识块")

    for result in results:
        print("[RAG] Source:", result["source_id"])


    print(
        "[RAG] ==============================\n"
    )


    return results


# =========================================================
# 22. 查看知识库状态
# =========================================================

@_with_index_lock
def get_knowledge_status():

    topic_counter = Counter()

    file_counter = Counter()


    for block in knowledge_blocks:

        topic = block.get(
            "topic"
        )


        source_file = block.get(
            "source_file"
        )


        if topic:

            topic_counter[
                topic
            ] += 1


        if source_file:

            file_counter[
                source_file
            ] += 1


    return {

        "knowledge_directory":
            KNOWLEDGE_DIR,

        "source_registry_path":
            SOURCE_REGISTRY_PATH,

        "registered_sources":
            len(
                source_registry
            ),

        "knowledge_blocks":
            len(
                knowledge_blocks
            ),

        "topics":
            dict(
                topic_counter
            ),

        "files":
            dict(
                file_counter
            ),

        "embedding_model":
            EMBEDDING_MODEL_NAME,

        "embedding_available":
            embedding_available,

        "fallback_mode":
            not embedding_available,

        "embedding_error":
            embedding_error
    }


# =========================================================
# 23. 初始化
# =========================================================

rebuild_index()

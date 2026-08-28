"""最小化的大模型调用封装。

本文件使用 OpenAI 兼容接口，因此可连接 OpenAI、DeepSeek 或课程指定的
兼容服务。
"""

from openai import OpenAI


# 直接在这里填写课程提供的大模型配置。
API_KEY = "请在这里填写 API Key"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"


def call_model(
    user_prompt: str,
    system_prompt: str = "你是一个严谨的问答助手。",
    timeout: float = 60.0,
) -> str:
    """调用一次大模型并返回纯文本答案。"""
    if API_KEY == "请在这里填写 API Key" or not API_KEY.strip():
        raise RuntimeError("请先在 call_model.py 顶部填写 API_KEY")

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=timeout,
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=8192,
        extra_body={"thinking": {"type": "disabled"}},
        stream=False,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("大模型返回了空答案")
    return content.strip()

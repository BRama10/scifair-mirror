from litellm import acompletion
from openai import AsyncOpenAI

LOCAL_LLM_LIST = ["simplescaling/s1.1-32B", "thehunter911/test"]


async def inference(model_name: str, system: str, user: str, *args, **kwargs):
    if model_name in LOCAL_LLM_LIST:
        return await query_llm_on_device(model_name, system, user, *args, **kwargs)

    return await query_llm_remote(model_name, system, user, *args, **kwargs)


async def query_llm_remote(model_name: str, system: str, user: str, *args, **kwargs):
    _messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    return await acompletion(model=model_name, messages=_messages, *args, **kwargs)


async def query_llm_on_device(model_name: str,  system: str, user: str, *args, **kwargs):
    aclient = AsyncOpenAI(
        base_url="https://r7pgla1bfg53sntx.us-east-1.aws.endpoints.huggingface.cloud/v1/",
        api_key="hf_sdfsdfsd",
    )

    _messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


    return await aclient.chat.completions.create(
        model="tgi",
        messages=_messages,
        top_p=None,
        temperature=None,
        max_tokens=32178,
        stream=False,
        seed=None,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None,
    )

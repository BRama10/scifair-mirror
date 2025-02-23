from litellm import acompletion

LOCAL_LLM_LIST = [
    'simplescaling/s1.1-32B'
]

async def inference(model_name: str, system: str, user: str, *args, **kwargs):
    if model_name in LOCAL_LLM_LIST:
        return await query_llm_on_device(model_name, system, user, *args, **kwargs)
    
    return await query_llm_remote(model_name, system, user, *args, **kwargs)

async def query_llm_remote(model_name: str, system: str, user: str, *args, **kwargs):
    _messages = [
        {
            'role': 'system',
            'content': system
        }, 
        {
            'role': 'user',
            'content': user
        }
    ]

    return await acompletion(model=model_name, messages=_messages, *args, **kwargs)

async def query_llm_on_device():
    pass
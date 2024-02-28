from tenacity import retry, stop_after_attempt, wait_random_exponential
import os
import json

class OpenaiPoolRequest:
    def __init__(self, pool_json_file=None):
        self.pool:List[Dict] = []
        __pool_file = pool_json_file
        if os.environ.get('API_POOL_FILE',None) is not None:
            __pool_file = os.environ.get('API_POOL_FILE')
        
        if os.path.exists(__pool_file):
            self.pool = json.load(open(__pool_file))
        
        if os.environ.get('OPENAI_KEY',None) is not None:
            self.pool.append({
                'api_key':os.environ.get('OPENAI_KEY'),
                'organization':os.environ.get('OPENAI_ORG',None),
                'api_type':os.environ.get('OPENAI_TYPE',None),
                'api_version':os.environ.get('OPENAI_VER',None)
            })

    # @retry(wait=wait_random_exponential(multiplier=1, max=30), stop=stop_after_attempt(10),reraise=True)
    def request(self,messages,**kwargs):
        import openai
        import random
        
        item = random.choice(self.pool)
        kwargs['api_key'] = item['api_key']
        
        if item.get('organization',None) is not None:
            kwargs['organization'] = item['organization']
        if item.get('api_base',None) is not None:
            kwargs['api_base'] = item['api_base']
        return openai.ChatCompletion.create(messages=messages,**kwargs)

    # @retry(wait=wait_random_exponential(multiplier=1, max=30), stop=stop_after_attempt(10),reraise=True)
    async def arequest(self,messages,**kwargs):
        import openai
        import random
        
        item = random.choice(self.pool)
        kwargs['api_key'] = item['api_key']
        
        if item.get('organization',None) is not None:
            kwargs['organization'] = item['organization']
        if item.get('api_base',None) is not None:
            kwargs['api_base'] = item['api_base']
        return await openai.ChatCompletion.acreate(messages=messages,**kwargs)
    
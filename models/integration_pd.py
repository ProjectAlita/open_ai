from typing import List, Optional
from pydantic import BaseModel, root_validator, validator

from tools import session_project, rpc_tools
from pylon.core.tools import log
from ...integrations.models.pd.integration import SecretField


CAPATIBILITIES_MAP = {
    'completion':
        ['gpt-3.5-turbo-instruct', 'babbage-002', 'davinci-002'],
    'chat_completion':
        ['gpt-4', 'gpt-4-0613', 'gpt-4-32k', 'gpt-4-32k-0613', 'gpt-3.5-turbo',
         'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-16k-0613'],
    'embeddings':
        ['text-embedding-ada-002', 'davinci', 'curie', 'babbage', 'ada']
}


class CapabilitiesModel(BaseModel):
    completion: bool = False
    chat_completion: bool = False
    embeddings: bool = False


class AIModel(BaseModel):
    id: str
    name: str
    capabilities: dict = {}

    @validator('name', always=True, check_fields=False)
    def name_validator(cls, value, values):
        return values.get('id', value)

    @validator('capabilities', always=True, check_fields=False)
    def capabilities_validator(cls, value, values):
        if value:
            return value
        capabilities = CapabilitiesModel()
        for capability, models in CAPATIBILITIES_MAP.items():
            if any([model in values['id'] for model in models]):
                setattr(capabilities, capability, True)
        return capabilities


class IntegrationModel(BaseModel):
    api_token: SecretField | str
    model_name: str = 'text-davinci-003'
    models: List[AIModel] = []
    api_version: str | None = None
    api_base: str = "https://api.openai.com/v1"
    api_type: str = "open_ai"
    temperature: float = 1.0
    max_tokens: int = 7
    top_p: float = 0.8

    @root_validator(pre=True)
    def prepare_model_list(cls, values):
        models = values.get('models')
        if models and isinstance(models[0], str):
            values['models'] = [AIModel(id=model, name=model).dict(by_alias=True) for model in models]
        return values

    def check_connection(self):
        import openai
        openai.api_key = self.api_token.unsecret(session_project.get())
        openai.api_type = self.api_type
        openai.api_version = self.api_version
        openai.api_base = self.api_base
        try:
            openai.Model.list()
        except Exception as e:
            log.error(e)
            return str(e)
        return True

    def refresh_models(self, project_id):
        integration_name = 'open_ai'
        payload = {
            'name': integration_name,
            'settings': self.dict(),
            'project_id': project_id
        }
        return getattr(rpc_tools.RpcMixin().rpc.call, f'{integration_name}_set_models')(payload)


class OpenAISettings(BaseModel):
    model_name: str = 'text-davinci-003'
    temperature: float = 1.0
    max_tokens: int = 7
    top_p: float = 0.8

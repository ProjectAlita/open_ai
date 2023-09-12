from pylon.core.tools import log  # pylint: disable=E0611,E0401
from pylon.core.tools import web

from tools import rpc_tools
from ..models.integration_pd import IntegrationModel, OpenAISettings
from pydantic import ValidationError
import openai
from ...integrations.models.pd.integration import SecretField


class RPC:
    integration_name = 'open_ai'

    @web.rpc(f'{integration_name}__predict')
    @rpc_tools.wrap_exceptions(RuntimeError)
    def predict(self, project_id, settings, text_prompt):
        """ Predict function """
        try:
            settings = IntegrationModel.parse_obj(settings)
        except ValidationError as e:
            return {"ok": False, "error": e}

        try:
            api_key = SecretField.parse_obj(settings.api_token).unsecret(project_id)
            response = openai.Completion.create(
                model=settings.model_name,
                prompt=text_prompt,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                top_p=settings.top_p,
                api_key=api_key,
            )
            result = response['choices'][0]['text']
        except Exception as e:
            log.error(str(e))
            return {"ok": False, "error": f"{str(e)}"}
        return {"ok": True, "response": result}

    @web.rpc(f'{integration_name}__parse_settings')
    @rpc_tools.wrap_exceptions(RuntimeError)
    def parse_settings(self, settings):
        try:
            settings = OpenAISettings.parse_obj(settings)
        except ValidationError as e:
            return {"ok": False, "error": e}
        return {"ok": True, "item": settings}

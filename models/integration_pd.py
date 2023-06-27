from pydantic import BaseModel

from tools import session_project
from pylon.core.tools import log
from ...integrations.models.pd.integration import SecretField
import openai


class IntegrationModel(BaseModel):
    api_token: SecretField | str

    def check_connection(self):
        openai.api_key = self.api_token.unsecret(session_project.get())
        try:
            openai.Model.list()
        except Exception as e:
            log.error(e)
            return str(e)
        return True

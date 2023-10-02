#!/usr/bin/python3
# coding=utf-8

#   Copyright 2021 getcarrier.io
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

""" Module """
import json
from pylon.core.tools import log  # pylint: disable=E0611,E0401
from pylon.core.tools import module

from tools import VaultClient  # pylint: disable=E0611,E0401

from .models.integration_pd import IntegrationModel


CAPATIBILITIES_MAP = {
    'completion':
        ['gpt-3.5-turbo-instruct', 'babbage-002', 'davinci-002'],
    'chat_completion':
        ['gpt-4', 'gpt-4-0613', 'gpt-4-32k', 'gpt-4-32k-0613', 'gpt-3.5-turbo',
         'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-16k-0613'],
    'embeddings':
        ['text-embedding-ada-002']
}


class Module(module.ModuleModel):
    """ Task module """

    def __init__(self, context, descriptor):
        self.context = context
        self.descriptor = descriptor

    def init(self):
        """ Init module """
        log.info('Initializing AI module')
        SECTION_NAME = 'ai'

        self.descriptor.init_blueprint()
        self.descriptor.init_slots()
        self.descriptor.init_rpcs()
        self.descriptor.init_events()
        self.descriptor.init_api()

        self.context.rpc_manager.call.integrations_register_section(
            name=SECTION_NAME,
            integration_description='Manage ai integrations',
        )
        self.context.rpc_manager.call.integrations_register(
            name=self.descriptor.name,
            section=SECTION_NAME,
            settings_model=IntegrationModel,
        )

        vault_client = VaultClient()
        secrets = vault_client.get_all_secrets()
        if 'open_ai_capatibilities_map' not in secrets:
            secrets['open_ai_capatibilities_map'] = json.dumps(CAPATIBILITIES_MAP)

        vault_client.set_secrets(secrets)

    def deinit(self):  # pylint: disable=R0201
        """ De-init module """
        log.info('De-initializing')

#!/usr/bin/python3
# coding=utf-8

#   Copyright 2024 EPAM Systems
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

""" Method """

import json

from pylon.core.tools import log  # pylint: disable=E0611,E0401,W0611
from pylon.core.tools import web  # pylint: disable=E0611,E0401,W0611

from tools import worker_client  # pylint: disable=E0401


class Method:  # pylint: disable=E1101,R0903,W0201
    """
        Method Resource

        self is pointing to current Module instance

        web.method decorator takes zero or one argument: method name
        Note: web.method decorator must be the last decorator (at top)
    """

    #
    # AI
    #

    @web.method()
    def ai_check_settings(  # pylint: disable=R0913
            self, settings,
        ):
        """ Check integration settings/test connection """
        settings = json.loads(json.dumps(settings))
        #
        result = {
            "routing_key": None,
            #
            "target": "plugins.open_ai_worker.utils.ai.Helper",
            "target_args": None,
            "target_kwargs": {
                "target_class": "langchain_openai.chat_models.base.ChatOpenAI",
                "target_args": None,
                "target_kwargs": {
                    "base_url": settings["api_base"],
                    "api_key": settings["api_token"],
                },
                "client_attr": "client._client",
            },
            "target_io_bound": True,
            #
            "method": "check_settings",
            "method_args": None,
            "method_kwargs": None,
        }
        #
        return result

    @web.method()
    def ai_get_models(  # pylint: disable=R0913
            self, settings,
        ):
        """ Get model list """
        settings = json.loads(json.dumps(settings))
        #
        result = {
            "routing_key": None,
            #
            "target": "plugins.open_ai_worker.utils.ai.Helper",
            "target_args": None,
            "target_kwargs": {
                "target_class": "langchain_openai.chat_models.base.ChatOpenAI",
                "target_args": None,
                "target_kwargs": {
                    "base_url": settings["api_base"],
                    "api_key": settings["api_token"],
                },
                "client_attr": "client._client",
            },
            "target_io_bound": True,
            #
            "method": "get_models",
            "method_args": None,
            "method_kwargs": None,
        }
        #
        return result

    @web.method()
    def count_tokens(  # pylint: disable=R0913
            self, settings, data,
        ):
        """ Count input/output/data tokens """
        #
        try:
            project_id = settings.integration.project_id
        except AttributeError:
            project_id = None
        #
        api_token = worker_client.unsecret_data(
            settings.merged_settings["api_token"], project_id
        )
        #
        model_parameters = {}
        #
        for param in ["max_tokens", "temperature", "top_p"]:
            if param in settings.merged_settings:
                model_parameters[param] = settings.merged_settings[param]
        #
        if isinstance(data, list):
            data = json.loads(json.dumps(data))
        #
        model_is_legacy_completion = False
        for model_data in settings.merged_settings["models"]:
            if model_data["name"] == settings.merged_settings["model_name"]:
                if not model_data["capabilities"]["chat_completion"]:
                    model_is_legacy_completion = True
        #
        target_class = "langchain_openai.chat_models.base.ChatOpenAI"
        if model_is_legacy_completion:
            target_class = "langchain_openai.llms.base.OpenAI"
        #
        result = {
            "routing_key": None,
            #
            "target": "plugins.open_ai_worker.utils.ai.Helper",
            "target_args": None,
            "target_kwargs": {
                "target_class": target_class,
                "target_args": None,
                "target_kwargs": {
                    "model": settings.merged_settings["model_name"],
                    #
                    **model_parameters,
                    #
                    "base_url": settings.merged_settings["api_base"],
                    "api_key": api_token,
                },
                "client_attr": None,
            },
            "target_io_bound": True,
            #
            "method": "count_tokens",
            "method_args": None,
            "method_kwargs": {
                "data": data,
            },
        }
        #
        return result

    #
    # LLM
    #

    @web.method()
    def llm_invoke(  # pylint: disable=R0913
            self, settings, text,
        ):
        """ Call model """
        #
        try:
            project_id = settings.integration.project_id
        except AttributeError:
            project_id = None
        #
        api_token = worker_client.unsecret_data(
            settings.merged_settings["api_token"], project_id
        )
        #
        model_parameters = {}
        #
        for param in ["max_tokens", "temperature", "top_p"]:
            if param in settings.merged_settings:
                model_parameters[param] = settings.merged_settings[param]
        #
        result = {
            "routing_key": None,
            #
            "target": "plugins.open_ai_worker.utils.ai.Helper",
            "target_args": None,
            "target_kwargs": {
                "target_class": "langchain_openai.llms.base.OpenAI",
                "target_args": None,
                "target_kwargs": {
                    "model": settings.merged_settings["model_name"],
                    #
                    **model_parameters,
                    #
                    "base_url": settings.merged_settings["api_base"],
                    "api_key": api_token,
                },
                "client_attr": None,
            },
            "target_io_bound": True,
            #
            "method": "llm_invoke",
            "method_args": None,
            "method_kwargs": {
                "text": text,
            },
        }
        #
        return result

    @web.method()
    def llm_stream(  # pylint: disable=R0913
            self, settings, text, stream_id,
        ):
        """ Stream model """
        #
        try:
            project_id = settings.integration.project_id
        except AttributeError:
            project_id = None
        #
        api_token = worker_client.unsecret_data(
            settings.merged_settings["api_token"], project_id
        )
        #
        model_parameters = {}
        #
        for param in ["max_tokens", "temperature", "top_p"]:
            if param in settings.merged_settings:
                model_parameters[param] = settings.merged_settings[param]
        #
        result = {
            "routing_key": None,
            #
            "target": "plugins.open_ai_worker.utils.ai.Helper",
            "target_args": None,
            "target_kwargs": {
                "target_class": "langchain_openai.llms.base.OpenAI",
                "target_args": None,
                "target_kwargs": {
                    "model": settings.merged_settings["model_name"],
                    #
                    **model_parameters,
                    #
                    "base_url": settings.merged_settings["api_base"],
                    "api_key": api_token,
                    #
                    "streaming": True,
                },
                "client_attr": None,
            },
            "target_io_bound": True,
            #
            "method": "llm_stream",
            "method_args": None,
            "method_kwargs": {
                "text": text,
                "stream_id": stream_id,
            },
        }
        #
        return result

    #
    # ChatModel
    #

    @web.method()
    def chat_model_invoke(  # pylint: disable=R0913
            self, settings, messages,
        ):
        """ Call model """
        #
        try:
            project_id = settings.integration.project_id
        except AttributeError:
            project_id = None
        #
        api_token = worker_client.unsecret_data(
            settings.merged_settings["api_token"], project_id
        )
        #
        model_parameters = {}
        #
        for param in ["max_tokens", "temperature", "top_p"]:
            if param in settings.merged_settings:
                model_parameters[param] = settings.merged_settings[param]
        #
        result = {
            "routing_key": None,
            #
            "target": "plugins.open_ai_worker.utils.ai.Helper",
            "target_args": None,
            "target_kwargs": {
                "target_class": "langchain_openai.chat_models.base.ChatOpenAI",
                "target_args": None,
                "target_kwargs": {
                    "model": settings.merged_settings["model_name"],
                    #
                    **model_parameters,
                    #
                    "base_url": settings.merged_settings["api_base"],
                    "api_key": api_token,
                },
                "client_attr": None,
            },
            "target_io_bound": True,
            #
            "method": "chat_invoke",
            "method_args": None,
            "method_kwargs": {
                "messages": json.loads(json.dumps(messages)),
            },
        }
        #
        return result

    @web.method()
    def chat_model_stream(  # pylint: disable=R0913
            self, settings, messages, stream_id,
        ):
        """ Stream model """
        #
        try:
            project_id = settings.integration.project_id
        except AttributeError:
            project_id = None
        #
        api_token = worker_client.unsecret_data(
            settings.merged_settings["api_token"], project_id
        )
        #
        model_parameters = {}
        #
        for param in ["max_tokens", "temperature", "top_p"]:
            if param in settings.merged_settings:
                model_parameters[param] = settings.merged_settings[param]
        #
        result = {
            "routing_key": None,
            #
            "target": "plugins.open_ai_worker.utils.ai.Helper",
            "target_args": None,
            "target_kwargs": {
                "target_class": "langchain_openai.chat_models.base.ChatOpenAI",
                "target_args": None,
                "target_kwargs": {
                    "model": settings.merged_settings["model_name"],
                    #
                    **model_parameters,
                    #
                    "base_url": settings.merged_settings["api_base"],
                    "api_key": api_token,
                    #
                    "streaming": True,
                },
                "client_attr": None,
            },
            "target_io_bound": True,
            #
            "method": "chat_stream",
            "method_args": None,
            "method_kwargs": {
                "messages": json.loads(json.dumps(messages)),
                "stream_id": stream_id,
            },
        }
        #
        return result

    #
    # Embed
    #

    @web.method()
    def embed_documents(  # pylint: disable=R0913
            self, settings, texts,
        ):
        """ Make embeddings """
        api_token = settings["integration_data"]["settings"]["api_token"]
        model_name = settings["model_name"]
        #
        result = {
            "routing_key": None,
            #
            "target": "plugins.open_ai_worker.utils.ai.Helper",
            "target_args": None,
            "target_kwargs": {
                "target_class": "langchain_openai.embeddings.base.OpenAIEmbeddings",
                "target_args": None,
                "target_kwargs": {
                    "model": model_name,
                    #
                    "base_url": settings["integration_data"]["settings"]["api_base"],
                    "api_key": api_token,
                },
                "client_attr": None,
            },
            "target_io_bound": True,
            #
            "method": "embed_documents",
            "method_args": None,
            "method_kwargs": {
                "texts": texts,
            },
        }
        #
        return result

    @web.method()
    def embed_query(  # pylint: disable=R0913
            self, settings, text,
        ):
        """ Make embedding """
        api_token = settings["integration_data"]["settings"]["api_token"]
        model_name = settings["model_name"]
        #
        result = {
            "routing_key": None,
            #
            "target": "plugins.open_ai_worker.utils.ai.Helper",
            "target_args": None,
            "target_kwargs": {
                "target_class": "langchain_openai.embeddings.base.OpenAIEmbeddings",
                "target_args": None,
                "target_kwargs": {
                    "model": model_name,
                    #
                    "base_url": settings["integration_data"]["settings"]["api_base"],
                    "api_key": api_token,
                },
                "client_attr": None,
            },
            "target_io_bound": True,
            #
            "method": "embed_query",
            "method_args": None,
            "method_kwargs": {
                "text": text,
            },
        }
        #
        return result

    #
    # Indexer
    #

    @web.method()
    def indexer_config(  # pylint: disable=R0913
            self, settings, model,
        ):
        """ Make indexer config """
        #
        model_info = None
        #
        for item in settings["settings"]["models"]:
            if item["name"] == model:
                model_info = item
                break
        #
        if model_info is None:
            raise RuntimeError(f"No model info found: {model}")
        #
        try:
            project_id = settings["project_id"]
        except (AttributeError, KeyError):
            project_id = None
        #
        api_token = worker_client.unsecret_data(
            settings["settings"]["api_token"], project_id
        )
        #
        if model_info["capabilities"]["embeddings"]:
            return {
                "embedding_model": "langchain_openai.embeddings.base.OpenAIEmbeddings",
                "embedding_model_params": {
                    "model": model,
                    #
                    "base_url": settings["settings"]["api_base"],
                    "api_key": api_token,
                },
            }
        #
        model_parameters = {}
        #
        for param in ["max_tokens", "temperature", "top_p"]:
            if param in settings["settings"]:
                model_parameters[param] = settings["settings"][param]
        #
        if not model_info["capabilities"]["chat_completion"]:
            return {
                "ai_model": "langchain_openai.llms.base.OpenAI",
                "ai_model_params": {
                    "model": model,
                    #
                    **model_parameters,
                    #
                    "base_url": settings["settings"]["api_base"],
                    "api_key": api_token,
                },
            }
        #
        return {
            "ai_model": "langchain_openai.chat_models.base.ChatOpenAI",
            "ai_model_params": {
                "model": model,
                #
                **model_parameters,
                #
                "base_url": settings["settings"]["api_base"],
                "api_key": api_token,
            },
        }

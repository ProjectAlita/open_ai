from .models.integration_pd import IntegrationModel

from pylon.core.tools import log
# from ..integrations.models.pd.integration import SecretField


def prepare_conversation(prompt_struct):
    conversation = []
    if prompt_struct.get('context'):
        conversation.append({
            "role": "system",
            "content": prompt_struct['context']
        })
    if prompt_struct.get('examples'):
        for example in prompt_struct['examples']:
            conversation.append({
                "role": "user",
                "content": example['input']
            })
            conversation.append({
                "role": "assistant",
                "content": example['output']
            })
    if prompt_struct.get('prompt'):
        conversation.append({
            "role": "user",
            "content": prompt_struct['prompt']
        })

    return conversation


def prerare_text_prompt(prompt_struct):
    example_template = '\ninput: {input}\noutput: {output}'

    for example in prompt_struct['examples']:
        prompt_struct['context'] += example_template.format(**example)
    if prompt_struct['prompt']:
        prompt_struct['context'] += example_template.format(input=prompt_struct['prompt'], output='')

    return prompt_struct['context']


def prepare_result(content):
    structured_result = {'messages': []}
    structured_result['messages'].append({
        'type': 'text',
        'content': content
    })
    return structured_result


def predict_chat(project_id: int, settings: dict, prompt_struct: dict) -> str:
    import openai

    settings = IntegrationModel.parse_obj(settings)

    api_key = settings.api_token.unsecret(project_id)
    openai.api_key = api_key
    openai.api_type = settings.api_type
    openai.api_version = settings.api_version
    openai.api_base = settings.api_base

    conversation = prepare_conversation(prompt_struct)

    response = openai.ChatCompletion.create(
        model=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        top_p=settings.top_p,
        messages=conversation
    )

    content = response['choices'][0]['message']['content']

    return prepare_result(content)


def predict_text(project_id: int, settings: dict, prompt_struct: dict) -> str:
    import openai

    settings = IntegrationModel.parse_obj(settings)

    api_key = settings.api_token.unsecret(project_id)
    openai.api_key = api_key
    openai.api_type = settings.api_type
    openai.api_version = settings.api_version
    openai.api_base = settings.api_base

    text_prompt = prerare_text_prompt(prompt_struct)

    response = openai.Completion.create(
        model=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        top_p=settings.top_p,
        prompt=text_prompt
    )

    content = response['choices'][0]['text']
    log.info('completion_response %s', content)

    return prepare_result(content)

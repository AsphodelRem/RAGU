entities_description_summary_prompt = """
Ты - полезный ассистент. Твоя задача по данной сущности и набору фраз для её описания сгенерировать одно общее краткое описание. Оно должно быть лаконичным и непротиворечивым.
В качестве ответа приведи только описание, не добавляя ничего лишнего.

Данные:
"""

relationships_description_summary_prompt = """
Ты - полезный ассистент. Твоя задача по данным парам сущностей и набору фраз для описания  их отношений сгенерировать одно общее краткое описание отношений между этими сущностями.
Оно должно быть лаконичным и непротиворечивым.
В качестве ответа приведи только описание, не добавляя ничего лишнего.

Данные:
"""
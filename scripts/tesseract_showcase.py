from PIL import Image
import pytesseract as pyt
from openai import OpenAI
import pickle as pkl

pyt.pytesseract.tesseract_cmd  =  r'C:\Program Files\Tesseract-OCR\tesseract.exe'

with open('credentials.pkl', 'rb') as cr:
    cr = pkl.load(cr)
    oa = cr['openai']
    prompt = cr['prompt2'].replace('\n', ' ')

    client = OpenAI(api_key=oa)



def llm_call(system, user):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    output = str(completion.choices[0].message).removeprefix('ChatCompletionMessage(content="').removesuffix(
        '", role=\'assistant\', function_call=None, tool_calls=None)')
    return (output)


for i in range(1):
    try:
        name = f'img.jpg'
        data = Image.open(name)
        extraction = pyt.image_to_string(data).replace('\n', ' ')
        answer = llm_call(prompt, extraction).removeprefix("ChatCompletionMessage(content='").removesuffix("', role='assistant', function_call=None, tool_calls=None)")
        print(answer)
        print('--------------------------')
    except:
        name = f'jpg.jpeg'
        data = Image.open(name)
        extraction = pyt.image_to_string(data)
        #answer = llm_call(prompt, extraction).removeprefix("ChatCompletionMessage(content='").removesuffix("', role='assistant', function_call=None, tool_calls=None)")
        print(extraction)
        print('--------------------------')
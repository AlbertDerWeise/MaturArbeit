import warnings
import saver
warnings.filterwarnings('ignore')
import praw as pw
import numpy as np
import os
import pickle as pkl
from openai import OpenAI
import pytesseract as pyt
from PIL import Image
import urllib
from indexer import Index as ix

'import tortoise'
from tts import convert

'''For reddit stories, tweak the accessed subreddits to point to any given story subreddit. Make sure to tweak saving, too. '''
if os.name == 'nt':
    pyt.pytesseract.tesseract_cmd  =  r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#windows specific
TagCheckError = type("TagCheckError", (Exception,), {"__init__": lambda self: Exception.__init__(self, 'Tags could not be checked. Reasons could be: 1. No tags were found. 2. Tags were found, however, they are not present in the legal tag array')})
#defines TagCheckError for global scope
'''parameters'''
class Params:
    '''gets credentials from pkl file'''
    def cred(self):
        with open('credentials.pkl', 'rb') as cr:
            cr = pkl.load(cr)
        id = cr['id']
        secret = cr['secret']
        uname = cr['username']
        password = cr['pwd']
        oa = cr['openai']
        prompt = cr['prompt'].replace('\\n', ' ')
        prompt2 = cr['prompt2']
        key1 = cr['elkey1']
        key2 = cr['elkey2']
        key3 = cr['elkey3']
        key4 = cr['elkey4']
        key5 = cr['elkey5']
        key6 = cr['elkey6']
        key7 = cr['elkey7']
        return np.array([id, secret, uname, password, oa, prompt, prompt2, [key1,key2,key3,key4,key5,key6,key7]], dtype=object)

    def pseudo_list(self):
        '''converts llm outputs to python arrays'''
        try:
            start = self.replace(' ', '').replace('),', ') ').removeprefix('[').removesuffix(']').split(' ')
            taglist = []
            for element in start:
                taglist.append([element.split(',')[0].removeprefix("('").removesuffix("'"),
                                round(float(element.split(',')[1].removeprefix("'").removesuffix("%')")) * 0.01, 2)])

            return(taglist)
        except:
            return False

    def checktags(self):

        '''checks the output tags against predetermined array to rule out hallucinations'''
        legal_tags: list[str] = ['school', 'games', 'work', 'family', 'everyday_life/health', 'money', 'relationships', 'animals', 'sports', 'science_technology', 'clothing', 'mental_state', 'literature_television', 'nostalgia', 'drugs', 'celebrities', 'society_and_social_procedures', 'gender_specific', 'common_relatability', 'politics', 'sex', 'music_art', 'ethnicity_culture_languages', 'political_incorrectness', 'misc']
        cleantag = []
        try:
            for tuple in self:
                cleantag.append(tuple[0])
            if set(cleantag).issubset(set(legal_tags)):
                return True
            else:
                diffs = []
                for tag in cleantag:
                    if tag not in legal_tags:
                        diffs.append(tag)
                print(diffs)
                return False
        except:
            raise TagCheckError()

    def get_url(self):
        '''gets the url of the current post'''
        cred = Params.cred(self)
        idd = cred[0]
        secret = cred[1]
        uname = cred[2]
        password = cred[3]
        reddit = pw.Reddit(user_agent='agent', client_id=idd, client_secret=secret, username=uname, password=password)
        return(str(reddit.config.reddit_url)+ str(reddit.submission(self).permalink))

    def get_name_order(self:str):
        charlist = list(self)[:-4]
        order = []
        for char in charlist:
            if char.isnumeric():
                order.append(char)
        order_tag:int= int(''.join(element for element in order))
        return(order_tag)

'''llm call handler'''
class LLM:
    def llm_call(self, args:str):
        cred = Params.cred(self)
        oa = cred[4]
        client = OpenAI(api_key=oa)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": self},
                {"role": "user", "content": args}
            ]
        )
        output = str(completion.choices[0].message).removeprefix('ChatCompletionMessage(content="').removesuffix(
            '", role=\'assistant\', function_call=None, tool_calls=None)')
        return (output)

'''returns clean text from an image'''
class Cleaner:
    def clean(self, src):
        data = Image.open(src)
        user = pyt.image_to_string(data)
        answer = LLM.llm_call(self, user).removeprefix("ChatCompletionMessage(content='").removesuffix("', role='assistant', function_call=None, tool_calls=None)")
        return(answer.replace('\\n', ' '))

'''saves data into pkl'''
class Save:
    def fetch_name(self:str):
        '''gets the current chronological name of an image'''
        with open(self, 'rb') as f:
            numarr:list[int] = pkl.load(f)
        suff_:int = numarr[len(numarr)-1]
        numarr.append(suff_+1)
        os.remove(self)
        with open(self, 'wb') as f:
            pkl.dump(numarr, f)
        name = 'image'+str(suff_)+'.png'
        print('//name fetched')
        return(name)
    def dump_img(self, loc: str):
        '''saves the image from a reddit link'''
        loc = os.path.join('../src', 'images', loc)
        url = self.preview['images'][0]['source']['url']
        urllib.request.urlretrieve(url, loc)
        return(loc)
        print('//image dumped')

    def dump_tags(self, loc:str):
        '''saves the corresponding tags'''
        loc = os.path.join('../src', loc)
        with open(loc, 'rb') as f:
            taglist = pkl.load(f)
        os.remove(loc)
        taglist.append(self)
        with open(loc, 'wb') as f:
            pkl.dump(taglist, f)
            print('//tags dumped')

    def dump_text(self:str, loc:str):
        '''saves the contained text from the respective image'''
        with open(loc, 'rb') as f:
            textlist = pkl.load(f)
        print('opened')
        textlist.append(self.replace('\n', ' ' ))
        os.remove(loc)
        with open(loc, 'wb') as f:
            pkl.dump(textlist, f)
            print('//text dumped')

    def dump_voice(self, loc):
        if self != 'False': exs = True
        else: exs = False
        model = 'tts_models/multilingual/multi-dataset/xtts_v2'
        convert(text=self, save_path=loc, voice_model=model, character='dude1', exist = exs)
        print('//tts saved')



'''the whole damn thing'''
class CommentFetcher:
    def __init__(self:int, time:str, subreddit='memes') -> None:
        '''basically glorified main()'''
        saver.Exec()
        cred = Params.cred(self)
        idd = cred[0]
        secret = cred[1]
        uname = cred[2]
        password = cred[3]
        context = cred[5]
        context2 = cred[6]
        reddit = pw.Reddit(user_agent='agent', client_id=idd, client_secret=secret, username=uname, password=password)
        #post = reddit.subreddit('memes').top(limit=self)
        post = reddit.subreddit(subreddit).top(time, limit=self)
        for sub in post:
            body = ''''''
            body = body + 'title: ' + str(sub.title) + '\n' + '\n' + 'comments: '+'\n'
            url = Params.get_url(sub)
            for comment in reddit.submission(sub).comments[:10]:
                body = body + comment.body + '\n'
            if '[gif]' in body:
                body.replace('[gif]', 'skip')
            print(body)
            #print('link: ', Params.get_url(sub))
            output = LLM.llm_call(context, body)
            clean_output = Params.pseudo_list(output)
            try:
                check = Params.checktags(clean_output)
            except TagCheckError as t:
                print(t)
                check = False
            print(clean_output, check)
            if check:
                Save.dump_tags(clean_output, 'tags.pkl')
                name = Save.fetch_name(os.path.join('../src', 'order.pkl'))
                loc = Save.dump_img(sub, name)
                image_text = Cleaner.clean(context2, loc)
                Save.dump_text(image_text, os.path.join('../src', 'text', 'text.pkl'))
                print(image_text)
                print(os.path.join('../src', 'speech', 'speech' + name.removeprefix('image').removesuffix('.png') + '.wav'))
                speechpath = os.path.join('../src', 'speech', 'speech' + name.removeprefix('image').removesuffix('.png') + '.wav')
                Save.dump_voice(image_text, speechpath)
                if image_text == 'False':
                    image_text = ''
                if not ix.url_in_list(url):
                    finalarr = [loc,speechpath, clean_output, image_text, url]
                    ix.dump_item(finalarr)

import warnings
import random
import compile_vid
warnings.filterwarnings('ignore')
import os
import pickle as pkl
import saver
from testdumpling import regen
if os.path.exists('../src'):
    print('successfully imported')
    import fetch_data
else:
    print('the src directory does not exist... regenerating...')
    saver.Exec()
    regen()
    import fetch_data
from gtts import gTTS
import ffmpeg


PATH = os.getcwd()
print(PATH)
'''Revert to ffmpeg + subprocess!!!'''
#fetch_data.CommentFetcher.__init__(self=10, time='month', subreddit='meirl')
class DIR:
    def __init__(self):
        def mkdir(dir):
            if not os.path.exists(dir):
                os.mkdir(dir)

        dirlist:list[str] = ['assets','outputs']
        for dir in dirlist:
            mkdir(dir)
        print('ojjij')



class tts:
    def read(self:str,args:list) -> None:
        with open(self, 'rb') as f:
            script_list = pkl.load(f)
            print(script_list)
        for index in args:
            name:str = 'speech' + str(index) +'.mp3'
            if not os.path.exists(os.path.join('../src', name)):
                reading = gTTS(script_list[index])
                reading.save(os.path.join('../src', name))


class mkvid:
    def get_name(self):
        try:
            with open(os.path.join('../src', self), 'rb') as f:
                label_list:list[int] = pkl.load(f)
                latest:int = label_list[len(label_list)-1]
                label_list.append(latest+1)
            with open(os.path.join('../src', self), 'wb') as f:
                pkl.dump(label_list, f)
            name = 'video' + str(latest)
            return(name)
        except Exception as e:
            print('relevant data:', self, ' ', e)

    def get_image_list(self:str):
        image_list = []
        for image in os.listdir(self):
            if image.endswith('.png'):
                image_list.append(os.path.join(self,image))
        return(image_list)
    def make(self:str, args:str, debug:bool=False):
        global PATH
        randnum = random.randint(0, 1000000)
        path_out = os.path.join(PATH, self + '/', args + str(randnum) + '.mp4')
        while os.path.exists(path_out):
            path_out = os.path.join(PATH, self + '/', args + str(randnum) +'.mp4')
        print('pathout:', path_out)
        if not os.path.exists(os.path.join(PATH, self)):
            os.mkdir(os.path.join(PATH, self))
        #title = mkvid.get_name('output_labels.pkl') #TODO: add tag indexing - separate indexing file for videos
        image_list:list = mkvid.get_image_list('../src/images')
        rand_asset = os.listdir('../assets/vids')
        rand_asset = rand_asset[random.randint(0, len(rand_asset) - 1)]
        bkgrd:str = os.path.join(PATH,'../assets', 'vids',rand_asset)
        image_list:list[str] = random.sample(image_list, len(image_list))[0:6]
        print(image_list)
        compile_vid.make_video(image_list, path_out, bkgrd=bkgrd)



'''tts.read(os.path.join('src', 'text.pkl'), [1,3])

DIR.__init__(1)'''

mkvid.make('../outputs', 'helloworld1', debug=True)



import moviepy as mp
import os
import random
import moviepy.editor as mpe
import pickle as pkl
import indexer as ix

'''If implemented, add algorithm functionality from train.py'''
def first_images_list(num:int):
    '''returns a list of the first num images in the images folder'''
    image_list:list[str] = [os.path.join('../src', 'images', image) for image in os.listdir(os.path.join('../src', 'images'))]
    return image_list[0:num]

def get_sorted_vid_and_audio(addresslist:list[str]) -> list[str]: #array in format: [path_image, path_tts, [tagarray], full_text, url]
    with open('../src/indexing.pkl', 'rb') as f:
        addressars = pkl.load(f)
        addressars = [(address[0],address[1]) for address in addressars if address[0] in addresslist]
    imgarr = [address[0] for address in addressars]
    audioarr = [address[1] for address in addressars]
    '''sorting purposes'''
    return([imgarr, audioarr])


#imglist in indexing form - universal path
def make_video(imglist:list[str], name:str, bkgrd=os.path.join('../assets', 'vids', 'bkgrd.mp4'), audio=os.path.join('../assets/music', random.choice(os.listdir(os.path.join('../assets', 'music'))))) -> None:
    background = bkgrd
    video = mpe.VideoFileClip(background)
    video_size = video.subclip(0, 1).size
    width = video_size[0]
    # height = video_size[1]
    iml = imglist
    print(imglist, '--------------------------------------------------')
    tlist = ix.Index.taglist(iml)
    tlist = ix.FullIndex.combine_tags(tlist)
    ix.FullIndex.index_video(name, tlist)
    lists = get_sorted_vid_and_audio(iml)
    iml = lists[0]
    print(iml)
    audl = lists[1]

    random_audio = audio
    print(random_audio)
    audio_background = mpe.AudioFileClip(random_audio).set_duration(video.duration)

    audcomplist = [audio_background]
    composite_list = [video]
    overallduration = 0
    for index, tts in enumerate(audl):
        audio = mpe.AudioFileClip(tts) #haha tiTTS
        duration = audio.duration
        audcomplist.append(audio.set_start(overallduration))
        image = iml[index] #only possible because the images are sorted the same way as the audios
        vclip = mpe.ImageClip(image).set_start(overallduration).set_duration(duration).set_pos(("center", "center")).resize(width=width)
        composite_list.append(vclip)

        overallduration += duration+0.5
        print('overall duration: ',overallduration)
    audio_background = mpe.CompositeAudioClip(audcomplist)
    video.audio = audio_background




    final = mpe.CompositeVideoClip(composite_list)
    final.write_videofile(f"{name}", threads=32)



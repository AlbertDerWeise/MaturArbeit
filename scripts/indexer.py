import numpy as np
import os
import pickle as pkl
import pandas as pd


class Index:
    def index_array_test(self='self') -> bool:
        if os.path.exists(os.path.join('../src', 'indexing.pkl')) and os.path.exists(os.path.join('../src', 'fullvideo_indexing.pkl')):
            return(True)
        else:
            with open(os.path.join('../src', 'indexing.pkl'), 'wb') as f:
                pkl.dump([], f)
            with open(os.path.join('../src', 'fullvideo_indexing.pkl'), 'wb') as f:
                pkl.dump([], f)
            return(True)

    #array in format: [path_image, path_tts, [tagarray], full_text, url]
    def load_subarray(self: str) -> np.array:
        if Index.index_array_test():
            with open(os.path.join('../src', 'indexing.pkl'), 'rb') as f:
                allarrays = pkl.load(f)
                controller = False
            for array in allarrays:
                if array[0] == self:
                    return(array)
                    controller = True

            if controller == False : raise FileNotFoundError(f'{self} not found in indexing.pkl, consider reindexing')

    def load_item(self: str, *args) -> np.ndarray:
        arr = Index.load_subarray(self)
        namesequence = ['path_image', 'path_tts', 'tagarray', 'full_text', 'url']
        final_items = \
            [
                arr[namesequence.index(element)]
                for element in args
                if element in namesequence
            ]
        return np.array(final_items)[0]

    def taglist(self: np.array) -> np.array:
        tags = []
        for entry in self:
            l_tags = Index.load_item(entry, 'tagarray')
            for tag in l_tags:
                tags.append(tag)
        return tags


    def dump_item(self: np.ndarray) -> None:
        if Index.index_array_test():
            with open(os.path.join('../src', 'indexing.pkl'), 'rb') as f:
                allarrays = pkl.load(f)
            allarrays.append(self)
            with open(os.path.join('../src', 'indexing.pkl'), 'wb') as f:
                pkl.dump(allarrays, f)
        else:
            raise FileNotFoundError('indexing.pkl not found, consider reindexing')

    def url_in_list(self:str) -> bool:
        if Index.index_array_test():
            with open(os.path.join(os.getcwd(),'../src', 'indexing.pkl'), 'rb') as f:
                allarrays = pkl.load(f)
            for array in allarrays:
                if array[4] == self:
                    return(True)
            return(False)
        else:
            raise FileNotFoundError('indexing.pkl not found, consider reindexing')

class FullIndex:
    def get_current_indices(verbose=False) -> np.ndarray:
        if Index.index_array_test():
            with open(os.path.join(os.getcwd(),'../src', 'fullvideo_indexing.pkl'), 'rb') as f:
                allarrays = pkl.load(f)
                if verbose:
                    print(allarrays)
            return(allarrays)
        else:
            raise FileNotFoundError('fullvideo_indexing.pkl not found, consider reindexing')

    def index_video(self:str, taglist) -> None:
        indexlist = FullIndex.get_current_indices()
        indexlist.append((self, np.array(taglist)))
        with open(os.path.join(os.getcwd(),'../src', 'fullvideo_indexing.pkl'), 'wb') as f:
            pkl.dump(indexlist, f)

    def combine_tags(self: np.ndarray) -> np.ndarray: #tags are in format: ('tag', percentage as float)
        sum = np.sum([float(tag[1]) for tag in self])
        print(f'{sum = }')
        return(np.array([(tag[0], round(float(tag[1])/sum, 3)) for tag in self]))



test = Index.url_in_list('https://www.reddit.com/r/meirl/comments/1c8ykc4/meirl/1')
print(not test)
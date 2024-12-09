import pickle as pkl
import os
import shutil

def regen() -> None:
    if not os.path.exists(os.path.join(os.getcwd(),'../src')):
        print('creating src directory')
        os.mkdir(os.path.join(os.getcwd(),'../src'))
        try:
            os.mkdir(os.path.join(os.getcwd(), '../src', 'speech'))
            os.mkdir(os.path.join(os.getcwd(), '../src', 'images'))
            os.mkdir(os.path.join(os.getcwd(), '../src', 'text'))
        except Exception as e:
            print('something went horribly wrong', e)
    if os.path.exists(os.path.join(os.getcwd(),'../src', 'order.pkl')):
        print('exists')
        try:
            shutil.rmtree(os.path.join('../src'), ignore_errors=True)
            #remove outputs/

            os.mkdir(os.path.join('../src'))
            os.mkdir(os.path.join('../src', 'speech'))
            os.mkdir(os.path.join('../src', 'images'))
            os.mkdir(os.path.join('../src', 'text'))
            shutil.rmtree(os.path.join('../outputs'), ignore_errors=True)
            os.mkdir(os.path.join(os.getcwd(),'../outputs'))
        except FileNotFoundError as e:
            print(e)
            pass
        except FileExistsError as e:
            l: list[int] = [0]
            with open(os.path.join(os.getcwd(), '../src', 'order.pkl'), 'wb') as f:
                pkl.dump(l, f)


    l:list[int] = [0]

    with open(os.path.join(os.getcwd(),'../src', 'order.pkl'), 'wb') as f:
        pkl.dump(l, f)

    with open(os.path.join('../src', 'order.pkl'), 'rb') as f:
        load = pkl.load(f)

    print(list(load))

    taglist = []
    with open(os.path.join('../src', 'tags.pkl'), 'wb') as f:
        pkl.dump(taglist, f)

    with open(os.path.join('../src', 'tags.pkl'), 'rb') as f:
        print(pkl.load(f))

    textlist = []
    with open(os.path.join('../src', 'text', 'text.pkl'), 'wb') as f:
        pkl.dump(textlist, f)

    with open(os.path.join('../src', 'text', 'text.pkl'), 'rb') as f:
        print(pkl.load(f))


    with open(os.path.join('../src', 'output_labels.pkl'), 'wb') as f:
        pkl.dump([0], f)

    print('regenerated')



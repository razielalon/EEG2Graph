import scipy.io as sio
import numpy as np
import os

file_path = "../dataset/Matlab files/resultsZAB_TSR.mat"

try:
    mat = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    print("Keys:", mat.keys())
    
    if 'sentenceData' in mat:
        data = mat['sentenceData']
        print(f"sentenceData length: {len(data)}")
        if len(data) > 0:
            item = data[0]
            print("Fields in first item:", dir(item))
            if hasattr(item, 'content'):
                print("Content:", item.content)
            if hasattr(item, 'word'):
                print("Word data found.")
                words = item.word
                if len(words) > 0:
                    w = words[0]
                    print("Fields in first word:", dir(w))
                    if hasattr(w, 'mean_g1'):
                        print("mean_g1 found:", w.mean_g1)
                    else:
                        print("mean_g1 NOT found in word")
    else:
        print("sentenceData not found")

except Exception as e:
    print(f"Error: {e}")

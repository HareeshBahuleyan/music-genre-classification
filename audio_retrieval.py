
# coding: utf-8

# In[1]:

import youtube_dl
import re
import os
from tqdm import tqdm
import pandas as pd
import numpy as np


# In[2]:

WAV_DIR = 'wav_files/'
genre_dict = {
            '/m/064t9': 'Pop_music',
            '/m/0glt670': 'Hip_hop_music',
            '/m/06by7': 'Rock_music',
            '/m/06j6l': 'Rhythm_blues',
            '/m/06cqb': 'Reggae',
            '/m/0y4f8': 'Vocal',
            '/m/07gxw': 'Techno',
            }

genre_set = set(genre_dict.keys())


# In[3]:

temp_str = []
with open('data-files/csv_files/unbalanced_train_segments.csv', 'r') as f:
    temp_str = f.readlines()


# In[5]:

data = np.ones(shape=(1,4)) 
for line in tqdm(temp_str):
    line = re.sub('\s?"', '', line.strip())
    elements = line.split(',')
    common_elements = list(genre_set.intersection(elements[3:]))
    if  common_elements != []:
        data = np.vstack([data, np.array(elements[:3]
                                         + [genre_dict[common_elements[0]]]).reshape(1, 4)])

df = pd.DataFrame(data[1:], columns=['url', 'start_time', 'end_time', 'class_label'])


# In[ ]:




# In[6]:

df['class_label'].value_counts() # Drop 10k from Techno - to make the data more balanced


# In[8]:

# Remove 10k Techno audio clips
np.random.seed(10)
drop_indices = np.random.choice(df[df['class_label'] == 'Techno'].index, size=10000, replace=False)
df.drop(labels=drop_indices, axis=0, inplace=True)
df.reset_index(drop=True, inplace=False)

# Time to INT 
df['start_time'] = df['start_time'].map(lambda x: np.int32(np.float(x)))
df['end_time'] = df['end_time'].map(lambda x: np.int32(np.float(x)))


# In[46]:




# Example:<br>
# Step 1:<br>
# `ffmpeg -ss 5 -i $(youtube-dl -f 140 --get-url 'https://www.youtube.com/embed/---1_cCGK4M') -t 10 -c:v copy -c:a copy test.mp4`<br>
# Starting time is 5 seconds, duration is 10s.
# 
# Refer: https://github.com/rg3/youtube-dl/issues/622
# 
# Step 2:<br>
# `ffmpeg -i test.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 1 output.wav` <br>
# PCM-16, 44k sampling, 1-channel (Mono)
# <br>
# Refer: https://superuser.com/questions/609740/extracting-wav-from-mp4-while-preserving-the-highest-possible-quality

# In[9]:

for i, row in tqdm(df.iterrows()):
    url = "'https://www.youtube.com/embed/" + row['url'] + "'"
    file_name = str(i)+"_"+row['class_label']
    
    try:
        command_1 = "ffmpeg -ss " + str(row['start_time']) + " -i $(youtube-dl -f 140 --get-url " +                    url + ") -t 10 -c:v copy -c:a copy " + file_name + ".mp4"

        command_2 = "ffmpeg -i "+ file_name +".mp4 -vn -acodec pcm_s16le -ar 44100 -ac 1 " + WAV_DIR + file_name + ".wav"

        command_3 = 'rm ' + file_name + '.mp4' 

        # Run the 3 commands
        os.system(command_1 + ';' + command_2 + ';' + command_3 + ';')
    
    except:
        print(i, url)
        pass


# In[ ]:




# In[ ]:




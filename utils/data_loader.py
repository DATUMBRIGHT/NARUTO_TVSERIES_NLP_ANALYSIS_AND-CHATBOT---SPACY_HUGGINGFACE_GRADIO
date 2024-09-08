from transformers import pipeline
import pandas as pd
from glob import glob


def load_subtitles(file_path):
        files = glob(f'{file_path}/*')
        files.sort()        
        #extract subtitles from dataset
        subs = []
        for file in files:
                with open(file,'r') as file:
                        lines = file.readlines()
                        lines = lines[27:]
                        
                        lines = [(line.split(',,')[-1]).strip() for line in lines]
                        subs.append(lines)
      
        #flatten_list
        subs = [' '.join(sublist) for sublist in subs ]
                     
        #clean texts
        subs = [text.replace('\\N',' ') for text in subs]
        #get

        #season = int(file.split('Season')[-1][:7].split('-')[0])
        #episode = int(file.split('Season')[-1][:7].split('-')[1])
        seasons = [int(file.split('Season')[-1][:7].split('-')[0]) for file in files]
        episodes = [int(file.split('Season')[-1][:7].split('-')[1]) for file in files]
        print(len(seasons),len(episodes),len(subs))
        data = {'subtitles': subs,
                'seasons' :seasons,
                'episodes' : episodes,
                }
        return pd.DataFrame(data)



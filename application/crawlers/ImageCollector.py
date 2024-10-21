
import pandas as pd
import os
import urllib.request

class WebCrawler():
    def __init__(datafame):
        self.df = DataFrame
        self.data_dir = 'data/'
        self.headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/116.0'}
        if os.path.exists(data_dir) == False:
            os.makedirs(data_dir)


        # expects a dataframe (img_name, img_url, label), should download it and create new csv with (img_full_path, label)
    def image_collector(self,image_column) -> pd.Dataframe:
        labels = []
        img_name = []
        for i,_ in enumerate(self.df.values):
            try:
                url = self.df.loc[i,image_column]
                request = urllib.request.Request(url, headers=self.headers)
                img_dir = os.path.join(self.data_dir, os.path.basename(self.df.loc[i, image_column]))
                # Abre a URL e lê o conteúdo da resposta
                with urllib.request.urlopen(request) as response:
                    image_data = response.read()
                    with open(img_dir, 'wb') as img_file:
                        img_name.append(img_file.name[5:])# substring nome da imagem
                        labels.append(self.df.loc[1,'label'])
                        img_file.write(image_data)
                print("Imagem {} baixada com sucesso.".format(i))
            except urllib.error.HTTPError as e:
                print("Erro HTTP:", e)
            except urllib.error.URLError as e:
                print("Erro de URL:", e)
        
        d = {'labels': labels, 'img_name': img_name}
        df_data_labels = pd.DataFrame(data=d)
        pd.DataFrame.to_csv(df_data_labels,'data_labels.csv')
        return df_data_labels

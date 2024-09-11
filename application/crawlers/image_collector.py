
import pandas as pd
import os
import urllib.request

#Carregar o df 
df = pd.read_csv('data.csv')

df[['label','image_path']].tail()

#Criar uma diretorio para armazenar essas imagens 
data_dir = 'data/'

if os.path.exists(data_dir) == False:
    os.makedirs(data_dir)

#set os seus headers
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/116.0'}

#caminho para salvar a imagem
#print(img_dir)
def image_collector(file,label_column,image_column,headers, data_dir):
    labels = []
    img_name = []
    df = pd.read_csv(file)
    for i,_ in enumerate(df.values):
        try:
            url = df.loc[i,image_column]
            request = urllib.request.Request(url, headers=headers)
            img_dir = os.path.join(data_dir, os.path.basename(df.loc[i, image_column]))
            # Abre a URL e lê o conteúdo da resposta
            with urllib.request.urlopen(request) as response:
                image_data = response.read()
                with open(img_dir, 'wb') as img_file:
                    img_name.append(img_file.name[5:])# substring nome da imagem
                    labels.append(df.loc[1,'label'])
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


image_collector('data.csv','label','image_path',headers, data_dir)

pd.read_csv('data_labels.csv')
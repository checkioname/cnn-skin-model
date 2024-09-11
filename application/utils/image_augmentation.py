import os
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from skimage import color, exposure, filters, io, morphology, transform, util
from random import randint

# Argumentos
arg_input_dir = "data/"
arg_output_dir = "test_out/"
arg_output_ext = "jpg"
arg_img_size = 224
arg_color = "rgb"
arg_img_flag_div = False
arg_img_flag_flip = True
arg_img_flag_rot = True
arg_img_flag_lum = False
arg_img_div = 0
arg_img_rot_max = 1
arg_img_rot_step = 1
arg_img_rot_flag_neg = True
arg_img_rot_flag_resize = True
arg_img_lum_max = 0.1
arg_img_lum_step = 0.1


# Realizar variacao de luminosidade
angle_step_stop = -arg_img_rot_max - arg_img_rot_step
for i in range(0, arg_img_div):
    print(1)


#Receber todas essas variaveis ou talvez uma lista?
#melhor guardar tudo em um arquivo(os parametros)??
#receber uma coluna do df com o path das imagens e uma com o label

def ImageAugmentation(label_columns,image_columns,arg_input_dir,arg_output_dir,arg_output_ext,arg_img_size,arg_color,arg_img_flag_div,arg_img_flag_flip,arg_img_flag_rot,arg_img_flag_lum,arg_img_div,arg_img_rot_max,arg_img_rot_step,arg_img_rot_flag_neg,arg_img_rot_flag_resize,arg_img_lum_max,arg_img_lum_step):

    # Criacao de diretorio de saida
    if os.path.exists(arg_output_dir):
        shutil.rmtree(arg_output_dir)
    try: 
        os.mkdir(arg_output_dir) 
    except OSError as error: 
        print(error)

    #unir nome da imagem ao seu diretorio
    img_path = arg_input_dir +image_columns
    df = pd.DataFrame(data=[label_columns,img_path])

    img_novas = []
    labels = []
    
    ic = io.ImageCollection(img_path.tolist())
    # Contador para a nomeacao das imagens
    n_image = 0

    # Duplica as imagens
    pbar = tqdm(total=len(ic))
    for img in tqdm(ic, desc="Loading..."):
        pbar.update(1)
        
        # Lista de imagens criads e alteradas
        l_img = []
        # ---
        # Pre-processamento da imagem 
        # (deve ser replicado para a aplicacao do modelo)
        
        # Converter imagem para uint8
        img = util.img_as_ubyte(img)
        
        # Definir dominio de cor
        if arg_color != "rgb":
        
            # Converter dominio de cor (RGB para Gray)
            if arg_color == "gray":
                
                img = color.rgb2gray(img)
                
            # Converter dominio de cor (RGB para HSV)    
            elif arg_color == "hsv":
                
                img = color.rgb2hsv(img)
        
        # Modificar tamanho da imagem (fixo)
        img = transform.resize(img,(arg_img_size, arg_img_size),anti_aliasing=True)

        # Armazena em memoria a imagem 'original'
        l_img.append(img)
        
        
        # ---
        # Aumento da base de dados
        # (apenas para treinamento)
        
        # Cortar imagem e redefinir o tamanho
        if arg_img_flag_div == True:
    
                i_num_size_base = int(arg_img_size / arg_img_div)

                n = randint(0,2)

                if n == 1:
                    
                    for i in range(0, arg_img_div):
                        
                        i_num_size_row_min = i * i_num_size_base
                        i_num_size_row_max= ((i + 1) * i_num_size_base) - 1
                        
                        for j in range(0, arg_img_div):
                            
                            i_num_size_col_min = j * i_num_size_base
                            i_num_size_col_max = ((j + 1) * i_num_size_base) - 1
                            
                            img_cut = img[i_num_size_row_min:i_num_size_row_max,i_num_size_col_min:i_num_size_col_max]
                    
                            img_cut = transform.resize(img_cut,
                                                        (arg_img_size, arg_img_size),
                                                        anti_aliasing=True)
                            
                            # Armazenar imagens modificadas em memoria
                            l_img.append(img_cut)
            
            
                n = randint(0,2)
                if n == 1:       
            
                    # Realizar espelhamento
                    if arg_img_flag_flip == True:
            
                        l_img_operation = []
                        for img_flip in l_img:
                            
                            # Armazenar imagem 'original'
                            l_img_operation.append(img_flip)
        
                            n = randint(0,1)
        
                            if n ==1:
                                # Espelhar imagem horizontalmente
                                img_flip_hor = np.flipud(img_flip)        
                                l_img_operation.append(img_flip_hor)
                                
                                #  Atualizar lista de imagens
                                l_img.clear()
                                l_img = l_img_operation.copy()
                                l_img_operation.clear()             
                            else:
        
                                # Espelhar imagem vericalmente
                                img_flip_vert = np.fliplr(img_flip)        
                                l_img_operation.append(img_flip_vert)
                                
                                #  Atualizar lista de imagens
                                l_img.clear()
                                l_img = l_img_operation.copy()
                                l_img_operation.clear()             
                            
                            # Espelhar imagem horizontalmente e vericalmente
                            # Se a rotacao nao estiver ativa
                            #img_flip_hor_vert = np.fliplr(img_flip_hor)        
                            #l_img_operation.append(img_flip_hor_vert)
                            


                n = randint(0,2)

                if n ==1:    
                    
                    # Rotacionar imagem
                    if arg_img_flag_rot == True:
            
                        l_img_operation = []
                        for img_rot in l_img:
                            
                            
                            # Armazenar imagem 'original'
                            l_img_operation.append(img_rot)
        
        
                            n = randint(0,1)
        
                            if n ==1:
                                # Rotacionar no sentido anti-horario
                                angle_step_stop = arg_img_rot_max + arg_img_rot_step
                                for angle_step in np.arange(arg_img_rot_step, angle_step_stop, arg_img_rot_step):
                        
                                    img_rotated = transform.rotate(img_rot, angle=angle_step, resize=arg_img_rot_flag_resize)
                                    l_img_operation.append(img_rotated)
                                    
                                    #  Atualizar lista de imagens
                                    l_img.clear()
                                    l_img = l_img_operation.copy()
                                    l_img_operation.clear()
        
                            else:
                                # Rotacionar no sentido horario        
                                if arg_img_rot_flag_neg == True:
                                    
                                    angle_step_stop = -arg_img_rot_max - arg_img_rot_step
                                    for angle_step in np.arange(-arg_img_rot_step, angle_step_stop, -arg_img_rot_step):
                            
                                        img_rotated = transform.rotate(img_rot, angle=angle_step, resize=arg_img_rot_flag_resize)
                                        l_img_operation.append(img_rotated)
                                        
                                        #  Atualizar lista de imagens
                                        l_img.clear()
                                        l_img = l_img_operation.copy()
                                        l_img_operation.clear()
                                

                n = randint(0,2)

                if n ==1:
                    # Alterar luminosidade da imagem
                    if arg_img_flag_lum == True:
                    
                        l_img_operation = []
                        for img_lum in l_img:
                            
                            # Armazenar imagem 'original'
                            l_img_operation.append(img_lum)
        
                            n = randint(0,2)
        
                            if n ==1:
                                # Realizar variacao de luminosidade
                                lum_step_stop = arg_img_lum_max + arg_img_lum_step
                                for gamma_step in np.arange(arg_img_lum_step, lum_step_stop, arg_img_lum_step):
            
                                    n = randint(0,2)
            
                                    if n ==1:
                                        # Variacao - aumento de luminosidade
                                        img_lum_adjusted = exposure.adjust_gamma(img_lum, gamma=(1 - gamma_step), gain=1)
                                        l_img_operation.append(img_lum_adjusted)
                                        
                                        #  Atualizar lista de imagens
                                        l_img.clear()
                                        l_img = l_img_operation.copy()
                                        l_img_operation.clear()
                                    else:
                                        # Variacao - reducao de luminosidade
                                        img_lum_adjusted = exposure.adjust_gamma(img_lum, gamma=(1 + gamma_step), gain=1)
                                        l_img_operation.append(img_lum_adjusted)
                                        #  Atualizar lista de imagens
                                        l_img.clear()
                                        l_img = l_img_operation.copy()
                                        l_img_operation.clear()
                    

                    
        
                # Salvar imagens modificadas
                print('tamanho do array: ',len(l_img))
                for img_save in l_img:
                    
                    # Converter imagem para uint8
                    img_save_ui8 = util.img_as_ubyte(img_save)
        
                    # Salvar imagem
                    io.imsave((arg_output_dir + str(n_image) + "." + arg_output_ext), img_save_ui8)
                    img_novas.append((arg_output_dir + str(n_image) + "." + arg_output_ext))
                    labels.append(label_columns[i])
                    n_image += 1
                    
                l_img.clear()
                pbar.close()
        # Save augmented images and labels to a DataFrame
        df_augmented = pd.DataFrame({'label': labels, 'image_path': img_novas})
        df_augmented.to_csv('augmented_data.csv', index=False)
        print('acabou')

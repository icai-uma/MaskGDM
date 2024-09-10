import os 
import re

def renameEmbbedingLatent():
    pathTrainningEmbedding = '/mnt2/fscratch/users/tic_163_uma/josdiafra/PHD/editGAN/ISIC/trainning_embedding'
    listDirectory = os.listdir(pathTrainningEmbedding)
    listDirectoryLatent = list(filter(lambda x: re.match('^latents_', x),listDirectory))
    
    for i in range (0,len(listDirectoryLatent)):
        pathLatent_src = os.path.join(pathTrainningEmbedding,listDirectoryLatent[i])

        newName = f'latents_image_{i}.npy'
        pathLatent_dst = os.path.join(pathTrainningEmbedding,newName)
        
        os.rename(pathLatent_src, pathLatent_dst)
        
        #print(pathLatent_src)

def renameDatasetMask_npy():
    pathDataset = '/mnt2/fscratch/users/tic_163_uma/josdiafra/datasets/ISIC/label_data/label_npy'
    listDirectory = os.listdir(pathDataset)

    for i in range (0,len(listDirectory)):
        path_src = os.path.join(pathDataset,listDirectory[i])

        newName = f'ISIC_{i}_segmentation.npy'
        path_dst = os.path.join(pathDataset,newName)
        
        os.rename(path_src, path_dst)

def renameDatasetImage():
    pathDataset = '/mnt2/fscratch/users/tic_163_uma/josdiafra/datasets/ISIC/label_data/image'
    listDirectory = os.listdir(pathDataset)

    for i in range (0,len(listDirectory)):
        path_src = os.path.join(pathDataset,listDirectory[i])

        newName = f'{i}.jpg'
        path_dst = os.path.join(pathDataset,newName)
        
        os.rename(path_src, path_dst)


def main():
    renameDatasetImage()
    #renameDataset()

if __name__ == '__main__':
    main()
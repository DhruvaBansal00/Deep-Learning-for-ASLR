from generator.classifierFromState import getClassifierFromStateAlignment
import glob
import tqdm
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA

def read_ark_files(ark_file):
    with open(ark_file,'r') as ark_file:
        fileAsString = ark_file.read()
    contentString = fileAsString.split("[ ")[1].split("\n]")[0].split("\n")
    content = [[float(i) for i in frame.split()] for frame in contentString]
    return np.array(content)

def _create_ark_file(df: pd.DataFrame, ark_filepath: str, title: str) -> None:
    """Creates a single .ark file

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing selected feature.

    ark_filepath : str
        File path at which to save .ark file.

    title : str
        Title containing label needed as header of .ark file.
    """

    if not os.path.exists(os.path.dirname(ark_filepath)):
        os.makedirs(os.path.dirname(ark_filepath))

    with open(ark_filepath, 'w+') as out:
        out.write('{} [ '.format(title))
        
    df.to_csv(ark_filepath, mode='a', header=False, index=False, sep=' ')

    with open(ark_filepath, 'a') as out:
        out.write(']')


def createNewArkFile(arkFile: str, trainedClassifier: object, pca_components: int, no_pca: bool, 
                    arkFileSave: str, parallel: bool, n_jobs: int):
    content = read_ark_files(arkFile)
    newContent = trainedClassifier.getTransformedFeatures(content, parallel, n_jobs)
    
    if not no_pca:
        pca = PCA(n_components=pca_components)
        newContent = pca.fit_transform(newContent)

    num_features = newContent.shape[1]
    arkFileName = arkFile.split("/")[-1]
    arkFileSavePath = arkFileSave + arkFileName

    _create_ark_file(pd.DataFrame(data=newContent), arkFileSavePath, arkFileName.replace(".ark", ""))
    return num_features

def generateFeatures(resultFile: str, arkFolder: str, classifier: str, include_state: bool, include_index: bool,
								 n_jobs: int, parallel: bool, trainMultipleClassifiers: bool, knn_neighbors: float,
                                 generated_features_folder: str, pca_components: int, no_pca: bool) -> object:
    
    trainedClassifier = getClassifierFromStateAlignment(resultFile, arkFolder, classifier=classifier, include_state=include_state, 
                        include_index=include_index, n_jobs=n_jobs, parallel=parallel, trainMultipleClassifiers=trainMultipleClassifiers,
                        knn_neighbors=int(knn_neighbors))
    
    arkFiles = glob.glob(arkFolder+"/*")

    print(f'Writing .ark files to {generated_features_folder}')
    
    for arkFile in tqdm.tqdm_notebook(arkFiles):
        num_features = createNewArkFile(arkFile, trainedClassifier, pca_components, no_pca, generated_features_folder, parallel, n_jobs)
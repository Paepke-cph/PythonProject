import pandas as pd
import numpy as np

def get_and_clean_df(file):
    print(file)
    dataset = pd.read_csv(file)
    return clean_df(dataset)

def clean_df(dataset):
    # Remove unused tables
    del dataset['_id']
    del dataset['Unnamed: 0.1']
    del dataset['Unnamed: 0']
    del dataset['Sidst synet']
    del dataset['Nummerplade']
    del dataset['Trækhjul']
    del dataset['Tank']
    del dataset['Registreringsdato']
    del dataset['Max påhæng']
    del dataset['Biltype']
    del dataset['Link']
    del dataset['Farve']
    del dataset['Cylindre']
    del dataset['Lasteevne']
    del dataset['Airbags']

    dataset = dataset.dropna(axis = 0, how ='any')  

    # Make
    # dataset['Make'] = dataset['Model'].apply(
    #     lambda x: (str(x).split(" ", 1)[0]))
    # dataset = dataset[~dataset['Make'].isin(['-'])]

    # Model
    # dataset['Model'] = dataset['Model'].apply(lambda x: "" if len(
    #     str(x).split(" ", 2)) <= 1 else str(x).split(" ", 2)[1])
    # dataset = dataset[~dataset['Model'].isin(['-', ' ', ''])]

    # Brændstoftype
    dataset = dataset[~dataset['Brændstoftype'].isin(['-'])]

    # Gearkasse
    dataset = dataset[~dataset['Gearkasse'].isin(['-'])]

    # Ny Pris
    dataset['Nypris'] = dataset['Nypris'].apply(
        lambda x: str(x).split(" ", 1)[0])
    dataset['Nypris'].replace('', np.nan, inplace=True)
    dataset.dropna(subset=['Nypris'], inplace=True)

    # Pris
    dataset['Pris'] = dataset['Pris'].apply(lambda x: str(x).split(" ", 1)[0])
    dataset = dataset[~dataset['Pris'].isin(
        ['ring', 'Ring', '[SOLGT]', '[', '-', 'Byd'])]
    dataset['Pris'] = dataset['Pris'].str.replace('.', '').astype(int)

    # 0 - 100 Km/t
    dataset['0 - 100 km/t'] = dataset['0 - 100 km/t'].apply(
        lambda x: str(x).split(" ", 1)[0])
    dataset = dataset[~dataset['0 - 100 km/t'].isin(['-'])]
    dataset['0 - 100 km/t'] = dataset['0 - 100 km/t'].str.replace(
        ',', '.').astype(float)

    # Km/l
    dataset['Km/l'] = dataset['Km/l'].apply(lambda x: str(x).split(" ", 1)[0])
    dataset = dataset[~dataset['Km/l'].isin(['-'])]
    dataset['Km/l'] = dataset['Km/l'].str.replace(',', '.').astype(float)

    # Tophastighed
    dataset['Tophastighed'] = dataset['Tophastighed'].apply(
        lambda x: str(x).split(" ", 1)[0])
    dataset = dataset[~dataset['Tophastighed'].isin(['-'])]
    dataset['Tophastighed'] = dataset['Tophastighed'].str.replace(
        ',', '.').astype(float)

    # Vægt
    dataset['Vægt'] = dataset['Vægt'].apply(lambda x: str(x).split(" ", 1)[0])
    dataset = dataset[~dataset['Vægt'].isin(['-'])]
    dataset['Vægt'] = dataset['Vægt'].str.replace(',', '.').astype(float)

    # Grøn ejerafgift
    dataset['Grøn Ejerafgift'] = dataset['Grøn Ejerafgift'].apply(
        lambda x: str(x).split(" ", 1)[0])
    dataset = dataset[~dataset['Grøn Ejerafgift'].isin(['-'])]
    dataset['Grøn Ejerafgift'] = dataset['Grøn Ejerafgift'].str.replace(
        ',', '.').astype(float)

    # Kilometer
    dataset = dataset[~dataset['Kilometer'].isin(['-', 'NaN'])]
    # dataset['Kilometer'] = dataset['Kilometer'].str.replace(
    #     '.', '').astype(int)

    # Årgang
    dataset = dataset[~dataset['Årgang'].isin(['-'])]
    dataset['Årgang'] = dataset['Årgang'].astype(int)

    # Antal døre
    dataset = dataset[~dataset['Antal døre'].isin(['-'])]

    # Antal gear
    dataset = dataset[~dataset['Antal gear'].isin(['-'])]

    # Antal Hestekræfter
    dataset = dataset[~dataset['Hestekræfter'].isin(['-'])]

    def df_column_switch(df, column1, column2):
        i = list(df.columns)
        a, b = i.index(column1), i.index(column2)
        i[b], i[a] = i[a], i[b]
        df = df[i]
        return df

    dataset = df_column_switch(dataset, '0 - 100 km/t', 'Brændstoftype')
    dataset = df_column_switch(dataset, 'Antal døre', 'Gearkasse')
    dataset = df_column_switch(dataset, 'Antal gear', 'Make')
    dataset = df_column_switch(dataset, '0 - 100 km/t', 'Model')
    dataset = df_column_switch(dataset, 'Pris', 'Årgang')
    return dataset

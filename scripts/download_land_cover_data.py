# This script downloads the data in a given dataset from Earth Engine's Data Catalog
# Note that the procedures varies depending on the dataset and you may still need
# to manually clean up the downloaded data (remaining commas for example).

import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_Landcover_100m_Proba-V-C3_Global'
source_code = requests.get(url).text
soup = BeautifulSoup(source_code, 'html.parser')

rows = soup.find_all('table')[1].find_all('tr')
for row in rows:
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]
    print(cols)

df = pd.DataFrame(columns=['Value', 'Color', 'Description'])

for row in rows:
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]
    if len(cols) == 3:
        df = df.append(
            {'Value': cols[0], 'Color': cols[1], 'Description': cols[2]}, ignore_index=True)

df = df.convert_dtypes()

df['Description'] = df['Description'].str.replace('\n', ' ')
df['Description'] = df['Description'].apply(lambda x: x.strip())
# The following lines may be used depending on the downloaded data
# df['Description'] = df['Description'].str.replace(',', ':')
# df['Description'] = df['Description'].apply(lambda x: x.split('>')[-1])

# Add '#' to the beginning of the color code
df['Color'] = df['Color'].apply(lambda x: '#' + x.lower())

# df.to_csv('data/Copernicus_Landcover_100m_Proba-V-C3_Global.csv', index=False)

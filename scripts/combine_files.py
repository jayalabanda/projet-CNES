import glob
import pandas as pd

extension = 'csv'
all_filenames = list(glob.glob(f'*.{extension}'))
combinded_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
combinded_csv.to_csv(f'combined_{extension}.{extension}', index=False)

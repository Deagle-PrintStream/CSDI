import tarfile
import zipfile
import sys
import os
import wget
import requests
import pandas as pd
import pickle

os.makedirs("data/", exist_ok=True)
#healthcare dataset, target download folder: ./data/physio/set-a/
if sys.argv[1] == "physio":
    url = "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download"
    wget.download(url, out="data")
    with tarfile.open("data/set-a.tar.gz", "r:gz") as t:
        t.extractall(path="data/physio")

elif sys.argv[1] == "pm25":
    url = "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/STMVL-Release.zip"
    urlData = requests.get(url).content
    filename = "data/STMVL-Release.zip"
    with open(filename, mode="wb") as f:
        f.write(urlData)
    with zipfile.ZipFile(filename) as z:
        z.extractall("data/pm25")
        
    def create_normalizer_pm25():
        df = pd.read_csv(
            "./data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
            index_col="datetime",
            parse_dates=True,
        )
        test_month = [3, 6, 9, 12]
        for i in test_month:
            df = df[df.index.month != i] #type:ignore
        mean = df.describe().loc["mean"].values
        std = df.describe().loc["std"].values
        path = "./data/pm25/pm25_meanstd.pk"
        with open(path, "wb") as f:
            pickle.dump([mean, std], f)
    create_normalizer_pm25()

elif sys.argv[1] == "stock":
    import qstock as qs
    os.chdir(sys.path[0])
    os.makedirs("./data/stock/", exist_ok=True)
    #stock_list=["SH","SZ","CYB","hs300","sz50","zz500",'DJIA','SPX','NDX','HSI']
    stock_list=["SH","SZ","CYB","hs300","sz50","zz500",'DJIA','SPX','NDX','HSI']
    for stock in stock_list:
        path=f"./data/stock/{stock}.csv"
        if os.path.isfile(path)==False:
            df=qs.get_data(code_list=[stock],freq="d")
            df=df[["open","high","low","close","volume"]]
            df.to_csv(path)
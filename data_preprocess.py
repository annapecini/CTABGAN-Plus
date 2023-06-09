import pandas as pd
import datetime

real_path = "Real_Datasets/nike_sales_short.csv"

def date_to_days(date):
    days = (date - datetime.datetime(1970, 1, 1)).days
    return days

def main():

    df = pd.read_csv(real_path)

    df = df.dropna(subset=['value'])

    # Convert date to datetime type
    df["date"] = pd.to_datetime(df["date"], format='%Y-%m-%d')
    #df["hour"] = pd.to_datetime(df["date"]).dt.hour

    #df = df.drop(["cal_holiday", "weather_main", "weather_description", "hour"], axis=1)

    # Convert date to number of days
    df['date'] = df['date'].apply(lambda x: date_to_days(x))

    print(df.info())
    df.to_csv('Real_Datasets/nike_short_processed.csv', index=False)

if __name__ == "__main__":
    main()

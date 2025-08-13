import time
import requests
import argparse
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

API_BASE_URL = "https://wikimedia.org/api/rest_v1/metrics"
TOP_ENDPOINT = "pageviews/top"
TOP_ARGS = "{project}/{access}/{year}/{month}/{day}"


def get_top_wiki_articles(project, year, month, day, access="all-access"):
    args = TOP_ARGS.format(project=project, access=access, year=year, month=month, day=day)
    return __api__(TOP_ENDPOINT, args)


def __api__(end_point, args, api_url=API_BASE_URL):
    url = "/".join([api_url, end_point, args])
    response = requests.get(url, headers={"User-Agent": "wiki parcer"})
    if response.status_code == 200:
        return response.json()
    else:
        pass


parser = argparse.ArgumentParser(description="Process start and end dates.")
parser.add_argument("start", type=str, help="The start date in YYYY-MM-DD format")
parser.add_argument("end", type=str, help="The end date in YYYY-MM-DD format")
args = parser.parse_args()
start = args.start
end = args.end

start_date = dt.datetime.strptime(start, "%Y%m%d")
end_date = dt.datetime.strptime(end, "%Y%m%d")
delta = dt.timedelta(days=1)

df = pd.DataFrame()
while start_date <= end_date:
    year = str(start_date.year)
    month = str(start_date.month)
    day = str(start_date.day)
    start_date += delta
    data = get_top_wiki_articles("en.wikipedia", year, month, day)
    data = pd.DataFrame(data["items"][0]["articles"])
    data["date"] = f"{year}{month}{day}"
    df = pd.concat([df, data])

df["date"] = pd.to_datetime(df["date"])
idx = pd.MultiIndex.from_product([df["article"].unique(), pd.date_range(start=df["date"].min(), end=df["date"].max())], names=["article", "date"])

df.set_index(["article", "date"], inplace=True)
df = df.reindex(idx)
df = df.reset_index(drop=False)

df["views"] = df.groupby("article")["views"].transform(lambda x: x.ffill())


last_values = df.groupby("article")["views"].last()
top_articles = last_values.nlargest(20)
df_top_articles = df[df["article"].isin(top_articles.index)]

views_sum = {article: 0 for article in df_top_articles["article"].unique()}
count = {article: 0 for article in df_top_articles["article"].unique()}

for i in range(len(df_top_articles)):
    article = df_top_articles.iloc[i]["article"]
    views = df_top_articles.iloc[i]["views"]
    views_sum[article] += views
    count[article] += 1

mean_views = {article: views_sum[article] / count[article] for article in views_sum}

mean_views = int(np.nanmean(list(mean_views.values())))
max_views = df["views"].max()
unique_articles = df["article"].nunique()

title = f"Top articles wiki views (Mean: {mean_views:.2f}, Max: {max_views}, Articles: {unique_articles})"

plt.figure(figsize=(12, 8))
for article in df_top_articles["article"].unique():
    df_article = df_top_articles[df_top_articles["article"] == article]
    plt.plot(df_article["date"], df_article["views"], label=article)

plt.yscale("log")
plt.title(title)
plt.legend()
plt.savefig("top_articles.png")
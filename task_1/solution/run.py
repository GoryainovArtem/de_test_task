import argparse
import logging
from datetime import (datetime,
                      date,
                      timedelta)
from typing import ClassVar, Any, Optional
from http import HTTPStatus
import json

from tqdm import tqdm
import requests
import backoff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class WikipediaPlotCreator:
    API_BASE_URL: ClassVar[str] = "https://wikimedia.org/api/rest_v1/metrics"
    PROJECT: ClassVar[str] = "en.wikipedia"
    TOP_ENDPOINT: ClassVar[str] = "pageviews/top"
    TOP_ARGS: ClassVar[str] = "{project}/{access}/{year}/{month}/{day}"
    HTTP_HEADERS: ClassVar[dict[str, str]] = {"User-Agent": "wiki parcer"}
    PLOT_STEP_INTERVAL: ClassVar[timedelta] = timedelta(days=1)
    DEFAULT_PLOT_FILE_PATH: ClassVar[str] = "top_articles.png"
    MAX_REQUEST_RETRIES: int = 5

    def __init__(self,
                 start_date: str,
                 end_date: str,
                 plot_file_path: Optional[str] = None):
        self._start_date: date = self.__format_date(start_date)
        self._end_date: date = self.__format_date(end_date)
        self._plot_file_path: str = (plot_file_path if plot_file_path
                                     else self.DEFAULT_PLOT_FILE_PATH)

    def __get_top_wiki_articles(self,
                                year: int,
                                month: int,
                                day: int,
                                access: str = "all-access") -> dict[Any, Any]:
        args = self.TOP_ARGS.format(project=self.PROJECT,
                                    access=access,
                                    year=year,
                                    month=month,
                                    day=day)
        return self.__get_wikipedia_data(args)

    def __format_url(self, endpoint_top_args: str) -> str:
        return "/".join([self.API_BASE_URL,
                         self.TOP_ENDPOINT,
                         endpoint_top_args])

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_tries=MAX_REQUEST_RETRIES
    )
    def __do_api_request(self, url, **kwargs):
        response = requests.get(url, **kwargs)
        return response

    def __get_wikipedia_data(self, endpoint_top_args: str) -> dict[str, Any]:
        url = self.__format_url(endpoint_top_args)
        response = self.__do_api_request(url, headers=self.HTTP_HEADERS)
        if response.status_code == HTTPStatus.OK:
            try:
                response_content: dict[Any, Any] = response.json()
                return response_content
            except json.JSONDecodeError as err:
                logger.error("Can't convert http response body to JSON."
                             " Error: %s", err)
                raise err
        else:
            msg: str = (f"Can't do http request to {response.url}. "
                        f"Status code: {response.status_code}. "
                        f"Error: {response.content}")
            logger.error(msg)
            raise Exception(msg)

    @staticmethod
    def __format_date(datetime_str: str, dt_format: str = "%Y%m%d") -> date:
        try:
            return datetime.strptime(datetime_str, dt_format)
        except ValueError as err:
            logger.error(err)
            raise err

    @staticmethod
    def __validate_api_response(response: dict[Any, Any]) -> None:
        items: Optional[list] = response.get("items")
        if not items:
            msg: str = "Key 'items' wasn't found in the response"
            logger.error(msg)
            raise ValueError(msg)
        elif not isinstance(items, list):
            msg: str = "Key 'items' isn't iterable"
            logger.error(msg)
            raise ValueError(msg)
        elif len(items) == 0:
            msg: str = "'items' is an empty list"
            logger.error(msg)
            raise ValueError(msg)
        articles: Optional[list] = items[0].get("articles")
        if not articles:
            msg: str = "Key 'articles' wasn't found in the response"
            logger.error(msg)
            raise ValueError(msg)

    def __extract_data(self) -> pd.DataFrame:
        df: pd.DataFrame = pd.DataFrame()
        start_date: date = self._start_date
        end_date: date = self._end_date
        with tqdm(total=(end_date - start_date).days) as pbar:
            while start_date <= end_date:
                step_year: int = start_date.year
                step_month: int = start_date.month
                step_day: int = start_date.day
                start_date += self.PLOT_STEP_INTERVAL
                step_wiki_top_articles: dict[str, Any] = (
                    self.__get_top_wiki_articles(
                        step_year,
                        step_month,
                        step_day
                    )
                )
                self.__validate_api_response(step_wiki_top_articles)
                df_top_wiki_articles_per_day = pd.DataFrame(
                    step_wiki_top_articles["items"][0]["articles"]
                )
                df_top_wiki_articles_per_day["date"] = date(year=step_year,
                                                            month=step_month,
                                                            day=step_day)
                df = pd.concat([df, df_top_wiki_articles_per_day])
                pbar.update(1)
        return df

    @staticmethod
    def __create_multi_index_df(df) -> pd.DataFrame:
        idx = pd.MultiIndex.from_product(
            [df["article"].unique(),
             pd.date_range(start=df["date"].min(), end=df["date"].max())],
            names=["article", "date"])
        df.set_index(["article", "date"], inplace=True)
        df = df.reindex(idx)
        df = df.reset_index(drop=False)
        return df

    @staticmethod
    def __get_top_articles(df, limit=20) -> pd.DataFrame:
        last_values = df.groupby("article")["views"].last()
        top_articles = last_values.nlargest(limit)
        df_top_articles = df[df["article"].isin(top_articles.index)]
        return df_top_articles

    @staticmethod
    def __get_plot_title(df: pd.DataFrame,
                         df_top_articles: pd.DataFrame) -> str:
        views_sum = {article: 0 for article
                     in df_top_articles["article"].unique()}
        count = {article: 0 for article in df_top_articles["article"].unique()}

        for i in range(len(df_top_articles)):
            article = df_top_articles.iloc[i]["article"]
            views = df_top_articles.iloc[i]["views"]
            views_sum[article] += views
            count[article] += 1

        mean_views = {article: views_sum[article] / count[article]
                      for article in views_sum}

        mean_views = int(np.nanmean(list(mean_views.values())))
        max_views = df["views"].max()
        unique_articles = df["article"].nunique()

        title = (f"Top articles wiki views (Mean: {mean_views:.2f}, "
                 f"Max: {max_views}, Articles: {unique_articles})")
        return title

    def __create_plot(self, title: str, df_top_articles: pd.DataFrame) -> None:
        plt.figure(figsize=(12, 8))
        for article in df_top_articles["article"].unique():
            df_article = df_top_articles[df_top_articles["article"] == article]
            plt.plot(df_article["date"], df_article["views"], label=article)

        plt.yscale("log")
        plt.title(title)
        plt.legend()
        plt.savefig("top_articles.png")

    def create_changes_in_page_popularity_plot(self):
        df = self.__extract_data()
        df_top_articles = self.__get_top_articles(df)
        df = self.__create_multi_index_df(df)
        plot_title: str = self.__get_plot_title(df, df_top_articles)
        self.__create_plot(plot_title, df_top_articles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process start "
                                                 "and end dates.")
    parser.add_argument("start", type=str,
                        help="The start date in YYYY-MM-DD format")
    parser.add_argument("end", type=str,
                        help="The end date in YYYY-MM-DD format")
    parser.add_argument("--plot_file_path", type=str,
                        required=False,
                        default=None,
                        help="Path to save plot")
    args = parser.parse_args()
    plot_creator = WikipediaPlotCreator(start_date="2222",
                                        end_date=args.end,
                                        plot_file_path=args.plot_file_path)
    plot_creator.create_changes_in_page_popularity_plot()

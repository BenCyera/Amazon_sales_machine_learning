import requests


class ExchangeRateFetcher:
    def __init__(self, base_currency):
        """
        ExchangeRateFetcher 객체 초기화. 초기화 시 기준 통화에 대한 환율 데이터를 가져옴.
        :param base_currency: 기준 통화 코드 (예: 'USD')
        """
        self.base_currency = base_currency
        self.exchange_rates = None
        self.fetch_exchange_rates()

    def fetch_exchange_rates(self):
        """
        API를 호출하여 환율 정보를 가져와 객체의 상태로 저장
        """
        response = requests.get(
            f"https://api.exchangerate-api.com/v4/latest/{self.base_currency}"
        )

        if response.status_code == 200:
            self.exchange_rates = response.json()["rates"]
        else:
            raise Exception(f"API 호출 실패, Status code: {response.status_code}")

    def convert(self, amount, from_currency, to_currency):
        """
        주어진 금액을 from_currency에서 to_currency로 변환
        :param amount: 변환할 금액
        :param from_currency: 변환할 통화의 코드 (예: 'USD')
        :param to_currency: 결과 통화의 코드 (예: 'EUR')
        :return: 변환된 금액
        """
        if not self.exchange_rates:
            raise Exception("환율 정보를 가져오지 못했습니다.")

        # 변환할 통화가 존재하지 않는 경우 오류 발생
        if (
            from_currency not in self.exchange_rates
            or to_currency not in self.exchange_rates
        ):
            raise ValueError("잘못된 통화 코드입니다.")

        # 금액 변환
        converted_amount = (
            amount / self.exchange_rates[from_currency]
        ) * self.exchange_rates[to_currency]
        return round(converted_amount, 2)


def compare_rows(df, id_column="review_id"):
    """
    동일 ID에 대한 행들을 비교하고 차이점을 출력
    :param df: 비교할 DataFrame
    :param id_column: 기준이 되는 ID 컬럼명
    """
    groups = df.groupby(id_column)
    duplicate_groups = [group for _, group in groups if len(group) > 1]

    for group in duplicate_groups:
        print(f"\nComparing rows with {id_column}: {group[id_column].iloc[0]}")

        first_row = group.iloc[0]
        for idx, row in group.iloc[1:].iterrows():
            compare_two_rows(df, group.index[0], idx)


def find_similar_rows(df, threshold=0.9):
    """
    비슷한 행을 찾아서 유사도를 기준으로 출력
    :param df: 비교할 DataFrame
    :param threshold: 유사도 임계값
    """
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            similarity = df.iloc[i].eq(df.iloc[j]).mean()
            if threshold < similarity < 1:
                print(f"\nRows {i} and {j} are {similarity:.2%} similar:")
                compare_two_rows(df, i, j)


def compare_two_rows(df, row1_index, row2_index):
    """
    두 개의 행을 비교하고 차이점을 출력
    :param df: 비교할 DataFrame
    :param row1_index: 첫 번째 행의 인덱스
    :param row2_index: 두 번째 행의 인덱스
    """
    row1 = df.iloc[row1_index]
    row2 = df.iloc[row2_index]

    print(f"\nComparing row {row1_index} and row {row2_index}:")
    for column in df.columns:
        if row1[column] != row2[column]:
            print(f"  {column}:")
            print(f"    Row {row1_index}: {row1[column]}")
            print(f"    Row {row2_index}: {row2[column]}")

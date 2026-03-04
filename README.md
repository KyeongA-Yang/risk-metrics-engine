## Notes (25.3.4)

# Rolling/time series basics #
- pd.to_datetime은 날짜/시간을 파이썬이 "날짜 타입"으로 인식하게 만든다.
- sort_values("date")는 시간 순서가 꼬인 데이터를 정렬한다.
- set_index("date")하면 날짜를 기준으로 시계열 연산(rolling)이 자연스러워진다.
- rolling(window=3)은 "최근 3개" 묶어서 계산하고, 초반(window-1)은 NaN이 생긴다.
- rolling 결과는 원래 데이터와 길이가 같고 인덱스도 그대로라서 "정렬/정합성"이 중요하다.

# dropna #
- dropna()는 NaN이 있는 행을 제거해서 "계산 가능한 구간만" 남긴다.
- rolling 결과는 초반 window-1 구간이 NaN이므로, 필요하면 dropna로 제거한다.
- pandas는 Series 연산 시 "행 순서"가 아니라 "인덱스(날짜)"로 자동 정렬(alignment)한다.
- 그래서 dropna 등으로 인덱스가 달라지면 비교/연산 결과가 예상과 달라질 수 있다.
- 안전하게 하려면 set_index + sort_index 후, 비교 전에 concat(...).dropna()로 인덱스를 맞춘다.
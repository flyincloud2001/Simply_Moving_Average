# BTC_USD
For this example, we try to understand whether BTC-USD prices can be predicted or interpreted well using a simple moving average strategy.

## Idea
Bitcoin prices present a trend that reflects the market's temper. I have not learned the strategy of when and how people buy/sell Bitcoin for their investment. Out of curiosity, I still applied the SMA (simple moving average) strategy on this type of data.

## Randomness
BTC-USD belongs to a type of financial data, therefore it incorporates the randomness of the market.

## Result
As shown in the results in the analysis folder, the best (sma1, sma2) = (7 days, 100 days), where sma1 and sma2 are the short and long time ranges respectively for the SMA strategy. The model was trained over the 2016-01 to 2023-09 period, and tested over the 2023-09 to 2026-06 period, achieving a total performance of 2.3949, which beats the actual price performance of 2.0806 over the same test period.

## Interpretation
During the training period, prices tend to be much smoother than in the later period, which leads to unstable predictability in the later time period. This can be observed in the BTC_USD_with_sma_strategy.png plot, where prices suddenly rise after 2026-03, shortly after the USA declared war on Iran.
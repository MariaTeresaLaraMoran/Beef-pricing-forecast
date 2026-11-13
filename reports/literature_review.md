# Literature Review

Forecasting weekly wholesale beef prices in Mexico with deep learning.

This review covers three bodies of work: (1) how beef prices in Mexico have been
modelled, (2) how deep learning has been applied to livestock prices, and (3) which
architectures perform best on agricultural price series. It closes with how this
project differs from previous work.

---

## 1. Beef price modelling in Mexico: linear, monthly, econometric

Mexican beef prices have been studied mainly with cointegration and error-correction
models rather than machine learning.

**Jaramillo Villanueva et al. (2022)** estimated a vector error correction model on
monthly real beef prices for 1990–2019, examining price transmission between
international and national live cattle prices and between live cattle and carcass
prices. Two findings matter directly for this project:

- Transmission runs **from international beef prices to domestic Mexican prices**, and
  from farm price to processor price. This is the empirical basis for using US prices
  as a predictor of Mexican prices.
- Transmission is **asymmetric**: when the international price rises, Mexican prices
  adjust significantly more slowly than when it falls.

**Rojas-Juárez et al.** estimated a similar error-correction model on monthly steer and
carcass prices for 2003–2020 across Veracruz, Jalisco and the State of Mexico, again
finding asymmetry between regional markets.

Related demand-side work (Rebollar-Rebollar et al., 2000–2020) uses double-logarithm
regressions on annual data to estimate price and income elasticities.

**Why this matters.** The asymmetry these studies document is a *nonlinearity*. A linear
cointegration model can detect it only by splitting the sample or adding regime terms,
whereas a recurrent neural network can represent it directly. The existing Mexican
literature therefore identifies a feature of the data that it is not well equipped to
model — which is the opening this project uses.

## 2. Deep learning for livestock prices

Work applying deep learning specifically to meat prices is recent and concentrated in
Asia.

- **Yoo (2022)**, Agricultural & Applied Economics Association annual meeting, forecast
  livestock prices with LSTM units, focusing on the shocks from African Swine Fever and
  COVID-19 — i.e. periods when linear models break down.
- **Ma, Peng, Chen and Yan (2025)**, *PLOS ONE* 20(2), combined complementary ensemble
  empirical mode decomposition, fuzzy information granulation and an attention-mechanism
  LSTM to produce both point and interval predictions on **weekly** pork, beef and mutton
  prices in China.
- **Zhu, Chuluunsaikhan, Choi and Nasridinov (2025)**, *Frontiers in Sustainable Food
  Systems* 9:1613616, proposed a framework that pairs seasonal decomposition of the price
  series with sentiment analysis of 14,588 news articles, feeding an attention-LSTM.
  Reported MAE improvements of 43–87% over statistical, machine learning and deep
  learning baselines for Korean pork prices, 2018–2021.
- **Yang, Zhang, Yin and Zhu (2022)**, SciTePress, compared ARIMA against LSTM for pork
  price prediction — the same benchmark-versus-deep-learning comparison this project's
  rubric requires.

## 3. Which architectures work on agricultural price series

- **Manogna, Dharmaji and Sarang (2025)**, *Scientific Reports* (10.1038/s41598-025-05103-z),
  evaluated stochastic, machine learning and deep learning models on 23 commodities using
  daily wholesale prices from January 2010 to June 2024. LSTM and GRU performed best, and
  the most volatile commodities benefited most. GRU reached MAPE of 14.59% for onions and
  10.58% for tomatoes, and was described as slightly less robust than LSTM but simpler.
- **A CNN–Transformer hybrid (2026)**, *Springer* (10.1007/s10791-026-09966-2), was tested
  on **weekly** potato prices in four heterogeneous Indian markets and outperformed CNN,
  LSTM, Bi-LSTM, Stacked LSTM, CNN-LSTM, GRU and a standalone Transformer on RMSE, MAE
  and MAPE.
- **Hybrid decomposition approaches** are the current research frontier: VMD-SGMD-LSTM for
  agricultural futures (Zhang and Tang, 2024), and swarm-intelligence-optimised deep
  learning stacks for corn and wheat (2025). These consistently beat plain LSTM but add
  substantial complexity.
- In Mexico, LSTM has mostly been applied to **financial** series rather than agricultural
  ones — for example a wavelet-LSTM for the S&P/BMV IPC index (Massa Roldán et al., 2022,
  *Revista Mexicana de Economía y Finanzas* 17(1)), which outperformed ARIMA and a dense
  network. Mexican *agricultural* price forecasting has used SARIMA and shallow neural
  networks (Paredes-Garcia et al., 2019, *Agronomy* 9(12):826).

**Takeaway for architecture choice.** LSTM and GRU are the established, well-validated
choice for weekly agricultural price series, and comparing the two is itself supported by
the literature, since GRU trades a little robustness for a simpler model. Transformer and
decomposition hybrids are stronger in recent studies but are designed for larger datasets
than ~1,300 weekly observations, and would risk overfitting here.

## 4. How this project differs

| Previous work | This project |
|---|---|
| Mexican beef prices modelled with linear VECM on **monthly** data | **Weekly** data and nonlinear recurrent models |
| Asymmetric transmission documented but modelled linearly | Nonlinearity modelled directly by LSTM/GRU |
| Deep learning for livestock prices focused on Korea and China | **Mexico**, a market with no comparable deep learning study |
| Most deep learning price studies are univariate, or decompose the price alone | Explicit **cross-border exogenous inputs**: US boxed beef cutout, USD/MXN exchange rate, US beef export volumes to Mexico |
| Data typically taken from existing repositories | Data collected directly from four primary sources (SNIIM, USDA AMS, USDA FAS, Banco de México) with reproducible scripts |

In short, the Mexican literature establishes *that* international prices drive domestic
prices and that the relationship is asymmetric; the Asian literature establishes *that*
recurrent networks handle such nonlinearity well. This project joins the two.

## 5. Feasibility and known risks

- **Sample size.** About 1,290 weekly observations after merging (limited by US export
  data starting December 2001). This is comparable to Ma et al. (2025), who used weekly
  Chinese meat prices, but it is small for deep learning. Mitigation: a single recurrent
  layer, modest hidden size, dropout, early stopping, and a time-ordered split rather
  than cross-validation.
- **The benchmark may be hard to beat.** Several of the studies above report large gains
  for deep learning, but those gains come partly from decomposition and attention
  components this project does not use. A modest improvement over linear regression — or
  none — is a legitimate result, and the rubric asks precisely whether the added
  complexity is justified.
- **Structural change.** US beef exports to Mexico fell roughly 70% between 2002 and 2026,
  with a sharp dip in 2004 coinciding with restrictions following the BSE case. A model
  trained across the whole period must handle a changing relationship between the series.
- **Nominal prices.** The Mexican price rose from about 23 to 111 MXN/kg over the period.
  Modelling nominal levels, deflated prices, or weekly changes is an open decision to be
  settled during exploratory analysis.

## References

1. Jaramillo Villanueva, J. L. et al. (2022). Vertical and spatial price transmission in
   the Mexican and international cattle and beef market. *Revista Mexicana de Ciencias
   Pecuarias*, 13(4). https://doi.org/10.22319/rmcp.v13i4.5839
2. Rojas-Juárez, L. A. et al. Vertical and horizontal price transmission in the livestock
   and beef industry in Mexico. *Ecosistemas y Recursos Agropecuarios*.
   https://doi.org/10.19136/era.a9n3.3307
3. Yoo, D. (2022). Livestock Price Forecasting using Long Short-Term Memory Units: the
   Case of African Swine Fever and the COVID-19 Pandemic. AAEA Annual Meeting.
   https://ageconsearch.umn.edu/record/322610
4. Ma, W., Peng, L., Chen, H. and Yan, H. (2025). Interval price prediction of livestock
   product based on fuzzy mathematics and improved LSTM. *PLOS ONE*, 20(2).
5. Zhu, Y., Chuluunsaikhan, T., Choi, J.-H. and Nasridinov, A. (2025). Integrating
   structured and unstructured data for livestock price forecasting: a sustainability
   study from South Korea. *Frontiers in Sustainable Food Systems*, 9:1613616.
   https://doi.org/10.3389/fsufs.2025.1613616
6. Yang, Z., Zhang, R., Yin, Z. and Zhu, Y. (2022). Pork Price Prediction Using LSTM Model
   Based on a New Dataset. SciTePress, pp. 423–429.
7. Manogna, R. L., Dharmaji, V. and Sarang, S. (2025). Enhancing agricultural commodity
   price forecasting with deep learning. *Scientific Reports*.
   https://doi.org/10.1038/s41598-025-05103-z
8. CNN-transformer hybrid deep learning model for agricultural commodities price
   forecasting (2026). *Springer*. https://doi.org/10.1007/s10791-026-09966-2
9. Massa Roldán, R., Reyna Miranda, M. and Gómez Salcido, V. (2022). *Revista Mexicana de
   Economía y Finanzas*, 17(1), e570.
10. Paredes-Garcia, W. J. et al. (2019). Price Forecasting and Span Commercialization
    Opportunities for Mexican Agricultural Products. *Agronomy*, 9(12):826.

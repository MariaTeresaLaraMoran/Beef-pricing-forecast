# Literature Review

Forecasting weekly prices of individual beef cuts in Mexico with deep learning.

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
| Prices studied at the aggregate or carcass level | **Individual cuts** — five cuts spanning five primals, forecast jointly |
| Most deep learning price studies are univariate, or decompose the price alone | Explicit **cross-border exogenous inputs**: US cut prices matched cut-to-cut by IMPS code, boxed beef cutout, USD/MXN, US beef export volumes |
| Models fitted one series at a time | One **multi-output** network forecasting all five cuts jointly, able to exploit the fact that cuts from the same carcass share drivers |
| Data typically taken from existing repositories | Data collected directly from four primary sources (SNIIM, USDA AMS, USDA FAS, Banco de México) with reproducible scripts, and cut names reconciled against the USMEF US–Mexico nomenclature standard |

In short, the Mexican literature establishes *that* international prices drive domestic
prices and that the relationship is asymmetric; the Asian literature establishes *that*
recurrent networks handle such nonlinearity well. This project joins the two, and adds a
level of detail neither has: individual cuts rather than an aggregate.

### Three comparisons

The design yields three distinct questions, each answerable from the same pipeline:

1. **Does nonlinearity help?** LSTM and GRU against ARIMA on the same single series.
2. **Does sharing across cuts help?** One joint five-output network against five
   independent single-cut models. Six separate ARIMAs cannot use the ribeye's history
   to inform the brisket; one network can.
3. **Does US market information help?** The same model trained with and without the US
   inputs — a direct ablation.

### A design that was considered and deferred

A two-stage cascade — forecast US cut prices first, then feed those forecasts into a
Mexican model — was considered and deliberately not built. Stage 2 would inherit stage 1's
forecast error, and the transmission estimates above imply it is unnecessary at a
four-week horizon: because Mexican prices adjust only 3–4% per month toward international
prices, the *observed* US price still carries information about Mexican prices a month
ahead. A cascade becomes necessary only for horizons longer than the transmission lag, and
is recorded as future work.

## 5. Feasibility and known risks

- **Sample size.** 1,248 complete weekly observations after merging, covering December
  2001 to October 2026, with five cut series each. The binding constraint is the US export
  series, which begins in December 2001. This is comparable to Ma et al. (2025), who used
  weekly Chinese meat prices, but it is small for deep learning. Mitigation: a single
  recurrent layer, modest hidden size, dropout, early stopping, and a time-ordered split
  rather than cross-validation. Forecasting five cuts jointly also helps here, since one
  network trained on five correlated series has more signal to learn from than five
  networks trained on one series each.

- **Price stickiness.** Between 21% and 33% of weeks show an exactly zero price change,
  because establishments hold reported prices unchanged for weeks at a time. A naive
  "price stays put" forecast is therefore right surprisingly often, which makes the naive
  baseline a harder floor to beat than it appears.

- **Data entry errors.** SNIIM is hand-entered and contains occasional impossible values —
  brisket recorded at 10 and at 2,850 MXN/kg against a median near 49. With only three to
  five establishments reporting a cut in a given week, one such record moves the average by
  an order of magnitude. Records beyond five times the median for their own cut in their own
  year are dropped, which removed 284 of 364,549 records (0.078%).
- **The benchmark may be hard to beat.** Several of the studies above report large gains
  for deep learning, but those gains come partly from decomposition and attention
  components this project does not use. A modest improvement over linear regression — or
  none — is a legitimate result, and the rubric asks precisely whether the added
  complexity is justified.
- **Structural change.** US beef exports to Mexico fell roughly 70% between 2002 and 2026,
  with a sharp dip in 2004 coinciding with restrictions following the BSE case. A model
  trained across the whole period must handle a changing relationship between the series.
- **Nominal prices.** Prices rose roughly fivefold over the period. Exploratory analysis
  settled this: the price levels are non-stationary (ADF p = 0.998 on the carcass series),
  and their first differences fail KPSS marginally because the variance grows with the
  level. The models therefore work in **weekly log returns** — logs stabilise the variance,
  differencing removes the trend.

- **One target has no US counterpart.** Arrachera (outside skirt) was in the original
  target set but was dropped: skirts do not appear among the 42 items of USDA's daily
  negotiated boxed beef report, so it was the only cut without a clean cross-border pair.
  The five retained cuts each map to a USDA item by IMPS code.

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
11. Valdez, A. (U.S. Meat Export Federation, Mexico). *Equivalencias EEUU y México en
    Nomenclatura de Cortes de Res y Cerdo e Inspección*. Used to map SNIIM cut names to
    US cuts: pulpa negra = top (inside) round, pulpa blanca = bottom round / gooseneck,
    pulpa bola = knuckle.
12. Huerta-Leidenz, N. (2013). *Guía de nomenclatura internacional de cortes de carne de
    res para EE.UU. y diferentes países objetivo de la U.S. Meat Export Federation.*
    U.S. Meat Export Federation, Mexico City.
13. Calderón-Alonso, J. et al. (2021). Equivalencia de la nomenclatura y rendimiento de
    cortes primarios de la canal bovina en el noroeste de México. *Nacameh*, 15(1), 30-45.
    Notes that cut nomenclature and coding are not standardised across most of Latin
    America, Mexico included - which is why the mapping in data/reference/ was needed.

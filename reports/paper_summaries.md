# Three Key Papers: Detailed Summaries

Prepared for the literature review section. One paper per strand of the review:
the Mexican beef market, deep learning for meat prices, and architecture comparison.

---

## Paper 1 — Jaramillo Villanueva et al. (2022)
### Vertical and spatial price transmission in the Mexican and international cattle and beef market
*Revista Mexicana de Ciencias Pecuarias* 13(4). https://doi.org/10.22319/rmcp.v13i4.5839

**Research question.** Are Mexican live cattle prices linked to international cattle
prices (spatial transmission) and to carcass prices (vertical transmission), and is the
transmission asymmetric? The stated hypothesis is that it is asymmetric. The purpose is
to assess market efficiency.

**Data.** Monthly spot prices from 1990 to 2019, deflated by the consumer price index so
all series are in real terms. Three series: a farmer price (calf prices), a processor
price (carcass prices), and an international price (USDA calf prices in US dollars,
converted to pesos at the bilateral exchange rate). Sources are USDA for international
prices, and SIAP, INEGI, the National Confederation of Livestock Organizations and the
ANETIF Foundation for the rest.

**Method.** Augmented Dickey-Fuller and Phillips-Perron unit root tests, then
cointegration testing by the Engle-Granger two-step procedure and the Johansen trace and
eigenvalue tests. Dynamics are modelled with symmetric and asymmetric vector error
correction models, where the error correction term is split into positive and negative
components following Cramon-Taubadel. An F-test of the hypothesis that the two adjustment
coefficients are equal tests for asymmetry. Two models are estimated: spatial
(international versus farm prices) and vertical (farm versus processor prices).

**Results.**
- *Cointegration.* For the spatial model the Johansen trace test rejects no cointegration
  at rank 0 (30.21 against a 15.41 critical value). The long-run coefficient on the
  international price is −0.826.
- *Spatial model.* Symmetric error correction coefficient −0.049 (t = −3.26). Split into
  components, ECT+ is −0.049 (t = −2.38) and ECT− is −0.059 (t = −2.05). The symmetry
  F-test is 12.08 and rejects symmetry at 5%.
- *Vertical model.* Symmetric coefficient −0.029 (t = −3.69); ECT+ −0.033 (t = −2.81) and
  ECT− −0.042 (t = −3.56). Symmetry F-test 14.37, rejected at 5%. Processor prices close
  roughly 4.2% of a negative deviation per month against roughly 3% of a positive one.

**Conclusions.** Transmission runs unidirectionally from international to Mexican farm
prices, and from farm to processor prices. It is asymmetric in both relationships: price
decreases pass through faster than increases. The authors attribute this to processor
market power, fixed costs that push plants to run near capacity, inelastic short-run
supply, and limited producer access to price information — but they present these as
possible explanations rather than tested ones. Because Mexico is a net beef importer,
they stress implications for food security and risk management, and suggest support
programmes may benefit processors more than farmers.

**Weaknesses to be aware of.** Read critically, the paper has internal inconsistencies
worth knowing if you cite it:
- The text reports R² of 0.78 (spatial) and 0.68 (vertical), while Table 2 shows 0.150
  and 0.164.
- The residual ADF statistics (−2.57 and −2.70) do not beat the −2.87 critical value,
  which weakens the Engle-Granger cointegration claim.
- The Johansen trace statistic for the vertical model (14.18) is below its 15.41 critical
  value despite being marked significant.
- The "international" series is a US calf price converted to pesos, so it is a proxy
  rather than a Mexican import price; the paper uses "international" and "import"
  interchangeably.
- There is no dedicated limitations section, and the authors state that the causes of the
  asymmetry need further research.

**What this gives our project.** Three things. First, published evidence that US beef
prices drive Mexican prices, which justifies the US cutout and exchange rate as inputs.
Second, a documented nonlinearity — asymmetric adjustment — that linear models capture
only by splitting the sample, and that a recurrent network can represent directly. Third,
an opening: adjustment of 3–4% per month is slow and was measured on monthly data, so
weekly data and a nonlinear model are a genuine extension rather than a repetition.

---

## Paper 2 — Zhu, Chuluunsaikhan, Choi and Nasridinov (2025)
### Integrating structured and unstructured data for livestock price forecasting: a sustainability study from South Korea
*Frontiers in Sustainable Food Systems* 9:1613616. https://doi.org/10.3389/fsufs.2025.1613616

**Research question.** Can news sentiment combined with seasonal decomposition improve
livestock price forecasting, particularly when prices are disturbed by external shocks
such as disease outbreaks? The authors argue earlier sentiment work relied on short texts
like headlines and social media posts, which lack context.

**Data.** Korean pork prices paired with 14,588 news articles from Naver News, covering
874 days between 2018 and 2021. Data from 2018 to 2020 is the training set and 2021 is
the test set — a time-ordered split, not random.

**Method.** The proposed SASD framework has two parts feeding one model:
1. *Seasonal decomposition* of the price series into trend, seasonal and residual
   components, so that seasonal patterns and irregular fluctuations are isolated rather
   than left mixed together.
2. *Sentiment analysis* of the news, using a Korean sentiment lexicon built with an
   improved TF-IDF algorithm that works at the morpheme level.
Both feed an attention-based LSTM, which weights the time steps where sentiment shifts
most.

**Baselines and tuning.** Compared against Ridge regression, Random Forest, Gradient
Boosting, and a plain LSTM. For Ridge, LSTM and AM-LSTM, hyperparameters were chosen by
cross-validation on the training data and then reused across all forecast horizons,
because the chosen settings were stable across horizon lengths. Random Forest and
Gradient Boosting were tuned per horizon with scikit-learn's GridSearchCV. The LSTM grid
is directly reusable for our project: learning rate {0.001, 0.05, 0.1}, recurrent layers
{1, 2, 3}, hidden units {16, 32, 64}.

**Results.** The sentiment classifier reached F1 scores of 0.74 for negative and 0.79 for
positive sentiment. The AM-LSTM beat the statistical, machine learning and deep learning
baselines, with MAE improvements ranging from 43.0% to 87.4%. The SASD framework reduced
MAE by roughly 41.8%, 60.7%, 62.6% and 71.7% at 1-day, 7-day, 15-day and 30-day horizons
respectively — that is, the longer the horizon, the larger the gain.

**Limitations the authors state.** Experiments covered only pork in South Korea with a
single news source, which may introduce bias. The method was not validated on other
commodities or regions, though the authors expect it would generalise to beef, lamb and
chicken. Building the Korean sentiment lexicon remains a bottleneck, and they suggest
expert annotation or transfer learning.

**What this gives our project.** The closest methodological relative: weekly-to-daily
livestock prices, a recurrent architecture, a time-ordered split, and exogenous
information beyond the price itself. Three practical takeaways. The hyperparameter grid is
a sensible starting point for our own tuning. Their decomposition step speaks to our open
question about modelling nominal levels versus deseasonalised series. And the finding that
gains grow with the forecast horizon is encouraging for our 4-week target, which is longer
than the 1-week horizon where their improvement was smallest. Their own stated gap —
untested on beef and outside Korea — is precisely the space this project occupies.

---

## Paper 3 — Manogna, Dharmaji and Sarang (2025)
### Enhancing agricultural commodity price forecasting with deep learning
*Scientific Reports* 15, article 20903. https://doi.org/10.1038/s41598-025-05103-z

**Research question.** Which family of models forecasts agricultural commodity prices
best: traditional stochastic models, machine learning, or deep learning? The motivation is
that ARIMA captures linear relationships but struggles with the nonlinear, nonstationary
behaviour of agricultural prices.

**Data.** Daily wholesale prices from India's AGMARKNET portal, 26 February 2010 to
11 June 2024, covering 23 commodities across 165 markets: three vegetables (potato, onion,
tomato), four oilseeds, five pulses, four spices, six cereals and cotton. Commodities were
chosen to span different volatility levels; onion is the most variable, with a coefficient
of variation of about 0.765. No series passed the Jarque-Bera normality test.

**Stationarity.** Tested with both ADF and KPSS. Most series are non-stationary; only
onion and tomato are clearly stationary. Several commodities give conflicting results
between the two tests, which the authors read as possible trend stationarity or structural
breaks — a useful precedent for how to report mixed test results.

**Models compared.** ARIMA, linear regression, SVR, XGBoost, MLP, RNN, LSTM, GRU and Echo
State Networks. Trained with a window size of 1000 and an 80/20 train/test split.

**Metrics.** RMSE, relative normalised MSE, MAE and MAPE. Crucially, they also apply the
**Diebold-Mariano test** to check whether the difference in forecast errors between two
models is statistically significant, reporting p-values below 0.05.

**Results.** GRU and LSTM were best across metrics, with the largest margins on the most
volatile commodities:

| Metric | Commodity | GRU | LSTM | ARIMA |
|---|---|---|---|---|
| RMSE | Onion | 369.54 | — | 1564.62 |
| RMSE | Tomato | 210.35 | — | 1298.60 |
| RNMSE | Onion | 0.08 | — | 0.32 |
| RNMSE | Tomato | 0.06 | — | 0.39 |
| MAE | Onion | 217.73 | 229.70 | 1234.25 |
| MAE | Tomato | 164.28 | 139.30 | 1138.77 |
| MAPE | Onion | 16.17% | — | 92.66% |
| MAPE | Tomato | 14.81% | — | 92.17% |

Note a small inconsistency: the abstract gives GRU MAPE as 14.59% for onions and 10.58%
for tomatoes, while the results discussion gives 16.17% and 14.81%. Cite the paper's
conclusion rather than a single figure.

GRU was described as slightly less robust than LSTM but simpler and cheaper to run. Echo
State Networks were competitive and computationally efficient, since only the output layer
is trained, but slightly worse on the most complex price dynamics. XGBoost and SVR sat in
the middle.

**The important caveat.** The authors note that despite its simplicity, linear regression
was competitive in some cases, and that ARIMA and linear regression occasionally delivered
acceptable results for commodities with relatively stable prices such as paddy and wheat.
They add that for less volatile commodities the simpler models may be preferable when
interpretability or computation matters.

**Conclusions and future work.** Deep learning, and GRU and LSTM in particular, should be
prioritised for practical forecasting. The authors recommend hybrid models and
incorporating external variables such as weather data.

**What this gives our project.** It is the strongest single justification for choosing
LSTM and GRU, and for comparing the two. It also supplies two things we should copy: the
Diebold-Mariano test, which lets us say whether our deep model's improvement over the
baseline is statistically significant rather than merely numerically smaller, and the
practice of reporting ADF alongside KPSS. Most valuably, it is a warning: their large
gains came on the most volatile series, and our weekly Mexican beef price is comparatively
smooth (weekly changes have a standard deviation of about 0.92%). Their own finding that
linear models compete on stable series is the honest prior for our section 4d — we should
expect a modest improvement, not a 75% error reduction.

---

## Summary of what the three establish together

| Paper | Establishes |
|---|---|
| Jaramillo Villanueva et al. (2022) | US prices drive Mexican beef prices, and the relationship is asymmetric — a nonlinearity, measured only on monthly data with a linear model |
| Zhu et al. (2025) | Recurrent networks with exogenous inputs and a time-ordered split work well on livestock prices, with gains growing at longer horizons — but untested on beef and outside Asia |
| Manogna et al. (2025) | LSTM and GRU are the right architectures for this data type, and the Diebold-Mariano test is how to prove a difference is real — while linear models stay competitive on smooth series |

The gap the three leave open is exactly this project: weekly Mexican beef prices,
nonlinear models, and cross-border drivers.

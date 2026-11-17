# Literature Review

Forecasting weekly US boxed beef prices four weeks ahead with deep learning.

This review covers four bodies of work: (1) how wholesale beef prices are formed and what
is known about the spread between what packers buy and what they sell, (2) how beef prices
move between the US and Mexico, which is what the project's ablation tests, (3) how deep
learning has been applied to livestock prices, and (4) which architectures perform best on
agricultural price series. It closes with how this project differs from previous work, the
risks known in advance, and what the data turned out to show — including two places where
the measurements contradicted the design.

A note on scope. The project began as a forecast of Mexican cut prices using US prices as
predictors. The target moved to US boxed beef after both were built and benchmarked
against each other; section 7 reports that test. The Mexican data was not discarded — it
became the ablation in comparison 3, and the fact that it does *not* improve US forecasts
is one of the project's cleaner results.

---

## 1. Wholesale beef prices and the packer margin

US wholesale beef trades as **boxed beef**: primals and sub-primals in vacuum packaging,
priced per hundredweight and reported by USDA Agricultural Marketing Service under
Livestock Mandatory Reporting. The headline series is the **cutout** — a volume-weighted
value of the whole carcass broken into its parts — and alongside it AMS reports individual
items identified by IMPS code.

The economics that matter for forecasting sit on both sides of the packer. A packer buys
fed cattle and sells boxed beef, and the difference between the two, measured on the same
carcass-weight basis, is its gross margin. USDA's Economic Research Service sets out how
these farm-to-wholesale spreads are constructed and documents that adjustment between
levels of the chain is **asymmetric** — the rate at which a price change passes through
depends on its direction (ERS, *Beef and Pork Values and Price Spreads Explained*).

The margin is not merely an accounting residual. It drives slaughter decisions: when it is
wide, packers run plants harder and more beef reaches the market a few weeks later; when
it is thin, they slow kills and supply tightens. That delay is what makes the margin a
*forecasting* variable rather than a contemporaneous one, and it is the central feature
this project adds to the standard price-history input set.

The most dramatic illustration is in the project's own sample. In spring 2020, plant
closures sent the cutout to record highs while fed cattle prices collapsed, and the spread
reached the widest level since the inception of Mandatory Price Reporting. USDA AMS
investigated it formally (*Boxed Beef & Fed Cattle Price Spread Investigation Report*,
July 2020). In this dataset that episode appears as a packer margin of +281 USD/cwt
against a median of 8.91 — a genuine structural event rather than a data error, and one
the error analysis has to acknowledge.

Published forecasting work on North American cattle and beef prices is mostly machine
learning rather than deep learning. **Rotondo et al. (2024, *Sustainability* 16(5):1789)**
modelled Alberta fed steer prices with support vector regression, random forest and
AdaBoost against ARIMA, SARIMA and SARIMAX, finding the multivariate learners competitive
with the univariate time-series models. That is the closest published analogue to this
project's benchmark layer, and it stops short of recurrent networks.

## 2. Price transmission between the US and Mexico

Mexico is among the largest-volume destinations for US beef, so a natural question is
whether destination-market information helps forecast US prices. The published evidence
says causality runs the other way.

**Jaramillo Villanueva et al. (2022)** estimated a vector error correction model on
monthly real beef prices for 1990–2019, examining transmission between international and
national live cattle prices and between live cattle and carcass prices. Two findings
matter here:

- Transmission runs **from international beef prices to domestic Mexican prices**, and
  from farm price to processor price — not in reverse.
- Transmission is **asymmetric**: when the international price rises, Mexican prices
  adjust significantly more slowly than when it falls, at roughly 3–4% per month.

**Rojas-Juárez et al.** estimated a similar error-correction model on monthly steer and
carcass prices for 2003–2020 across Veracruz, Jalisco and the State of Mexico, again
finding asymmetry between regional markets.

This literature is why comparison 3 is framed as it is. If transmission is
one-directional and slow, then Mexican prices should be partly predictable from US
prices, and US prices should *not* be predictable from Mexican ones. Section 7 reports
both halves of that prediction, tested at weekly frequency in a neural network rather
than monthly in a VECM. The asymmetry these studies document is also a *nonlinearity*,
which a linear cointegration model captures only by splitting the sample or adding regime
terms.

## 3. Deep learning for livestock prices

Work applying deep learning specifically to meat prices is recent and concentrated in
Asia.

- **Yoo (2022)**, Agricultural & Applied Economics Association annual meeting, forecast
  livestock prices with LSTM units, focusing on the shocks from African Swine Fever and
  COVID-19 — i.e. periods when linear models break down.
- **Ma, Peng, Chen and Yan (2025)**, *PLOS ONE* 20(2), combined complementary ensemble
  empirical mode decomposition, fuzzy information granulation and an attention-mechanism
  LSTM to produce both point and interval predictions on **weekly** pork, beef and mutton
  prices in China. Their interval work is the closest published parallel to the prediction
  bands produced here, though theirs are model-based and these are empirical.
- **Zhu, Chuluunsaikhan, Choi and Nasridinov (2025)**, *Frontiers in Sustainable Food
  Systems* 9:1613616, paired seasonal decomposition with sentiment analysis of 14,588 news
  articles feeding an attention-LSTM, reporting MAE improvements of 43–87% over
  statistical, machine learning and deep learning baselines for Korean pork prices.
- **Yang, Zhang, Yin and Zhu (2022)**, SciTePress, compared ARIMA against LSTM for pork
  price prediction — the same benchmark-versus-deep-learning comparison this project's
  rubric requires.

**A caution about these reported gains.** Several of these studies benchmark against ARIMA
rather than against a naive random walk, and their improvements come partly from
decomposition and attention components not used here. Section 7 shows ARIMA losing to the
naive forecast on two of this project's seven series, so "beats ARIMA" and "beats the
honest floor" are different claims. This project reports skill against naive throughout
for that reason.

## 4. Which architectures work on agricultural price series

- **Manogna, Dharmaji and Sarang (2025)**, *Scientific Reports*, evaluated stochastic,
  machine learning and deep learning models on 23 commodities using daily wholesale prices
  from January 2010 to June 2024. LSTM and GRU performed best, and the most volatile
  commodities benefited most. GRU was described as slightly less robust than LSTM but
  simpler — a trade-off that mattered here, in the opposite direction to what they report.
- **A CNN–Transformer hybrid (2026)**, *Springer*, was tested on **weekly** potato prices
  in four heterogeneous Indian markets and outperformed CNN, LSTM, Bi-LSTM, Stacked LSTM,
  CNN-LSTM, GRU and a standalone Transformer on RMSE, MAE and MAPE.
- **Hybrid decomposition approaches** are the current frontier: VMD-SGMD-LSTM for
  agricultural futures (Zhang and Tang, 2024), and swarm-intelligence-optimised stacks for
  corn and wheat (2025). These consistently beat plain LSTM but add substantial complexity.
- In Mexico, LSTM has mostly been applied to **financial** series rather than agricultural
  ones — for example a wavelet-LSTM for the S&P/BMV IPC index (Massa Roldán et al., 2022),
  which outperformed ARIMA and a dense network. Mexican *agricultural* price forecasting
  has used SARIMA and shallow neural networks (Paredes-Garcia et al., 2019).

**Takeaway for architecture choice.** LSTM and GRU are the established choice for weekly
agricultural price series, and comparing the two is itself supported by the literature.
Transformer and decomposition hybrids are stronger in recent studies but are designed for
larger datasets than the ~800 training windows available here, and would risk overfitting.

## 5. How this project differs

| Previous work | This project |
|---|---|
| North American cattle price forecasting uses SVR, random forest and ARIMA families | **LSTM and GRU**, with those traditional methods as the benchmark layer rather than the contribution |
| Deep learning for livestock prices focused on Korea and China | **US boxed beef**, a market with no comparable published deep learning study at cut level |
| Price studied at the aggregate or cutout level | **Seven series**: the Choice and Select cutouts plus five individual cuts identified by IMPS code |
| Price-history models, or decomposition of the price alone | An explicit **supply-side state variable**: the gross packer margin, cutout minus dressed fed cattle price on a common carcass basis |
| Cross-border transmission studied with monthly VECMs | The same question as a **neural-network ablation** at weekly frequency |
| Deep learning benchmarked against ARIMA | Benchmarked against **naive**, with Diebold–Mariano tests and seed variance reported |
| A single training run reported | **Five random seeds per configuration**, reported as mean ± standard deviation, with the ensemble alongside |
| Data taken from existing repositories | Collected from **six primary sources**, two of them scraped, with reproducible scripts and cut names reconciled against the USMEF US–Mexico nomenclature standard |

### Three comparisons

1. **Does nonlinearity help?** LSTM and GRU against naive, drift, ridge and ARIMA on the
   same test weeks.
2. **Does sharing across series help?** One joint seven-output network against seven
   independent single-series networks.
3. **Does destination-market information help?** The same architecture trained with and
   without Mexican prices, spreads and the exchange rate — a direct ablation, and a test
   of the transmission asymmetry in section 2.

### A design that was considered and deferred

A two-stage cascade — forecast one market first, then feed those forecasts into a model of
the other — was considered and deliberately not built. Stage 2 would inherit stage 1's
forecast error, and at a four-week horizon the *observed* price still carries the
information a forecast of it would. A cascade becomes necessary only for horizons longer
than the transmission lag, and is recorded as future work.

## 6. Feasibility and known risks

- **Sample size.** About 1,170 complete weekly observations for the US family, of which
  roughly 800 become training windows once a 26-week input window and the train/validation
  split are applied. Consecutive windows overlap by 25 weeks, so the effective sample is
  smaller still. This is small for deep learning. Mitigation: a single recurrent layer, 32
  units, dropout, early stopping, and a time-ordered split rather than cross-validation.

- **The naive floor is harder than it looks.** The US cutout is close to a random walk at
  a four-week horizon. Section 7 shows ARIMA failing to beat it on both cutout series
  despite those series having the strongest weekly momentum in the dataset.

- **The benchmark may simply win.** Ridge regression on lagged features, properly
  regularised, is hard to beat on 800 observations precisely because it cannot overfit
  much. A modest improvement — or none — is a legitimate result, and the rubric asks
  precisely whether the added complexity is justified.

- **Seed variance can exceed the effect.** With 169 overlapping test windows, the spread
  between random initialisations of a small recurrent network can be larger than the gap
  between the models being compared. Every configuration is therefore trained five times
  and reported with its standard deviation.

- **Structural change.** The 2020 packer margin episode described in section 1 sits inside
  the training period. Separately, the 5-Area fed cattle series changes composition when
  AMS suppresses a region for confidentiality: Colorado has been excluded since December
  2022, Texas/Oklahoma/New Mexico since June 2026 and Kansas since August 2026, with all
  five restored on 26 October 2026. Part of that falls inside the test period. The AMS
  narratives recording it are kept in `data/raw/usda_5area_cattle_notes.txt`.

- **Nominal prices.** Prices roughly tripled over the period. The levels are
  non-stationary and their variance grows with the level, so the models work in **weekly
  log returns** — logs stabilise the variance, differencing removes the trend.

- **Data entry errors in the Mexican source.** SNIIM is hand-entered and contains
  occasional impossible values — brisket recorded at 10 and at 2,850 MXN/kg against a
  median near 49. Records beyond five times the median for their own cut in their own year
  are dropped, removing 284 of 364,549 records (0.078%).

## 7. What the data actually shows

The sections above are what the literature claims. This section records what the
exploratory analysis measured, including where it contradicts an expectation the design
had been built on. Figures are in `reports/figures/`, produced by `eda_cuts.py`.

### The two markets behave in opposite ways

Lag-1 autocorrelation of the weekly log return:

| | lag 1 | zero-change weeks |
|---|---|---|
| US Choice cutout | **+0.556** | 0.0% |
| US cuts (five) | +0.41 to +0.53 | 0.0% |
| Mexican cuts (five) | **−0.18 to −0.33** | 21% to 28% |

US prices show momentum: a week up tends to be followed by a week up. Mexican prices show
the opposite — a rise is typically reversed the following week. With three to five
establishments hand-reporting a price that sits unchanged a quarter of the time, much of
that reversal is measurement noise around a slow-moving true price rather than market
behaviour. This is a property of how the two series are produced, not only of how the two
markets work, and it bears directly on how forecastable each one is. It is the reason the
target moved to the US series.

### How much there is to share across series

Correlation of weekly log returns within each family:

| | range | mean |
|---|---|---|
| Mexican cuts | 0.08 – 0.30 | **0.19** |
| US series | 0.19 – 0.95 | **0.57** |

This corrects a claim the design originally rested on. An earlier draft asserted that cuts
from one carcass "share drivers" and that a joint network therefore has more to learn
from, citing a correlation of 0.81. That figure was measured on price *levels*, where any
two series with a common upward trend correlate highly regardless of whether they share
short-run information. The returns are the relevant quantity and they are far less
correlated — and even for the US series, where sharing looks most promising, comparison 2
found that joint training **hurts**.

### Error correction is visible at weekly frequency, in both markets

Raw cross-border returns carry almost nothing: US cut returns correlate with Mexican cut
returns between −0.04 and 0.05, and US price levels at 0.01 to 0.04. The informative
quantity is the *gap between levels*, which mean-reverts:

| Error-correction term | corr. with next 4 weeks | with next 13 weeks |
|---|---|---|
| Mexico-minus-US spread, five cuts | −0.16 to −0.32 | −0.20 to −0.40 |
| US packer margin (cutout − dressed fed steer), Choice | −0.31 | −0.37 |

Both are negative, as error correction predicts, and both strengthen with horizon, which
is the signature of slow adjustment. The Mexican spread replicates the monthly VECM result
of Jaramillo Villanueva et al. (2022) at weekly, cut-level frequency. The packer margin is
the same mechanism operating inside the US market, and matches the asymmetric
farm-to-wholesale adjustment documented by ERS.

For comparison, the US cutout's own weekly return correlates with its next four weeks at
only **+0.09**. The margin is roughly three times as informative about the four-week
horizon as momentum is, despite momentum being far stronger week to week. A model given
only lagged returns would miss most of what is predictable here.

### Which target is more forecastable

Both targets were built and benchmarked under the same procedure. The US family produced
four Diebold–Mariano-significant results against the naive forecast; the Mexican family
produced none. The best Mexican skill was 0.957 (p = 0.315) against 0.805 for the US. The
target moved accordingly, and the Mexican series became the ablation.

### Consequences for the design

- The level spreads, not the raw foreign returns, are the features that matter.
- The multi-output argument cannot be asserted from "cuts come from one carcass"; it has
  to be measured, and when measured it did not hold.
- Whichever market is forecast, the naive floor is harder than it looks: in Mexico because
  prices genuinely do not move a quarter of the time, in the US because the cutout is
  close to a random walk at four weeks.

---

## References

1. Jaramillo Villanueva, J. L. et al. (2022). Vertical and spatial price transmission in
   the Mexican and international cattle and beef market. *Revista Mexicana de Ciencias
   Pecuarias*, 13(4). https://doi.org/10.22319/rmcp.v13i4.5839
2. Rojas-Juárez, L. A. et al. Vertical and horizontal price transmission in the livestock
   and beef industry in Mexico. *Ecosistemas y Recursos Agropecuarios*.
   https://doi.org/10.19136/era.a9n3.3307
3. USDA Economic Research Service. *Beef and Pork Values and Price Spreads Explained*
   (LDP-M-118-01). Farm-to-wholesale spread construction, packer gross margins, and
   asymmetric price transmission.
4. USDA Agricultural Marketing Service (2020). *Boxed Beef & Fed Cattle Price Spread
   Investigation Report*, 22 July 2020.
   https://www.ams.usda.gov/sites/default/files/media/CattleandBeefPriceMarginReport.pdf
5. Rotondo, F. et al. (2024). Using Probabilistic Machine Learning Methods to Improve Beef
   Cattle Price Modeling and Promote Beef Production Efficiency and Sustainability in
   Canada. *Sustainability*, 16(5):1789. https://doi.org/10.3390/su16051789
6. Yoo, D. (2022). Livestock Price Forecasting using Long Short-Term Memory Units: the
   Case of African Swine Fever and the COVID-19 Pandemic. AAEA Annual Meeting.
   https://ageconsearch.umn.edu/record/322610
7. Ma, W., Peng, L., Chen, H. and Yan, H. (2025). Interval price prediction of livestock
   product based on fuzzy mathematics and improved LSTM. *PLOS ONE*, 20(2).
   https://doi.org/10.1371/journal.pone.0318823
8. Zhu, Y., Chuluunsaikhan, T., Choi, J.-H. and Nasridinov, A. (2025). Integrating
   structured and unstructured data for livestock price forecasting: a sustainability
   study from South Korea. *Frontiers in Sustainable Food Systems*, 9:1613616.
   https://doi.org/10.3389/fsufs.2025.1613616
9. Yang, Z., Zhang, R., Yin, Z. and Zhu, Y. (2022). Pork Price Prediction Using LSTM Model
   Based on a New Dataset. SciTePress, pp. 423–429.
10. Manogna, R. L., Dharmaji, V. and Sarang, S. (2025). Enhancing agricultural commodity
    price forecasting with deep learning. *Scientific Reports*.
    https://doi.org/10.1038/s41598-025-05103-z
11. CNN-transformer hybrid deep learning model for agricultural commodities price
    forecasting (2026). *Springer*. https://doi.org/10.1007/s10791-026-09966-2
12. Massa Roldán, R., Reyna Miranda, M. and Gómez Salcido, V. (2022). *Revista Mexicana de
    Economía y Finanzas*, 17(1), e570.
13. Paredes-Garcia, W. J. et al. (2019). Price Forecasting and Span Commercialization
    Opportunities for Mexican Agricultural Products. *Agronomy*, 9(12):826.
14. Valdez, A. (U.S. Meat Export Federation, Mexico). *Equivalencias EEUU y México en
    Nomenclatura de Cortes de Res y Cerdo e Inspección*. Used to map SNIIM cut names to
    USDA IMPS items.
15. Huerta-Leidenz, N. (2013). *Guía de nomenclatura internacional de cortes de carne de
    res para EE.UU. y diferentes países objetivo de la U.S. Meat Export Federation.*
    U.S. Meat Export Federation, Mexico City.
16. Calderón-Alonso, J. et al. (2021). Equivalencia de la nomenclatura y rendimiento de
    cortes primarios de la canal bovina en el noroeste de México. *Nacameh*, 15(1), 30–45.
17. Diebold, F. X. and Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of
    Business & Economic Statistics*, 13(3), 253–263.
18. Harvey, D., Leybourne, S. and Newbold, P. (1997). Testing the equality of prediction
    mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.

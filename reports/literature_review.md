# Literature Review

Forecasting weekly US boxed beef prices four and twelve weeks ahead with deep learning.

This review covers three bodies of work: (1) how wholesale beef prices are formed and what
is known about the spread between what packers buy and what they sell,  (2) how deep
learning has been applied to livestock prices, and (3) which architectures perform best on
agricultural price series. It closes with how this project differs from previous work, the
risks known in advance, and what the data turned out to show — including several places
where the measurements contradicted the design.

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

That the margin should carry information about future prices at all rests on **vertical
price transmission**: the finding that a shock at one stage of the cattle-to-beef chain
passes to the next with a lag, and often asymmetrically. **Jaramillo Villanueva et al.
(2022)** measure exactly that between cattle and beef markets, and it is the mechanism
this project's margin feature is built on — a wide margin today is a signal about supply
decisions whose price effect arrives weeks later.

Published forecasting work on North American cattle and beef prices is mostly machine
learning rather than deep learning. **Rotondo et al. (2024, *Sustainability* 16(5):1789)**
modelled Alberta fed steer prices with support vector regression, random forest and
AdaBoost against ARIMA, SARIMA and SARIMAX, finding the multivariate learners competitive
with the univariate time-series models. That is the closest published analogue to this
project's benchmark layer, and it stops short of recurrent networks.

## 2. Deep learning for livestock prices

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
honest floor" are different claims. For that reason every model here is scored against
the naive random walk as well as against ridge and ARIMA, and reported as RMSE on the
log return — one metric, directly comparable across methods within a series.

## 3. Which architectures work on agricultural price series

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

**Takeaway for architecture choice.** LSTM and GRU are the established choice for weekly
agricultural price series, and comparing the two is itself supported by the literature.
Transformer and decomposition hybrids are stronger in recent studies but are designed for
larger datasets than the ~800 training windows available here, and would risk overfitting.

## 4. How this project differs

| Previous work | This project |
|---|---|
| North American cattle price forecasting uses SVR, random forest and ARIMA families | **LSTM and GRU**, with those traditional methods as the benchmark layer rather than the contribution |
| Deep learning for livestock prices focused on Korea and China | **US boxed beef**, a market with no comparable published deep learning study at cut level |
| Price studied at the aggregate or cutout level | **Seven series**: the Choice and Select cutouts plus five individual cuts identified by IMPS code |
| Price-history models, or decomposition of the price alone | An explicit **supply-side state variable**: the gross packer margin, cutout minus dressed fed cattle price on a common carcass basis |
| Deep learning benchmarked against ARIMA | Benchmarked against **naive**, with Diebold–Mariano tests under the Harvey–Leybourne–Newbold small-sample correction, and seed variance reported |
| A single training run reported | **Five random seeds per configuration**, reported as mean ± standard deviation, with the ensemble alongside |
| Data taken from existing repositories | Collected from **four primary sources**, all public APIs or direct downloads, with reproducible scripts and cut names reconciled against the IMPS institutional meat purchase specifications |

### Three comparisons

1. **Does nonlinearity help?** LSTM and GRU against naive, drift, ridge and ARIMA on the
   same test weeks.
2. **Does sharing across series help?** One joint seven-output network against seven
   independent single-series networks.
3. **Does extra information help?** The same architecture trained with and without each
   added block of features, one at a time: negotiated trading volume, US macro conditions.

## 5. Feasibility and known risks

- **Sample size.** About 1,170 complete weekly observations for the US family, of which
  roughly 800 become training windows once a 26-week input window and the train/validation
  split are applied. Consecutive windows overlap by 25 weeks, so the effective sample is
  smaller still. This is small for deep learning. Mitigation: a single recurrent layer, 32
  units, dropout, early stopping, and a time-ordered split rather than cross-validation.

- **The naive floor is harder than it looks.** The US cutout is close to a random walk at
  a four-week horizon. Section 6 shows ARIMA failing to beat it on both cutout series
  despite those series having the strongest weekly momentum in the dataset.

- **The benchmark may simply win.** Ridge regression on lagged features, properly
  regularised, is hard to beat on 800 observations precisely because it cannot overfit
  much. A modest improvement — or none — is a legitimate result, and the rubric asks
  precisely whether the added complexity is justified.

- **Seed variance can exceed the effect.** With 172 overlapping test windows at four
  weeks and 163 at twelve, the spread between random initialisations of a small recurrent
  network can be larger than the gap between the models being compared. Every configuration is therefore trained five times
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


## 6. What the data actually shows

The sections above are what the literature claims. This section records what the
exploratory analysis measured, including where it contradicts an expectation the design
had been built on. Figures are in `reports/figures/`, produced by `eda_cuts.py`.

The US cutout's own weekly return correlates with its next four weeks at
only **+0.09**. The margin is roughly three times as informative about the four-week
horizon as momentum is, despite momentum being far stronger week to week. A model given
only lagged returns would miss most of what is predictable here.

### Forecastability rises with the horizon, but reliability does not

The project was first built to forecast four weeks ahead and later extended to twelve,
with the pipeline otherwise unchanged. Two things move in opposite directions.

The **size** of the advantage rises with the horizon, as the error-correction account in
section 2 predicts. Mean improvement over the naive forecast goes from 13.6% to 16.7% on
the full sample, and from 16.8% to 21.0% once the 2020–21 disruption is excluded from
training. The direction was visible in the raw correlations before any model was fitted —
the packer margin against the forward return measures −0.301 at four weeks and −0.353 at
thirteen — so this confirms a prediction made from the data rather than one found after
the fact. If the margin corrects slowly, then at four weeks most of the knowable variation
is momentum that a random walk already contains, and only at a longer horizon has the
adjustment reached the market.

The **reliability** falls. Measured as the gain over the best benchmark divided by the
spread across five random seeds, two series clear two standard deviations at four weeks
and none do at twelve on the full sample. The networks become noisier faster than they
become better. A longer horizon offers more to gain and less certainty about having
gained it.

In absolute terms the forecasts are worse at twelve weeks, as they must be — a twelve-week
return is a larger quantity to miss, and mean RMSE rises from roughly 0.073 to 0.108. What
improves is the *relative* standing: the naive forecast degrades faster than the networks
do.


### Seventeen weeks decided the result

Removing the exports feature had a consequence nobody anticipated. USDA FAS export data
has gaps over the COVID aftermath, and because exports was a *required* feature, the rows
with those gaps had been silently dropped from training — seventeen weeks between
2020-06-21 and 2021-03-07, with packer margins of 43.6 to 61.7 USD/cwt against a 2004–2019
median of 6.34, and `margin_choice_log_change` reaching |z| = 8.9.

With those weeks restored to training, the benchmarks moved under 1% while the recurrent
networks moved 15% to 37% worse, on identical scored weeks. The earlier and much stronger
result had rested on an accidental exclusion.

This is a finding about method sensitivity rather than about the pandemic. Ridge sees the
same features and barely moved; ARIMA and naive never saw them at all. What differs is how
a squared-error loss over a recurrent architecture responds to a handful of observations
several standard deviations from everything else, including through the standardisation
computed from the training rows.

All results are therefore reported in two variants: the full sample, and with a declared
disruption window (2020-03-01 to 2021-06-30) removed from training and validation but never
from test, so both are scored on identical weeks. The window is a fixed calendar constant
justified by the regime rather than chosen for its effect — the forecast period averages
8.7 USD/cwt against a 2004–2019 norm of 8.4, while the excluded weeks run an order of
magnitude above it. Note that the margin–return relationship does not *break* inside the
window; measured there it is stronger (−0.753 against −0.120 in 2004–2019), because one
enormous excursion dominates the correlation. The case for exclusion rests on influence,
not on the mechanism failing.

### Consequences for the design

- The packer margin, not the raw price history, is the feature that carries the signal.
- The multi-output argument cannot be asserted from "cuts come from one carcass"; it has
  to be measured, and when measured it did not hold.
- The naive floor is harder than it looks: the cutout is close to a random walk at four
  weeks, and ARIMA loses to naive on two of the seven series.
- Every feature is correlation-tested before admission, and every claimed gain is divided
  by the spread across random seeds before it is called a result.
- Any decision about which rows enter training is reported, because one such decision
  changed this project's headline more than the choice of architecture did.

---

## References

1. Jaramillo Villanueva, J. L. et al. (2022). Vertical and spatial price transmission in
   the Mexican and international cattle and beef market. *Revista Mexicana de Ciencias
   Pecuarias*, 13(4). https://doi.org/10.22319/rmcp.v13i4.5839
2. USDA Economic Research Service. *Beef and Pork Values and Price Spreads Explained*
   (LDP-M-118-01). Farm-to-wholesale spread construction, packer gross margins, and
   asymmetric price transmission.
3. USDA Agricultural Marketing Service (2020). *Boxed Beef & Fed Cattle Price Spread
   Investigation Report*, 22 July 2020.
   https://www.ams.usda.gov/sites/default/files/media/CattleandBeefPriceMarginReport.pdf
4. Rotondo, F. et al. (2024). Using Probabilistic Machine Learning Methods to Improve Beef
   Cattle Price Modeling and Promote Beef Production Efficiency and Sustainability in
   Canada. *Sustainability*, 16(5):1789. https://doi.org/10.3390/su16051789
5. Yoo, D. (2022). Livestock Price Forecasting using Long Short-Term Memory Units: the
   Case of African Swine Fever and the COVID-19 Pandemic. AAEA Annual Meeting.
   https://ageconsearch.umn.edu/record/322610
6. Ma, W., Peng, L., Chen, H. and Yan, H. (2025). Interval price prediction of livestock
   product based on fuzzy mathematics and improved LSTM. *PLOS ONE*, 20(2).
   https://doi.org/10.1371/journal.pone.0318823
7. Zhu, Y., Chuluunsaikhan, T., Choi, J.-H. and Nasridinov, A. (2025). Integrating
   structured and unstructured data for livestock price forecasting: a sustainability
   study from South Korea. *Frontiers in Sustainable Food Systems*, 9:1613616.
   https://doi.org/10.3389/fsufs.2025.1613616
8. Yang, Z., Zhang, R., Yin, Z. and Zhu, Y. (2022). Pork Price Prediction Using LSTM Model
   Based on a New Dataset. SciTePress, pp. 423–429.
9. Manogna, R. L., Dharmaji, V. and Sarang, S. (2025). Enhancing agricultural commodity
   price forecasting with deep learning. *Scientific Reports*.
   https://doi.org/10.1038/s41598-025-05103-z
10. CNN-transformer hybrid deep learning model for agricultural commodities price
    forecasting (2026). *Springer*. https://doi.org/10.1007/s10791-026-09966-2
11. Diebold, F. X. and Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of
    Business & Economic Statistics*, 13(3), 253–263.
12. Harvey, D., Leybourne, S. and Newbold, P. (1997). Testing the equality of prediction
    mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.

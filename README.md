# FCRL-energy: fair and sustainable allocation of energy support under price shocks

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![Energy support run](../../actions/workflows/energy-support.yml/badge.svg)](../../actions/workflows/energy-support.yml)

**Dimitrios Sargiotis** · School of Civil Engineering, National Technical University of Athens · [ORCID 0009-0006-3644-4561](https://orcid.org/0009-0006-3644-4561) · [users.ntua.gr/dims](https://users.ntua.gr/dims/)

This repository develops frameworks introduced in

> Sargiotis, D. (2026). *AI with Integrity: Shaping Ethical, Transparent, and Equitable Urban Futures*. Springer Nature Switzerland. Chapter 3, Ethically Governed AI Frameworks for Fair, Transparent, and Sustainable Urban Resource Management. https://doi.org/10.1007/978-3-032-31133-7

into a runnable decision system for one of the most pressing problems in European cities: how to share a limited energy support budget among regions so that the households least able to heat or cool their homes are protected first.

| Framework in the book | Role in this repository | Module |
|---|---|---|
| Fairness-Constrained Multi-Agent Reinforcement Learning for Urban Resource Allocation (FCRL-MARL) | Fairness-constrained allocation; every region acts as an agent with a shared policy competing for one budget | `policy.py`, `pipeline.py` |
| Sustainability-Constrained Reinforcement Learning (SC-RL) | Penalty on support beyond need, with unmet need treated as the human comfort cost | `policy.py` |
| Explainable Reinforcement Learning for Urban Resource Management (XRL-URM) | Plain-language statement for every regional allocation | `pipeline.py` |
| Accountability-Constrained Reinforcement Learning (AC-RL), fairness-constrained redress | Emergency reserve paid after the month to the regions left with the lowest coverage | `policy.py`, `pipeline.py` |

The book sets out the architecture and its mathematical core: resource allocation subject to a fairness disparity constraint $`D_{\text{fair}}(f_w(X),S) \le \tau`$, a multi-agent value function that balances each agent's reward against environmental impact,

```math
V_{\text{MARL}}(s_t,a_t) = \sum_{k=1}^{K} \mathbb{E}\left[ R_k(s_t,a_t) - \lambda E_k(s_t,a_t) \right],
```

a sustainability-adjusted value that weighs efficiency against human comfort, an explainability layer that makes each decision traceable, and redress mechanisms that correct adverse outcomes. Building on that foundation, this repository applies the frameworks to energy support under price shocks and extends them in five operational directions, described below.

## Why it matters

Energy poverty is now one of the sharpest social consequences of high prices and falling purchasing power. In 2024, 19.0% of people in Greece and in Bulgaria could not keep their homes adequately warm, the highest shares in the EU, against 9.2% across the Union (Eurostat, February 2026). Among people below the poverty line in Greece the share was 43.6%. In Southern Europe the problem also returns in summer, when households cannot afford to cool their homes during heatwaves.

Governments answer with heating and cooling allowances, but the budget is fixed in advance while need depends on how cold the winter or how hot the summer turns out to be and on how energy prices move. A rule that looks fair on paper, such as an equal amount per resident at risk of poverty, can leave households in colder regions, or in regions hit hardest by a price rise, with a much smaller share of their need covered than households elsewhere. This repository makes that trade-off measurable and offers an allocation rule that protects the worst-off regions first, guarantees a minimum, stays within the budget and explains every decision.

## What it does

1. **Measures need.** For every region and month, the energy need of residents at risk of poverty is estimated from heating degree days in winter, cooling degree days in summer and the national energy price index.
2. **Allocates before the month begins.** The budget is set and shared using only information available at the end of the previous month: normal weather for the season, last month's conditions and the latest prices.
3. **Guarantees a minimum.** Every region receives a guaranteed share of its need, sized so that forecast errors break the guarantee only at a chosen low rate, and the size of the guarantee adapts automatically after price shocks.
4. **Shares the rest fairly.** A policy shared by all regions shares the remaining budget, trained separately for winter and summer to maximise the welfare of the worst covered first, to keep the gap between regions within a limit and to avoid paying above need.
5. **Keeps an emergency reserve.** A small part of the budget, at most 10%, is held back and paid at the end of the month, raising the regions left with the lowest coverage first, so that a cold spell or heatwave that no forecast foresaw does not leave a region behind.
6. **Selects its settings on a crisis year.** The reserve and the guaranteed minimum are chosen separately for winter and summer on 2022, the year of the energy crisis, and only if they lower neither total coverage nor the coverage of the worst-off regions.
7. **Compares with the rules in use.** The same budget is shared by an equal amount per resident, an equal amount per resident at risk of poverty, last year's need and forecast need, alongside a hindsight benchmark that knows the future.
8. **Tests resilience.** Every rule is re-run under an unexpected 30% rise in energy prices, a 20% cut in the budget and both together.
9. **Explains every allocation.** Each region receives a statement in plain language: how much it received, how much came from the guarantee and from the reserve, what raised or lowered its share and how much of its need was covered.

## Operational extensions

The book defines the frameworks and their layers. The extensions below build on that foundation to carry it into operational use for energy support.

**1. Decisions taken before need is known.** Need for region $`r`$ in month $`t`$ and its forecast at the end of month $`t-1`$ are

```math
n_{r,t} = V_r \left( D_{r,t} + d_0 \right) p_{c(r),t}, \qquad \hat{n}_{r,t} = V_r \left( \bar{D}_{r,m(t)} + d_0 \right) p_{c(r),t-1},
```

where $`V_r`$ is the number of residents at risk of poverty, $`D_{r,t}`$ the heating or cooling degree days, $`\bar{D}_{r,m(t)}`$ their normal value for the calendar month, $`d_0`$ a base need and $`p_{c,t}`$ the national energy price index. The budget is a fixed share of forecast need, $`B_t = \beta \sum_r \hat{n}_{r,t}`$, and coverage is $`c_{r,t} = \min(a_{r,t}, n_{r,t}) / n_{r,t}`$.

**2. A guaranteed minimum that holds under price shocks.** Each region receives a guaranteed part of its forecast need and the rest is shared by the policy of the multi-agent framework:

```math
a_{r,t} = \phi\, q_t\, \hat{n}_{r,t} + \Big( B_t - \sum_j \phi\, q_t\, \hat{n}_{j,t} \Big) \frac{\hat{n}_{r,t}\, e^{\theta^\top z_{r,t}}}{\sum_j \hat{n}_{j,t}\, e^{\theta^\top z_{j,t}}}.
```

Whenever need does not exceed $`q_t`$ times its forecast, region $`r`$ is covered at least at the level $`\phi`$. The bound $`q_t`$ is a conformal quantile of past need-to-forecast ratios, and its level is adjusted after every month by adaptive conformal inference (Gibbs and Candès, 2021),

```math
\alpha_{t+1} = \alpha_t + \gamma \left( \delta - \mathrm{err}_t \right),
```

so that the long-run share of region-months in which the guarantee is missed tends to the target $`\delta`$ even when prices shift abruptly. If the budget cannot afford the guarantee in a given month, it is scaled down and the shortfall is reported.

**3. Prioritarian fairness with an explicit sustainability term.** The book's reward and environmental penalty $`R_k - \lambda E_k`$ and its fairness constraint $`D_{\text{fair}} \le \tau`$ are made concrete for energy support. The shared policy weights $`\theta`$ are trained on past seasons to maximise

```math
\frac{1}{T} \sum_t \left[ \sum_r \frac{V_r}{\sum_j V_j}\, u(c_{r,t}) - \lambda\, \frac{\sum_r \left( a_{r,t} - n_{r,t} \right)^{+}}{B_t} \right] - \rho\, \frac{1}{T} \sum_t \left( D_{\text{fair},t} - \tau \right)_{+}^{2}, \qquad u(c) = \frac{c^{1-\eta} - 1}{1 - \eta},
```

where $`D_{\text{fair},t} = \max_r c_{r,t} - \min_r c_{r,t}`$. With inequality aversion $`\eta = 2`$, a gain in coverage counts more the lower a region's coverage already is. The sustainability term discourages support above need, which would fund extra consumption rather than protect households, while unmet need is the comfort cost that the welfare term minimises. The policy uses five public indicators per region: the share at risk of poverty, the share aged 65 or over, last month's heating or cooling need relative to normal, the energy price change over the last three months, and the region's past tendency to need more than forecast.

**4. An emergency reserve that redresses the largest shortfalls first.** A share $`\rho`$ of each month's budget is held back. Once the month's degree days and prices are known, it is paid to the regions with the lowest coverage, raising them to a common level $`L_t`$:

```math
a^{\text{final}}_{r,t} = a_{r,t} + \left( L_t\, n_{r,t} - a_{r,t} \right)^{+}, \qquad \sum_r \left( L_t\, n_{r,t} - a_{r,t} \right)^{+} = \rho B_t, \quad L_t \le 1.
```

This is a water-filling rule: no region is topped up before every region below it has been raised to the same coverage, so the reserve gives priority to the worst-off in the strict sense of the maximin principle. Any part of the reserve not needed to cover need in full is returned unspent. The reserve is the operational form of the fairness-constrained redress mechanism of the accountability framework: when an allocation decided in advance proves inadequate, the shortfall is corrected, and the correction is recorded.

**5. Validation on the crisis year, stress tests and explanations.** For each season, the reserve is chosen on the 2022 energy crisis as the share that maximises the coverage of the worst-off tenth of regions without lowering total coverage relative to forecast-proportional sharing. The guaranteed minimum $`\phi`$ is then the largest that the budget could afford in every month and that keeps both total coverage and worst-off coverage within one percentage point. Every rule is stress-tested on the test seasons, and every allocation is accompanied by a statement that decomposes it into the guaranteed part, the reserve, the forecast part and the effect of each indicator.

## Data

All data are open and downloaded from the Eurostat API when the workflow runs, then cached for a month.

| Quantity | Eurostat dataset | Level |
|---|---|---|
| Heating and cooling degree days, monthly | `nrg_chddr2_m` (NUTS 3 average used where a NUTS 2 series is missing) | NUTS 2 |
| Population on 1 January | `demo_r_d2jan` | NUTS 2 |
| Share aged 65 or over | `demo_r_pjanind2` | NUTS 2 |
| At-risk-of-poverty rate | `ilc_li41` (national `ilc_li02` where no regional value exists) | NUTS 2 |
| Price index of electricity, gas and other fuels (CP045, 2015 = 100) | `prc_hicp_midx` | Country |

The real-data configuration covers the NUTS 2 regions of Greece, Bulgaria, Spain, Portugal and Italy, excluding the outermost regions. Training uses the seasons 2015 to 2021, validation 2022 and testing 2023 to 2025. Winter runs from November to March and summer from June to September. Annual indicators enter with a one-year lag, as they would be available to a ministry at the time. For each country the chain uses the NUTS version, 2021 or 2024, with the most complete series and records the choice.

## Outputs

Each run writes an Excel report with the sheets Summary, Strategies, Compared with other rules, Selection on validation, Emergency reserve, Stress tests, Allocation statements, Policy weights, Regional coverage, Guarantee, Training history, Data coverage and Data sources. Coverage measures are reported with 95% confidence intervals from a bootstrap over regions.

## Running it on GitHub

Nothing needs to be run locally.

- **CI** runs the test suite on Python 3.10 to 3.13 on every push and pull request, including a test of the full real-data path against a simulated Eurostat API.
- **Energy support run** runs on every push to `main`, every Monday at 04:41 UTC and on demand (Actions → Energy support run → Run workflow). It runs the synthetic demonstration and the real-data configuration, writes the main tables to the run summary and uploads both reports as the `energy-support-reports` artifact.

## Results on synthetic data

The synthetic configuration has 40 regions in four countries, weather anomalies that persist from month to month, a warming trend in summer and an energy price shock in 2022. With a budget of 60% of forecast need, the chain selected a guaranteed minimum of 20% in winter and 30% in summer and an emergency reserve of 10% in both seasons. On the five test seasons:

| Rule | Share of vulnerable need covered | Coverage of the worst-off tenth of regions | Lowest regional coverage | Gap between 90th and 10th percentile | Support above need |
|---|---|---|---|---|---|
| Equal amount per resident | 0.609 | 0.328 | 0.220 | 0.589 | 3.1% |
| Equal amount per resident at risk of poverty | 0.616 | 0.364 | 0.288 | 0.550 | 2.0% |
| Proportional to last year's need | 0.619 | 0.385 | 0.299 | 0.371 | 1.4% |
| Proportional to forecast need | 0.623 | 0.448 | 0.383 | 0.287 | 0.9% |
| **FCRL-energy** | **0.628** | **0.557** | **0.487** | **0.107** | **0.1%** |
| Hindsight benchmark (not implementable) | 0.628 | 0.592 | 0.526 | 0.003 | 0.0% |

Against an equal amount per resident at risk of poverty, FCRL-energy raised the coverage of the worst-off tenth of regions by 0.193 (95% CI 0.162 to 0.205) and narrowed the gap between regions by 0.443 (95% CI 0.348 to 0.479), with no loss in total coverage (+0.012, 95% CI −0.005 to 0.029). Against forecast-proportional sharing, the gain for the worst-off was 0.108 (95% CI 0.093 to 0.112), three quarters of the distance to the hindsight benchmark. The guaranteed minimum held in every test region-month, and under a 30% price rise combined with a 20% budget cut it held in 87.4% of region-months, against 81.4% under the equal amount per resident at risk of poverty.

When the budget is smaller than need, total coverage changes only through support paid above need. Fairer sharing therefore costs almost nothing in total coverage, which the results confirm: the gains for the worst-off came without any measurable loss in total coverage.

## Findings on real data

The first run on real data used an earlier version of the chain, with a single guaranteed minimum for both seasons and no reserve. All 63 NUTS 2 regions of Greece, Bulgaria, Spain, Portugal and Italy were retrieved with regional poverty rates and degree days; for Portugal, the NUTS 2024 regions were complete. The run produced three findings that shaped the present version:

1. **Targeting by forecast need protects the worst-off far better than the rules in use.** On the test seasons 2023 to 2025, the earlier version raised the coverage of the worst-off tenth of regions from 0.322 under an equal amount per resident at risk of poverty to 0.481 (+0.159, 95% CI 0.124 to 0.184), narrowed the gap between regions by 0.495 and cut support paid above need from 5.0% to 0.3% of the budget, with no loss in total coverage. The equal amount per resident at risk of poverty covered the full need in some regions with mild climates and less than a quarter of it in others. Most of the gain came from sharing in proportion to forecast need; the learned policy added a smaller but consistent gain of 0.014 (95% CI 0.010 to 0.018).
2. **Early-summer heatwaves are the main source of unprotected need.** Cooling need in June exceeded the bound derived from past forecast errors in 43% of regions in 2022, 35% in 2024 and 62% in 2025, while winter misses stayed mostly below 10%. The lowest regional coverage was 0.32 to 0.41 in summer against 0.57 to 0.63 in winter.
3. **A guarantee sized for summer uncertainty consumes most of the budget.** The adaptive bound on need relative to forecast reached 1.6 to 2.0 in summer against about 1.2 in winter, so a meaningful summer guarantee left little budget for the policy, and the earlier selection rule, which looked only at average coverage of the worst-off, kept no guarantee at all. The present version therefore selects the guarantee separately for each season, as the largest that harms neither total nor worst-off coverage, and adds the emergency reserve for shortfalls that no forecast can foresee.

### Present version

On the same 63 regions and test seasons, the present version selected on the 2022 crisis year a guaranteed minimum of 40% of need in winter and 20% in summer, and an emergency reserve of 10% of the budget in both seasons.

| Rule | Share of vulnerable need covered | Coverage of the worst-off tenth of regions | Lowest regional coverage | Gap between 90th and 10th percentile | Support above need |
|---|---|---|---|---|---|
| Equal amount per resident at risk of poverty | 0.581 | 0.322 | 0.244 | 0.625 | 5.0% |
| Proportional to forecast need | 0.611 | 0.467 | 0.282 | 0.152 | 0.2% |
| **FCRL-energy** | **0.611** | **0.539** | **0.409** | **0.071** | **0.1%** |
| Hindsight benchmark (not implementable) | 0.612 | 0.554 | 0.442 | 0.018 | 0.0% |

- **The worst-off regions gain most.** Against an equal amount per resident at risk of poverty, the coverage of the worst-off tenth of regions rose by 0.217 (95% CI 0.185 to 0.239), the lowest coverage of any region in any season rose from 0.244 to 0.409, and the gap between regions fell by 0.554 (95% CI 0.485 to 0.583), with no loss in total coverage (+0.030, 95% CI −0.012 to 0.075). Against forecast-proportional sharing, the gain for the worst-off was 0.072 (95% CI 0.058 to 0.082), which closes 83% of the distance to the hindsight benchmark.
- **The guarantee held.** The guaranteed minimum was met in every test region-month, and the forecast bound was exceeded in 9.1% of region-months against a target of 10%.
- **Summer remains the harder season.** The lowest summer coverage rose from 0.28 to 0.37 under forecast-proportional sharing to 0.41 to 0.50, but stayed below the winter range of 0.62 to 0.69. The reserve did most of this work: on validation, coverage of the worst-off tenth in summer rose from 0.343 without a reserve to 0.418 with it.
- **The reserve acts as a settlement, not only as an emergency fund.** It was fully used in every test season and reached between 57 and 63 of the 63 regions, because water-filling lifts every region below the common level. On validation, worst-off coverage rose with every increase in the reserve up to the 10% cap, so a larger reserve would protect the worst-off further, at the cost of paying more of the support after the month.
- **Resilience under shocks.** Under a 30% price rise combined with a 20% budget cut, the guarantee held in 84.8% of region-months against 72.9% under the equal amount per resident at risk of poverty, and the lowest regional coverage was 0.250 against 0.150.
- **What the learned policy adds.** The weights learned for the five indicators change a region's claim by 5% or less for a one-standard-deviation difference in any indicator. Most of the gain comes from sharing by forecast need, from the guaranteed minimum and from the reserve; the learned policy fine-tunes the shares within them.

In policy terms, the results suggest that an energy allowance should be shared in proportion to the forecast energy need of residents at risk of poverty rather than by a flat amount per resident, should guarantee every region a minimum share of its need, and should keep a small reserve that is settled after each month in favour of the regions left furthest behind.

## Limitations

- Need is estimated from degree days, prices and the number of residents at risk of poverty. It is an index of the energy cost of keeping a home warm or cool, not a measured household expenditure, and it does not capture differences in housing quality or heating systems within a region.
- The price index is national, so regional differences in tariffs are not captured.
- The at-risk-of-poverty rate measures low income relative to each country's median and is published with a lag; national values are used where regional values are missing.
- The guarantee holds on average over time at the chosen miss rate; a single month can miss it, particularly at the start of an unexpected shock, before the adaptive bound has caught up.
- The emergency reserve is paid after the month, once degree days and prices are known, so it reaches households later than the advance payment. The reserve is capped at 10% of the budget so that at least 90% of the support is paid in advance; the cap is a policy choice in the configuration.
- The privacy-preserving federated layer of FCRL-MARL is not needed here because all inputs are public regional aggregates; it becomes relevant when household-level data are used.
- The chain supports, and does not replace, the decisions of public authorities.

## Data attribution

Degree days, population, poverty and price data: © European Union, Eurostat; reuse is authorised provided the source is acknowledged (Commission Decision 2011/833/EU). Monthly degree days are re-published by Eurostat from the Joint Research Centre's AGRI4CAST Resources Portal. Region label points: © EuroGeographics for the administrative boundaries, Eurostat GISCO.

## References

- Gibbs, I. and Candès, E. (2021). Adaptive conformal inference under distribution shift. *Advances in Neural Information Processing Systems 34*.
- Eurostat (2026). 9.2% of EU population struggled to keep their home warm. News release, 2 February 2026.

## Citation

Please cite both the software and the book; GitHub's **Cite this repository** button reads `CITATION.cff`.

> Sargiotis, D. (2026). *FCRL-energy: fair and sustainable allocation of energy support under price shocks* (Version 1.0.0) [Computer software]. https://github.com/uomoathens/fcrl-energy

> Sargiotis, D. (2026). *AI with Integrity: Shaping Ethical, Transparent, and Equitable Urban Futures*. Springer Nature Switzerland. https://doi.org/10.1007/978-3-032-31133-7

## Licence

MIT. See `LICENSE`.

## [15.1.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v15.0.0...v15.1.0-rc.1) (2026-05-29)


### Features

* **config:** add warm_start_lags flag to ConfigMulti and ConfigEntsoe ([3a80cec](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/3a80cec84198d67ee4d4be7d41dcdc379a7b4059))

## [15.0.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v14.0.0...v15.0.0) (2026-05-29)


### ⚠ BREAKING CHANGES

* The `include_poly_features` config field and the
`select_exogenous_features(..., include_poly_features=...)` parameter are
removed. Use `poly_features_degree` (>= 2 to enable interactions) plus the
new `max_poly_features` cap instead.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

### Features

* replace include_poly_features with poly_features_degree + MI-capped interactions ([53b2de1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/53b2de102fa7daf266fa95d890a9ef10da607ef6))


### Documentation

* regenerate reference pages to match current source ([b1d2813](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/b1d2813f3d3d9aec211bbc6ec53760a57f487d17))
* remove orphaned forecaster.utils sidebar nav entry ([cde60bd](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/cde60bd1066b8ddef59c6c626525667b47f38ced))

## [15.0.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v14.0.0...v15.0.0-rc.1) (2026-05-29)


### ⚠ BREAKING CHANGES

* The `include_poly_features` config field and the
`select_exogenous_features(..., include_poly_features=...)` parameter are
removed. Use `poly_features_degree` (>= 2 to enable interactions) plus the
new `max_poly_features` cap instead.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

### Features

* replace include_poly_features with poly_features_degree + MI-capped interactions ([53b2de1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/53b2de102fa7daf266fa95d890a9ef10da607ef6))


### Documentation

* regenerate reference pages to match current source ([b1d2813](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/b1d2813f3d3d9aec211bbc6ec53760a57f487d17))
* remove orphaned forecaster.utils sidebar nav entry ([cde60bd](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/cde60bd1066b8ddef59c6c626525667b47f38ced))

## [14.0.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.4.3...v14.0.0) (2026-05-28)


### ⚠ BREAKING CHANGES

* **manager:** package_prediction(), get_model_prediction(),
load_iteration(), get_last_model(), load_forecasters(), and setup_logging()
now raise typed exceptions on failure instead of returning sentinel values.
The corresponding test assertions and docstring examples are updated.

Full suite: 1574 passed, 1 skipped (unchanged from baseline).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

### Bug Fixes

* **forecaster:** tighten utils.py bugs, dedupe __all__, move to top ([ceb00e2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ceb00e24bc9c82322a8dfe1f425156390e70f4bc))
* keep data before configurator in root __init__.py ([7e14322](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/7e14322849b093d272b9b8c50c160f890540c59a)), closes [#252](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/252)
* **manager:** raise typed exceptions instead of silent returns ([53c9f8f](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/53c9f8fb0c9646ded11fcaed16ad580c51802131))


### Documentation

* **stats:** add reference pages for stats.stationarity and stats.spectral ([55ea780](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/55ea780f779340edcef4bd17a0b8d428d0023695)), closes [#248](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/248)
* **stats:** add Stats section to sidebar navigation ([1771ca1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/1771ca19d68d1ee2123339730b102b7edd950fda))


### Code Refactoring

* **forecaster:** clean up utils.py code smells ([4467cb2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/4467cb2c93e7e0ee10be9782af74f436586b2e6c))
* **forecaster:** demote utils.py to internal aggregator ([b1ddd8d](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/b1ddd8d9df2b16767a645eb2a2f8d05ca25f7b6b))

## [14.0.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.4.4-rc.2...v14.0.0-rc.1) (2026-05-28)


### ⚠ BREAKING CHANGES

* **manager:** package_prediction(), get_model_prediction(),
load_iteration(), get_last_model(), load_forecasters(), and setup_logging()
now raise typed exceptions on failure instead of returning sentinel values.
The corresponding test assertions and docstring examples are updated.

Full suite: 1574 passed, 1 skipped (unchanged from baseline).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

### Bug Fixes

* **manager:** raise typed exceptions instead of silent returns ([53c9f8f](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/53c9f8fb0c9646ded11fcaed16ad580c51802131))

## [13.4.4-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.4.4-rc.1...v13.4.4-rc.2) (2026-05-27)


### Code Refactoring

* **forecaster:** clean up utils.py code smells ([4467cb2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/4467cb2c93e7e0ee10be9782af74f436586b2e6c))
* **forecaster:** demote utils.py to internal aggregator ([b1ddd8d](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/b1ddd8d9df2b16767a645eb2a2f8d05ca25f7b6b))

## [13.4.4-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.4.3...v13.4.4-rc.1) (2026-05-27)


### Bug Fixes

* **forecaster:** tighten utils.py bugs, dedupe __all__, move to top ([ceb00e2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ceb00e24bc9c82322a8dfe1f425156390e70f4bc))

## [13.4.3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.4.2...v13.4.3) (2026-05-26)


### Bug Fixes

* **curate_weather:** include gap size and index range in shape warning ([75a70b5](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/75a70b50e1d65fcab95335265a0ab2a1cf35a451))

## [13.4.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.4.1...v13.4.2) (2026-05-25)


### Bug Fixes

* **weather:** trim raw weather_df to [start, cov_end] before curate_weather ([98ecd4f](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/98ecd4f7540f46328ff7232f3e2d05dde21b31d8))

## [13.4.2-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.4.1...v13.4.2-rc.1) (2026-05-25)


### Bug Fixes

* **weather:** trim raw weather_df to [start, cov_end] before curate_weather ([98ecd4f](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/98ecd4f7540f46328ff7232f3e2d05dde21b31d8))

## [13.4.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.4.0...v13.4.1) (2026-05-24)


### Bug Fixes

* **predictor:** populate metrics_future_one_day in build_prediction_package ([e733388](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/e7333886fa6345047b98f5a647269f71cd9f24c8))

## [13.4.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.3.0...v13.4.0) (2026-05-24)


### Features

* **processing:** honour on_weather_failure='skip' in n2n_predict_with_covariates ([e6517ea](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/e6517eaddaa2e6f9860278e8ce4d613de5057579))
* **stats,manager,preprocessing:** port chag25a primitives (ADF, periodogram, deterministic process, retrain cadence) ([a038202](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/a038202ae8cf5f817fd47c6073bf2cc69cba41f0))


### Bug Fixes

* DateTime ([f6f45ba](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/f6f45ba2df36b064e2a372915d6e1b6e78f51594))

## [13.4.0-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.4.0-rc.1...v13.4.0-rc.2) (2026-05-24)


### Features

* **processing:** honour on_weather_failure='skip' in n2n_predict_with_covariates ([e6517ea](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/e6517eaddaa2e6f9860278e8ce4d613de5057579))

## [13.4.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.3.1-rc.1...v13.4.0-rc.1) (2026-05-24)


### Features

* **stats,manager,preprocessing:** port chag25a primitives (ADF, periodogram, deterministic process, retrain cadence) ([a038202](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/a038202ae8cf5f817fd47c6073bf2cc69cba41f0))

## [13.3.1-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.3.0...v13.3.1-rc.1) (2026-05-23)


### Bug Fixes

* DateTime ([f6f45ba](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/f6f45ba2df36b064e2a372915d6e1b6e78f51594))

## [13.3.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.2.0...v13.3.0) (2026-05-23)


### Features

* **configurator:** add test_data_loader hook for ground-truth test data ([25f18b1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/25f18b1e11715dc3299605aae76822eddec793d2))

## [13.2.0-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.2.0-rc.1...v13.2.0-rc.2) (2026-05-23)


### Features

* **configurator:** add test_data_loader hook for ground-truth test data ([25f18b1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/25f18b1e11715dc3299605aae76822eddec793d2))

## [13.2.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.1.0...v13.2.0-rc.1) (2026-05-23)


### Features

* opt-in fail-safe handling for Open-Meteo failures ([70e310d](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/70e310d5bf8cfcca0e594489fc1aa5a5d91df35c))

## [13.1.0-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.1.0-rc.1...v13.1.0-rc.2) (2026-05-23)


### Features

* opt-in fail-safe handling for Open-Meteo failures ([70e310d](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/70e310d5bf8cfcca0e594489fc1aa5a5d91df35c))

## [13.1.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.0.2...v13.1.0) (2026-05-23)


### Features

* **configurator:** add auto_save_models, data_frame_name, number_folds ([0d439fa](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/0d439facaf583dd8e4dcf2cd8b6d1c671ebc6e3d))


### Code Refactoring

* step 1.5 ([600b9cc](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/600b9cc8d06afd728eb3dbc6d970aeeb4afd6c69))

## [13.1.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.0.3-rc.1...v13.1.0-rc.1) (2026-05-23)


### Features

* **configurator:** add auto_save_models, data_frame_name, number_folds ([0d439fa](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/0d439facaf583dd8e4dcf2cd8b6d1c671ebc6e3d))

## [13.0.3-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.0.2...v13.0.3-rc.1) (2026-05-22)


### Code Refactoring

* step 1.5 ([600b9cc](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/600b9cc8d06afd728eb3dbc6d970aeeb4afd6c69))

## [13.0.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.0.1...v13.0.2) (2026-05-22)


### Bug Fixes

* step1 ([a92e23d](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/a92e23df082f3c7998d0229dc61b93ab065a1005))

## [13.0.2-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.0.1...v13.0.2-rc.1) (2026-05-22)


### Bug Fixes

* step1 ([a92e23d](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/a92e23df082f3c7998d0229dc61b93ab065a1005))

## [13.0.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.0.0...v13.0.1) (2026-05-20)


### Bug Fixes

* DemoConfig -> ConfigDemo, Data_classes, demo10 ([85ecf74](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/85ecf74b5b838f54461f1dea2b53d874ab8da7c3))


### Documentation

* add aggregation to n2n_explained and demo_100 ([ab46277](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ab462770cf65898b67fb648e0e077d67b3e3fbe0))
* add demo100 step-by-step walkthrough of n2n_predict_with_covariates ([1b51e3f](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/1b51e3f69e7df5d7979e0e89f0aa157056a7e25c))
* all n2n ([ef897e1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ef897e16b995eaacc10ca9d2889720f4accc9023))
* cleanup tutorials and tasks ([b3e712e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/b3e712e68ba98de35b3e239cd55a69258450c48e))
* completed Examples: ([4465045](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/44650456ad629f97b65b9e90391a68a4ce36cf09))
* **manager:** convert manager/logger.py Examples to live {python} cells ([aba3204](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/aba32043dadc96446bc6541c4061eb1957b74d98))
* update introductions ([e72a7b9](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/e72a7b919c7d690af30c2fa753d2b5683936b592))

## [13.0.1-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v13.0.0...v13.0.1-rc.1) (2026-05-20)


### Bug Fixes

* DemoConfig -> ConfigDemo, Data_classes, demo10 ([85ecf74](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/85ecf74b5b838f54461f1dea2b53d874ab8da7c3))


### Documentation

* add aggregation to n2n_explained and demo_100 ([ab46277](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ab462770cf65898b67fb648e0e077d67b3e3fbe0))
* add demo100 step-by-step walkthrough of n2n_predict_with_covariates ([1b51e3f](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/1b51e3f69e7df5d7979e0e89f0aa157056a7e25c))
* all n2n ([ef897e1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ef897e16b995eaacc10ca9d2889720f4accc9023))
* cleanup tutorials and tasks ([b3e712e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/b3e712e68ba98de35b3e239cd55a69258450c48e))
* completed Examples: ([4465045](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/44650456ad629f97b65b9e90391a68a4ce36cf09))
* **manager:** convert manager/logger.py Examples to live {python} cells ([aba3204](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/aba32043dadc96446bc6541c4061eb1957b74d98))
* update introductions ([e72a7b9](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/e72a7b919c7d690af30c2fa753d2b5683936b592))

## [13.0.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v12.0.0...v13.0.0) (2026-05-20)


### ⚠ BREAKING CHANGES

* `spotforecast2_safe.manager._parse_bool` is removed.
Import `parse_bool` from `spotforecast2_safe.utils.parse` instead.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

### Documentation

* **utils:** convert convert_to_utc + cpe Examples to live {python} cells ([1095ded](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/1095dedfa22a79509b9ab30cdebe02d4caa20b33))


### Code Refactoring

* move manager/tools.py → utils/parse.py and make parse_bool public ([8b52c0b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/8b52c0bcc472ca40a26d21e74d883dd8b49fcdab))

## [12.0.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v11.0.0...v12.0.0) (2026-05-19)


### ⚠ BREAKING CHANGES

* move utils/data_transform + utils/forecaster_config into preprocessing/

### Code Refactoring

* move utils/data_transform + utils/forecaster_config into preprocessing/ ([8276463](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/82764636010823b5c88bb40730a9d0f61a2b953f))

## [11.0.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v10.0.0...v11.0.0) (2026-05-19)


### ⚠ BREAKING CHANGES

* promote RollingFeatures module from _rolling to rolling

### Code Refactoring

* promote RollingFeatures module from _rolling to rolling ([2fea5d7](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/2fea5d75e104ac43345ac3ba41116067c67eac7d))

## [10.0.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v9.0.0...v10.0.0) (2026-05-19)


### ⚠ BREAKING CHANGES

* move utils/validation.py → preprocessing/checking.py
* split model_selection into splitter + backtesting, relocate preprocessing/split

### Code Refactoring

* move utils/validation.py → preprocessing/checking.py ([9ce56cf](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/9ce56cfc07c1bfff20ea80605a0331a6e3224d1e))
* split model_selection into splitter + backtesting, relocate preprocessing/split ([750e73a](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/750e73a724794b6f07e841e54577f9108373d331))

## [9.0.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v8.0.0...v9.0.0) (2026-05-19)


### ⚠ BREAKING CHANGES

* **configurator:** `spotforecast2_safe.manager.configurator` no longer
exists. Use `spotforecast2_safe.configurator` instead. The package-root
`spotforecast2_safe.Config` alias is unchanged.

The sibling `spotforecast2` repo ships its parallel import updates in a
separate coordinated PR.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

### Documentation

* **preprocessing:** convert curate_data Examples to {python} cells and fix types ([190e176](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/190e176425e01e2a05af30e181a383768a6dae81))
* **preprocessing:** convert imputation Examples to self-contained {python} cells ([2d3a4b6](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/2d3a4b65a95740026a0b13eb289bfc0074a785d0))
* **preprocessing:** convert outlier Examples to self-contained {python} cells ([c1a1f19](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/c1a1f191931804ff19fda7a5a03e7fd868e1be5b))


### Code Refactoring

* **configurator:** promote configurator out of manager subpackage ([9ff97c3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/9ff97c36dfcf73954779d5866e621f7e75e5f6fc))

## [9.0.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v8.0.0...v9.0.0-rc.1) (2026-05-19)


### ⚠ BREAKING CHANGES

* **configurator:** `spotforecast2_safe.manager.configurator` no longer
exists. Use `spotforecast2_safe.configurator` instead. The package-root
`spotforecast2_safe.Config` alias is unchanged.

The sibling `spotforecast2` repo ships its parallel import updates in a
separate coordinated PR.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

### Documentation

* **preprocessing:** convert curate_data Examples to {python} cells and fix types ([190e176](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/190e176425e01e2a05af30e181a383768a6dae81))
* **preprocessing:** convert imputation Examples to self-contained {python} cells ([2d3a4b6](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/2d3a4b65a95740026a0b13eb289bfc0074a785d0))
* **preprocessing:** convert outlier Examples to self-contained {python} cells ([c1a1f19](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/c1a1f191931804ff19fda7a5a03e7fd868e1be5b))


### Code Refactoring

* **configurator:** promote configurator out of manager subpackage ([9ff97c3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/9ff97c36dfcf73954779d5866e621f7e75e5f6fc))

## [8.0.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v7.0.2...v8.0.0) (2026-05-19)


### ⚠ BREAKING CHANGES

* **manager:** callers of the underscore-prefixed persistence helpers
must update to the new public names.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

### Code Refactoring

* **manager:** promote persistence helpers to public API and modernize docstrings ([3f35f97](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/3f35f978a96fbfc1259d54d408352579f361d413))

## [8.0.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v7.0.2...v8.0.0-rc.1) (2026-05-19)


### ⚠ BREAKING CHANGES

* **manager:** callers of the underscore-prefixed persistence helpers
must update to the new public names.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

### Code Refactoring

* **manager:** promote persistence helpers to public API and modernize docstrings ([3f35f97](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/3f35f978a96fbfc1259d54d408352579f361d413))

## [7.0.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v7.0.1...v7.0.2) (2026-05-19)


### Documentation

* **processing:** convert n2n_predict_with_covariates Examples to {python} cells ([175de2e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/175de2effbe2a8763b50601e07382e80fc73b1a4))
* **tasks:** embed runnable n_to_1_with_covariates example in task page ([e9f1e81](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/e9f1e815860bc53b644d63110310beb0023cee24))


### Code Refactoring

* **processing:** remove backward-compat underscore aliases from n2n_predict_with_covariates ([86b92c5](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/86b92c527a98cf93099027bc87e9e005859c0e07))

## [7.0.2-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v7.0.1...v7.0.2-rc.1) (2026-05-19)


### Code Refactoring

* **processing:** remove backward-compat underscore aliases from n2n_predict_with_covariates ([86b92c5](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/86b92c527a98cf93099027bc87e9e005859c0e07))

## [7.0.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v7.0.0...v7.0.1) (2026-05-17)


### Bug Fixes

* **manager:** add NaN/length validation to calculate_metrics and update tests ([45feffc](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/45feffc388d74be2969c8833331246006ddf4a2c))


### Documentation

* **forecaster:** convert method docstring Examples to {python} cells ([ce0b97a](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ce0b97ac9a905a0eb300d7a67bf6a05c1e748e41))
* **forecaster:** convert module and class docstring Examples to {python} cells ([6e49fbb](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/6e49fbb4e627984279ffa6ae19e97ac22b1df1ea))
* **forecaster:** fix broken Examples and require {python} cells in CLAUDE.md ([7bb1ab7](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/7bb1ab7a7eb776c8272eec005ef064ae75eb6295))


### Code Refactoring

* **manager:** rename metrics.py to demo_metrics.py to disambiguate from forecaster.metrics ([56c15fa](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/56c15fae685e0d8122877da9ba974e0c692d8b8f))
* **security:** extract mask_estimator into spotforecast2_safe.security.masking ([66d0b94](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/66d0b947be926142124f4fc7c2342f0f1204bf43))

## [7.0.1-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v7.0.1-rc.1...v7.0.1-rc.2) (2026-05-17)


### Code Refactoring

* **security:** extract mask_estimator into spotforecast2_safe.security.masking ([66d0b94](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/66d0b947be926142124f4fc7c2342f0f1204bf43))

## [7.0.1-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v7.0.0...v7.0.1-rc.1) (2026-05-17)


### Bug Fixes

* **manager:** add NaN/length validation to calculate_metrics and update tests ([45feffc](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/45feffc388d74be2969c8833331246006ddf4a2c))


### Documentation

* **forecaster:** convert method docstring Examples to {python} cells ([ce0b97a](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ce0b97ac9a905a0eb300d7a67bf6a05c1e748e41))
* **forecaster:** convert module and class docstring Examples to {python} cells ([6e49fbb](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/6e49fbb4e627984279ffa6ae19e97ac22b1df1ea))
* **forecaster:** fix broken Examples and require {python} cells in CLAUDE.md ([7bb1ab7](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/7bb1ab7a7eb776c8272eec005ef064ae75eb6295))


### Code Refactoring

* **manager:** rename metrics.py to demo_metrics.py to disambiguate from forecaster.metrics ([56c15fa](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/56c15fae685e0d8122877da9ba974e0c692d8b8f))

## [7.0.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v6.0.0...v7.0.0) (2026-05-17)


### ⚠ BREAKING CHANGES

* **forecaster:** spotforecast2_safe.manager.models is removed.
Replace imports of spotforecast2_safe.manager.models.forecaster_recursive_*
with spotforecast2_safe.forecaster.wrappers. The three wrapper classes
keep the same names, signatures, and inheritance relationships.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

### Code Refactoring

* **forecaster:** consolidate manager/models into forecaster/wrappers ([05cfe16](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/05cfe16d507b624f02f83029fa3295665cc739d3))

## [7.0.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v6.0.0...v7.0.0-rc.1) (2026-05-17)


### ⚠ BREAKING CHANGES

* **forecaster:** spotforecast2_safe.manager.models is removed.
Replace imports of spotforecast2_safe.manager.models.forecaster_recursive_*
with spotforecast2_safe.forecaster.wrappers. The three wrapper classes
keep the same names, signatures, and inheritance relationships.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

### Code Refactoring

* **forecaster:** consolidate manager/models into forecaster/wrappers ([05cfe16](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/05cfe16d507b624f02f83029fa3295665cc739d3))

## [6.0.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v5.0.1...v6.0.0) (2026-05-17)


### ⚠ BREAKING CHANGES

* spotforecast2_safe.manager.exo,
spotforecast2_safe.manager.exo.weather and the
spotforecast2_safe.manager.get_weather_features re-export are removed.
Import get_weather_features from spotforecast2_safe.weather.
spotforecast2_safe.weather.weather_client is renamed to
spotforecast2_safe.weather.client; WeatherClient and WeatherService
remain importable from the top-level spotforecast2_safe.weather package.
spotforecast2_safe.calendar._common is deleted; to_utc_timestamp is now
available from spotforecast2_safe.utils.convert_to_utc.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

### Documentation

* formatting ([481104c](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/481104ce921f2842c83bcd5f9a3c014ab3f5e86e))


### Code Refactoring

* consolidate weather into weather/ package, drop manager/exo/ ([ff15185](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ff15185a54163fafabeac77c2a056a091e4d5592))

## [6.0.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v5.0.1...v6.0.0-rc.1) (2026-05-17)


### ⚠ BREAKING CHANGES

* spotforecast2_safe.manager.exo,
spotforecast2_safe.manager.exo.weather and the
spotforecast2_safe.manager.get_weather_features re-export are removed.
Import get_weather_features from spotforecast2_safe.weather.
spotforecast2_safe.weather.weather_client is renamed to
spotforecast2_safe.weather.client; WeatherClient and WeatherService
remain importable from the top-level spotforecast2_safe.weather package.
spotforecast2_safe.calendar._common is deleted; to_utc_timestamp is now
available from spotforecast2_safe.utils.convert_to_utc.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

### Documentation

* formatting ([481104c](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/481104ce921f2842c83bcd5f9a3c014ab3f5e86e))


### Code Refactoring

* consolidate weather into weather/ package, drop manager/exo/ ([ff15185](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ff15185a54163fafabeac77c2a056a091e4d5592))

## [5.0.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v5.0.0...v5.0.1) (2026-05-15)


### Bug Fixes

* **docs:** pass on_missing='ffill_bfill' in load_timeseries examples ([e298a68](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/e298a68f20c6546a6f7f55c95717bd89e2d5e837)), closes [#153](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/153)

## [5.0.1-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v5.0.0...v5.0.1-rc.1) (2026-05-15)


### Bug Fixes

* **docs:** pass on_missing='ffill_bfill' in load_timeseries examples ([e298a68](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/e298a68f20c6546a6f7f55c95717bd89e2d5e837)), closes [#153](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/153)

## [5.0.0-rc.3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v5.0.0-rc.2...v5.0.0-rc.3) (2026-05-15)


### Bug Fixes

* **docs:** pass on_missing='ffill_bfill' in load_timeseries examples ([e298a68](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/e298a68f20c6546a6f7f55c95717bd89e2d5e837)), closes [#153](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/153)


### Documentation

* **contributing:** require full local verification before pushing ([fd78870](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/fd788705e3dd3f8e5121d540e3948d3f7eaee548))

## [5.0.0-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v5.0.0-rc.1...v5.0.0-rc.2) (2026-05-15)


### Features

* data cleanup ([cf54cfa](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/cf54cfa74cd152f77d6a2ba1513958b53e04736d))


### Bug Fixes

* **docs:** unbreak demo_loader render by removing \n escapes in {python} blocks ([cf6fa98](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/cf6fa98f261e878accf4abbcbdcc5fbf18fc7ab7))

## [5.0.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v4.0.0...v5.0.0-rc.1) (2026-05-15)


### ⚠ BREAKING CHANGES

*   spotforecast2_safe.holiday and spotforecast2_safe.manager.exo.calendar
  no longer exist. Import from spotforecast2_safe.calendar instead.
  Manager and manager.exo no longer re-export the calendar functions.

  data.fetch_data.fetch_holiday_data remains as a public helper.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

### Code Refactoring

* consolidate holiday and manager.exo.calendar into calendar/ ([53573f5](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/53573f50d1454f538f5fd79ec30206ce0bbdb09f))

## [4.0.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v3.0.4...v4.0.0) (2026-05-15)


### ⚠ BREAKING CHANGES

* spotforecast2_safe.manager.datasets is gone. Import
DemoConfig and load_actual_combined from spotforecast2_safe.data instead.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

### Code Refactoring

* move DemoConfig and load_actual_combined to data/ ([ba512ce](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ba512ce6e5c78462102959f0f2da70492ddefe98))

## [3.0.4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v3.0.3...v3.0.4) (2026-05-15)


### Documentation

* docstring mv to __init.py__ ([11a98e0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/11a98e08770b2004e1864e8e9b3509a770ead2a7))
* **downloader:** convert entsoe docstring examples to Quarto {python} blocks ([c58b685](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/c58b685dcf94d7d7a8b1d889464413da7a02419d))


### Code Refactoring

* **downloader:** raise on persistent ENTSO-E failure, guard NaT, extract retry constants ([06e5f84](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/06e5f846eeac3b948b2be8a5652196b22a5d29e0))
* **holiday:** promote generate_holiday into top-level holiday package ([bf77263](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/bf77263d959d162433ceb6ef7a506c81562b8ec7))

## [3.0.4-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v3.0.4-rc.1...v3.0.4-rc.2) (2026-05-15)


### Documentation

* **downloader:** convert entsoe docstring examples to Quarto {python} blocks ([c58b685](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/c58b685dcf94d7d7a8b1d889464413da7a02419d))


### Code Refactoring

* **downloader:** raise on persistent ENTSO-E failure, guard NaT, extract retry constants ([06e5f84](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/06e5f846eeac3b948b2be8a5652196b22a5d29e0))

## [3.0.4-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v3.0.3...v3.0.4-rc.1) (2026-05-15)


### Documentation

* docstring mv to __init.py__ ([11a98e0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/11a98e08770b2004e1864e8e9b3509a770ead2a7))


### Code Refactoring

* **holiday:** promote generate_holiday into top-level holiday package ([bf77263](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/bf77263d959d162433ceb6ef7a506c81562b8ec7))

## [3.0.3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v3.0.2...v3.0.3) (2026-05-15)


### Documentation

* weather ([914316d](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/914316d5ef695fe93aeba858e0ac6e07c83ef0dc))


### Code Refactoring

* **weather:** drop unused timezone arg from _create_fallback / _finalize_df ([5870f6b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/5870f6b939092b9a7eaea32c079a64af885742d7))
* **weather:** modernise typing, urllib3 import, add NaT guard ([1362fca](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/1362fca022b53b6baa15e000938d2b4c831f8e63))

## [3.0.3-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v3.0.2...v3.0.3-rc.1) (2026-05-15)


### Documentation

* weather ([914316d](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/914316d5ef695fe93aeba858e0ac6e07c83ef0dc))


### Code Refactoring

* **weather:** drop unused timezone arg from _create_fallback / _finalize_df ([5870f6b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/5870f6b939092b9a7eaea32c079a64af885742d7))

## [3.0.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v3.0.1...v3.0.2) (2026-05-13)


### Bug Fixes

* **weather:** repair broken docstring examples in WeatherClient ([921852a](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/921852a79b80d859d1d6a6b08f5683d529c94449))


### Documentation

* regenerate stale quartodoc reference pages ([93102f4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/93102f4f3ebdaa833694a85be57a723b6e68215a))

## [3.0.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v3.0.0...v3.0.1) (2026-05-13)


### Code Refactoring

* **manager:** drop unused _BASE_LOGRECORD_ATTRS in logger ([b29f876](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/b29f8760c0ca1e227958f795c367cc0a740742d1))

## [3.0.1-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v3.0.0...v3.0.1-rc.1) (2026-05-13)


### Code Refactoring

* **manager:** drop unused _BASE_LOGRECORD_ATTRS in logger ([b29f876](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/b29f8760c0ca1e227958f795c367cc0a740742d1))

## [3.0.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v2.0.1...v3.0.0) (2026-04-26)


### ⚠ BREAKING CHANGES

* **preprocessing:** `LinearlyInterpolateTS()` no longer silently fills
endpoint NaN; callers that relied on the implicit fill must opt in
via `on_missing="ffill_bfill"` (or `"passthrough"` if they want to
handle residuals themselves). In-tree callers updated:
- preprocessing/imputation.py: switched to `passthrough` so residual
  NaN flows into the existing post-imputation WARNING path instead of
  being silently bridged.
- manager/models/forecaster_recursive_model.py: 8 callsites switched
  to `ffill_bfill` to preserve the fit-time invariant that no NaN
  reaches the estimator.

Tests: existing tests updated for the new defaults; new tests cover
the raise path, the ffill_bfill leading-NaN behaviour, the
passthrough residual-survival behaviour, and the invalid-value error.
quartodoc reference page regenerated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

### Features

* **preprocessing:** add on_missing contract to LinearlyInterpolateTS ([3c17cd6](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/3c17cd64ebccb6d3b27dec1cef3c264a05bc64a8))


### Bug Fixes

* **ci:** prune SBOM and sigstore bundles before PyPI upload ([27e7dbc](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/27e7dbc55651a564db4820da8c16a87c2ef85fc8))


### Documentation

* **bart26hDE:** compliance figure + longtable + title-bold + section reorder ([16763ac](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/16763acf764d5812cab55c05d4edd3e10e71b900))

## [3.0.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v2.0.1-rc.2...v3.0.0-rc.1) (2026-04-26)


### ⚠ BREAKING CHANGES

* **preprocessing:** `LinearlyInterpolateTS()` no longer silently fills
endpoint NaN; callers that relied on the implicit fill must opt in
via `on_missing="ffill_bfill"` (or `"passthrough"` if they want to
handle residuals themselves). In-tree callers updated:
- preprocessing/imputation.py: switched to `passthrough` so residual
  NaN flows into the existing post-imputation WARNING path instead of
  being silently bridged.
- manager/models/forecaster_recursive_model.py: 8 callsites switched
  to `ffill_bfill` to preserve the fit-time invariant that no NaN
  reaches the estimator.

Tests: existing tests updated for the new defaults; new tests cover
the raise path, the ffill_bfill leading-NaN behaviour, the
passthrough residual-survival behaviour, and the invalid-value error.
quartodoc reference page regenerated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

### Features

* **preprocessing:** add on_missing contract to LinearlyInterpolateTS ([3c17cd6](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/3c17cd64ebccb6d3b27dec1cef3c264a05bc64a8))


### Documentation

* **bart26hDE:** compliance figure + longtable + title-bold + section reorder ([16763ac](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/16763acf764d5812cab55c05d4edd3e10e71b900))

## [2.0.1-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v2.0.1-rc.1...v2.0.1-rc.2) (2026-04-22)


### Bug Fixes

* **ci:** prune SBOM and sigstore bundles before PyPI upload ([27e7dbc](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/27e7dbc55651a564db4820da8c16a87c2ef85fc8))

## [2.0.1-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v2.0.0...v2.0.1-rc.1) (2026-04-22)


### Bug Fixes

* **ci:** disable @semantic-release/github PR/issue comment hooks ([aed3762](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/aed37622af294c70e98c1c4d5bd9c4da18eb9323)), closes [#123](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/123)

## [2.0.0-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v2.0.0-rc.1...v2.0.0-rc.2) (2026-04-22)


### Bug Fixes

* **ci:** disable @semantic-release/github PR/issue comment hooks ([aed3762](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/aed37622af294c70e98c1c4d5bd9c4da18eb9323)), closes [#123](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/123)

## [2.0.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v1.1.0...v2.0.0-rc.1) (2026-04-22)


### ⚠ BREAKING CHANGES

* introduces the audit-log schema as a public
compliance contract (EU AI Act Art. 12, IEC 62443-4-2 SAR 6.1,
IEC 61508-3 sec. 7.4.7). Downstream log consumers must now validate
against audit_log_schema.json.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>

### Features

* **ci:** add prohibited-dependencies guard job and compliance test ([d22b0af](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/d22b0af33b09c16092e5ec610ced2eca795ee9a2))
* introduce versioned audit-log JSON schema with CI gate ([74541d1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/74541d1f406ef4b49d94b2eea375a2c34c60c516))
* **manager:** pin LightGBM determinism flags to eliminate multi-core histogram drift ([e7b6b63](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/e7b6b6360ee7772758890fd7ccff829b8f470a72))
* require STRIDE updates on attack-surface pull requests ([186c02b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/186c02b6bf1d72a26b2dd5a197423c2f9167cd9e))
* ship signed CycloneDX SBOM with project CPE on each release ([2759f80](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/2759f809ff2ddded74e7644025681e46591bf998))


### Bug Fixes

* **ci:** generate uv.lock before prohibited-deps and pytest jobs ([fa4bd60](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/fa4bd60f94e6ebc41f9f54364d07664964e3ffef))
* **ci:** generate uv.lock before release-workflow tests ([6a2ea97](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/6a2ea97a137b74341fe2e8ff231d33301a61475b))
* **tasks:** route demo CSV fetches through get_package_data_home ([1329bd9](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/1329bd967fc3f3f9cbd5e770faf2ec4df5d2d98a))


### Documentation

* frame regulatory gap in English abstract and mirror compliance updates ([13a81dc](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/13a81dc992f44990a86fb49e61c46a0e3040cf26))
* iso/iec added ([15524f7](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/15524f7b727f89cd364dddcbbcf4e78090273930))
* until 9.4 ([6b80fd9](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/6b80fd9ec6f5b8517f8733ab9e0c637fcde7a225))

## [1.1.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v1.0.1...v1.1.0) (2026-04-19)


### Features

* model card and reuse ([f5ff52b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/f5ff52b04f93511acfda49e21856080d21bb2649))


### Documentation

* bart26h ([9a673aa](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/9a673aa3effeed3c30729779e7c4cd8fab82d792))

## [1.0.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v1.0.0...v1.0.1) (2026-04-18)


### Bug Fixes

* **downloader:** narrow ENTSO-E resume fallback to real 'no-data' signals ([d5e2c72](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/d5e2c727864b4fad1cdcdbbf752fa00aa2e666f9))

## [1.0.1-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v1.0.0...v1.0.1-rc.1) (2026-04-18)


### Bug Fixes

* **downloader:** narrow ENTSO-E resume fallback to real 'no-data' signals ([d5e2c72](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/d5e2c727864b4fad1cdcdbbf752fa00aa2e666f9))

## [1.0.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.10...v1.0.0) (2026-04-18)


### ⚠ BREAKING CHANGES

* **data:** callers relying on silent NaN imputation in these
public entry points must now pass `on_missing="ffill_bfill"` or
`fill_missing=True`. See `MODEL_CARD.md` §4 for the migration note.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>

### Features

* **data:** raise on NaN in public data loaders and weather client ([8a207b3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/8a207b3650f6db058f50cb4d2e4d9034ca39c10b))


### Bug Fixes

* **data:** import Literal for OnMissing type alias ([93c7c0a](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/93c7c0add54877661c41cbc6f2d7db08f4fe01e2))
* **weather:** quarantine corrupt parquet cache instead of silent None ([e03d134](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/e03d1347b5ad5136d2bd818b64df085222eb8ee8))

## [1.0.0-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v1.0.0-rc.1...v1.0.0-rc.2) (2026-04-18)


### Bug Fixes

* **weather:** quarantine corrupt parquet cache instead of silent None ([e03d134](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/e03d1347b5ad5136d2bd818b64df085222eb8ee8))

## [1.0.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.10...v1.0.0-rc.1) (2026-04-18)


### ⚠ BREAKING CHANGES

* **data:** callers relying on silent NaN imputation in these
public entry points must now pass `on_missing="ffill_bfill"` or
`fill_missing=True`. See `MODEL_CARD.md` §4 for the migration note.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>

### Features

* **data:** raise on NaN in public data loaders and weather client ([8a207b3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/8a207b3650f6db058f50cb4d2e4d9034ca39c10b))


### Bug Fixes

* **data:** import Literal for OnMissing type alias ([93c7c0a](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/93c7c0add54877661c41cbc6f2d7db08f4fe01e2))

## [0.30.10](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.9...v0.30.10) (2026-03-25)


### Bug Fixes

* cache ([1950750](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/1950750e8760743f7ac04d122c3d273bcfe30621))

## [0.30.10-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.9...v0.30.10-rc.1) (2026-03-25)


### Bug Fixes

* cache ([1950750](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/1950750e8760743f7ac04d122c3d273bcfe30621))

## [0.30.9](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.8...v0.30.9) (2026-03-25)


### Bug Fixes

* fetch_data ([1a70a6f](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/1a70a6f82b9dbeb47ca572786ef6e8f4b6a153fc))

## [0.30.9-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.8...v0.30.9-rc.1) (2026-03-25)


### Bug Fixes

* fetch_data ([1a70a6f](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/1a70a6f82b9dbeb47ca572786ef6e8f4b6a153fc))

## [0.30.8](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.7...v0.30.8) (2026-03-25)


### Bug Fixes

* caching ([8f3e0dc](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/8f3e0dc67f7d8e14c456b599c258c8ccbce28bc3))


### Documentation

* weather ([f7ab994](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/f7ab994fcc2aaaa8e1d7c9523c18e09b66f2138b))

## [0.30.8-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.7...v0.30.8-rc.1) (2026-03-25)


### Bug Fixes

* caching ([8f3e0dc](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/8f3e0dc67f7d8e14c456b599c258c8ccbce28bc3))


### Documentation

* weather ([f7ab994](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/f7ab994fcc2aaaa8e1d7c9523c18e09b66f2138b))

## [0.30.7](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.6...v0.30.7) (2026-03-25)


### Bug Fixes

* tests ([71398da](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/71398da5910cd1b0374421edd9dfc7aac3fd7f98))

## [0.30.7-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.6...v0.30.7-rc.1) (2026-03-25)


### Bug Fixes

* tests ([71398da](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/71398da5910cd1b0374421edd9dfc7aac3fd7f98))

## [0.30.6](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.5...v0.30.6) (2026-03-25)


### Bug Fixes

* .spotforecast2_cache, weather ([47911c1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/47911c13e6fda41613ebdadfaf17f6e913eda4d7))
* .spotforecast2_cache, weather ([95017dd](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/95017dd12f13e845711afd51ce31d728f3be3d64))
* fetch_date weather cache ([6823760](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/682376057a1aac3c4cb5c288e3c9900b9c1aba4a))
* fetch_date weather cache ([dccf6a3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/dccf6a3ce6876a80996db6410c078c2c80d8cd5c))

## [0.30.6-rc.3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.6-rc.2...v0.30.6-rc.3) (2026-03-25)


### Bug Fixes

* fetch_date weather cache ([6823760](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/682376057a1aac3c4cb5c288e3c9900b9c1aba4a))

## [0.30.6-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.6-rc.1...v0.30.6-rc.2) (2026-03-25)


### Bug Fixes

* .spotforecast2_cache, weather ([95017dd](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/95017dd12f13e845711afd51ce31d728f3be3d64))

## [0.30.6-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.5...v0.30.6-rc.1) (2026-03-24)


### Bug Fixes

* fetch_date weather cache ([dccf6a3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/dccf6a3ce6876a80996db6410c078c2c80d8cd5c))

## [0.30.5](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.4...v0.30.5) (2026-03-24)


### Bug Fixes

* weather data in cache ([703663e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/703663eab273c590cf5a194f1b5317fa01fcb1fc))

## [0.30.5-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.4...v0.30.5-rc.1) (2026-03-24)


### Bug Fixes

* weather data in cache ([703663e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/703663eab273c590cf5a194f1b5317fa01fcb1fc))

## [0.30.4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.3...v0.30.4) (2026-03-24)


### Bug Fixes

* runner ([3c8a74d](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/3c8a74d77181bf8ecd5c899efb8a36a1116ab4a5))

## [0.30.4-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.3...v0.30.4-rc.1) (2026-03-24)


### Bug Fixes

* runner ([3c8a74d](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/3c8a74d77181bf8ecd5c899efb8a36a1116ab4a5))

## [0.30.3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.2...v0.30.3) (2026-03-24)


### Bug Fixes

* curate_data reordered ([f4d5e3f](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/f4d5e3fbcb52600245e0cf1260313f5ded6f468f))

## [0.30.3-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.2...v0.30.3-rc.1) (2026-03-24)


### Bug Fixes

* curate_data reordered ([f4d5e3f](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/f4d5e3fbcb52600245e0cf1260313f5ded6f468f))

## [0.30.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.1...v0.30.2) (2026-03-24)


### Bug Fixes

* data_home data_source removed ([2377f2d](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/2377f2d7b3aee1d996bd65fb439ad4354cea040a))

## [0.30.2-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.1...v0.30.2-rc.1) (2026-03-24)


### Bug Fixes

* data_home data_source removed ([2377f2d](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/2377f2d7b3aee1d996bd65fb439ad4354cea040a))

## [0.30.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.0...v0.30.1) (2026-03-24)


### Bug Fixes

* regressor->estimator ([f5a0909](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/f5a0909b2a260eb3652a9c182cac407e91943ce2))

## [0.30.1-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.30.0...v0.30.1-rc.1) (2026-03-24)


### Bug Fixes

* regressor->estimator ([f5a0909](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/f5a0909b2a260eb3652a9c182cac407e91943ce2))

## [0.30.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.29.0...v0.30.0) (2026-03-23)


### Features

* demo10 reduced, full is demo100 ([d3e9390](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/d3e939054517312ab759b30053a35fae86e6f0ec))

## [0.29.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.28.0...v0.29.0) (2026-03-23)


### Features

* create ([04d054a](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/04d054a95cb497d4c53acac2b3ba9f08f5bbba8f))

## [0.29.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.28.0...v0.29.0-rc.1) (2026-03-23)


### Features

* create ([04d054a](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/04d054a95cb497d4c53acac2b3ba9f08f5bbba8f))

## [0.28.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.27.0...v0.28.0) (2026-03-22)


### Features

* imputation updatd ([b28c1ed](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/b28c1ed31b2315aeb73b0003bd428ab41b17e0a2))

## [0.28.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.27.0...v0.28.0-rc.1) (2026-03-22)


### Features

* imputation updatd ([b28c1ed](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/b28c1ed31b2315aeb73b0003bd428ab41b17e0a2))

## [0.27.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.26.0...v0.27.0) (2026-03-22)


### Features

* build_prediction_package ([9f13699](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/9f13699c900fd49c56b6b6f947a2ef376632f07b))
* config_multi extended ([cafce6f](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/cafce6fb525b25d8f1749094c4a94ff36c28396a))
* config_multi with agg_weights ([bb79d48](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/bb79d4841e3e5bdfb74f9b6d22a56644a2248214))
* safe_forecaster ([6de0633](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/6de0633aa8a5ad9d61c2c06d1fe377ced9a29e16))

## [0.27.0-rc.4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.27.0-rc.3...v0.27.0-rc.4) (2026-03-22)


### Features

* safe_forecaster ([6de0633](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/6de0633aa8a5ad9d61c2c06d1fe377ced9a29e16))

## [0.27.0-rc.3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.27.0-rc.2...v0.27.0-rc.3) (2026-03-22)


### Features

* config_multi with agg_weights ([bb79d48](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/bb79d4841e3e5bdfb74f9b6d22a56644a2248214))

## [0.27.0-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.27.0-rc.1...v0.27.0-rc.2) (2026-03-22)


### Features

* build_prediction_package ([9f13699](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/9f13699c900fd49c56b6b6f947a2ef376632f07b))

## [0.27.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.26.0...v0.27.0-rc.1) (2026-03-22)


### Features

* config_multi extended ([cafce6f](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/cafce6fb525b25d8f1749094c4a94ff36c28396a))

## [0.26.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.25.5...v0.26.0) (2026-03-21)


### Features

* exo ([b502cde](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/b502cdecb6d12d31f1908010a182618239b98cab))
* manager ([12e3fd4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/12e3fd47d6bf9a4f986385cc759ea6d615fb6b19))

## [0.26.0-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.26.0-rc.1...v0.26.0-rc.2) (2026-03-21)


### Features

* manager ([12e3fd4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/12e3fd47d6bf9a4f986385cc759ea6d615fb6b19))

## [0.26.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.25.5...v0.26.0-rc.1) (2026-03-21)


### Features

* exo ([b502cde](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/b502cdecb6d12d31f1908010a182618239b98cab))

## [0.25.5](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.25.4...v0.25.5) (2026-03-21)


### Bug Fixes

* multi_conf updated ([9a0647a](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/9a0647afe0501f826667e118ae86e44e6026bdea))

## [0.25.5-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.25.4...v0.25.5-rc.1) (2026-03-21)


### Bug Fixes

* multi_conf updated ([9a0647a](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/9a0647afe0501f826667e118ae86e44e6026bdea))

## [0.25.4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.25.3...v0.25.4) (2026-03-21)


### Bug Fixes

* reset_index accepts timezone ([cb2ac50](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/cb2ac50f00c02b2b9ffbabe44dd95bc83e8c73ec))
* tz error ([55aa7cd](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/55aa7cda0e8156a8cb3f1ade3d272b4649e1be06))

## [0.25.4-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.25.3...v0.25.4-rc.1) (2026-03-21)


### Bug Fixes

* reset_index accepts timezone ([cb2ac50](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/cb2ac50f00c02b2b9ffbabe44dd95bc83e8c73ec))
* tz error ([55aa7cd](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/55aa7cda0e8156a8cb3f1ade3d272b4649e1be06))

## [0.25.3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.25.2...v0.25.3) (2026-03-21)


### Bug Fixes

* reset_index ([82dcbad](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/82dcbadb970c2785f009dac7c902e0b95ac52b60))

## [0.25.3-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.25.2...v0.25.3-rc.1) (2026-03-21)


### Bug Fixes

* reset_index ([82dcbad](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/82dcbadb970c2785f009dac7c902e0b95ac52b60))

## [0.25.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.25.1...v0.25.2) (2026-03-21)


### Bug Fixes

* targets to config_multi ([33ae63a](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/33ae63a8fe265ae04042c1462110f9b7298d6c2c))


### Documentation

* fixed config ([f04ab3c](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/f04ab3c5a328dd28f927705c218f6d8a58439dfa))

## [0.25.2-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.25.1...v0.25.2-rc.1) (2026-03-21)


### Bug Fixes

* targets to config_multi ([33ae63a](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/33ae63a8fe265ae04042c1462110f9b7298d6c2c))

## [0.25.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.25.0...v0.25.1) (2026-03-21)


### Bug Fixes

* fetch_data ([a66d9af](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/a66d9af30458929baa5e61b747353397176cd08b))
* fetch_data ([010c15b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/010c15bdb664ef4e1a51e8ba1556464d8d9267fd))

## [0.25.1-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.25.1-rc.1...v0.25.1-rc.2) (2026-03-21)


### Bug Fixes

* fetch_data ([a66d9af](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/a66d9af30458929baa5e61b747353397176cd08b))

## [0.25.1-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.25.0...v0.25.1-rc.1) (2026-03-21)


### Bug Fixes

* fetch_data ([010c15b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/010c15bdb664ef4e1a51e8ba1556464d8d9267fd))

## [0.25.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.24.0...v0.25.0) (2026-03-20)


### Features

* ConfMulti extended ([0f80417](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/0f80417e9003e690efa91f11ac160a6a7803ad8d))

## [0.25.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.24.0...v0.25.0-rc.1) (2026-03-20)


### Features

* ConfMulti extended ([0f80417](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/0f80417e9003e690efa91f11ac160a6a7803ad8d))

## [0.24.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.23.1...v0.24.0) (2026-03-20)


### Features

* demo10 and 11 daatasets ([2bebfa4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/2bebfa42d0ef6fd8b4b65e3570b0e4c6a0657a5b))

## [0.24.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.23.1...v0.24.0-rc.1) (2026-03-20)


### Features

* demo10 and 11 daatasets ([2bebfa4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/2bebfa42d0ef6fd8b4b65e3570b0e4c6a0657a5b))

## [0.23.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.23.0...v0.23.1) (2026-03-20)


### Bug Fixes

* imputation ([e126562](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/e126562a70919e0cf070b34983b5dbb8241c10de))
* imputation zero weights ([160bf15](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/160bf157b9431d21136073e7e678d276c249f9da))

## [0.23.1-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.23.0...v0.23.1-rc.1) (2026-03-20)


### Bug Fixes

* imputation ([e126562](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/e126562a70919e0cf070b34983b5dbb8241c10de))
* imputation zero weights ([160bf15](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/160bf157b9431d21136073e7e678d276c249f9da))

## [0.23.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.22.0...v0.23.0) (2026-03-18)


### Features

* .from_config() ([7fccf26](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/7fccf26383afec24e0e21784a93ee79ef1d7201c))

## [0.23.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.22.0...v0.23.0-rc.1) (2026-03-18)


### Features

* .from_config() ([7fccf26](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/7fccf26383afec24e0e21784a93ee79ef1d7201c))

## [0.22.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.21.0...v0.22.0) (2026-03-18)


### Features

* multiConf ([013256f](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/013256ff1ae3e8c511bf6ce70792da6dffca8960))

## [0.22.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.21.0...v0.22.0-rc.1) (2026-03-18)


### Features

* multiConf ([013256f](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/013256ff1ae3e8c511bf6ce70792da6dffca8960))

## [0.21.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.20.0...v0.21.0) (2026-03-18)


### Features

* get_model_prediction() ([93fa22b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/93fa22ba8151b56f77d218a41fcfe718eec81c92))

## [0.21.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.20.0...v0.21.0-rc.1) (2026-03-18)


### Features

* get_model_prediction() ([93fa22b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/93fa22ba8151b56f77d218a41fcfe718eec81c92))

## [0.20.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.19.0...v0.20.0) (2026-03-18)


### Features

* remove_duplicate_timestamps ([02ff49a](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/02ff49a6833fc02aa06f0e2785ad6f5381d808e1))


### Documentation

* typo ([3e6744b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/3e6744b1dd7ccfb16a3fa7b7608c541c6a50ec39))

## [0.20.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.19.0...v0.20.0-rc.1) (2026-03-18)


### Features

* remove_duplicate_timestamps ([02ff49a](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/02ff49a6833fc02aa06f0e2785ad6f5381d808e1))


### Documentation

* typo ([3e6744b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/3e6744b1dd7ccfb16a3fa7b7608c541c6a50ec39))

## [0.19.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.18.1...v0.19.0) (2026-03-17)


### Features

* demo03.csv ([04a7fbe](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/04a7fbe3cf52e9e5f13980dbba4717f93f2b96fb))

## [0.19.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.18.1...v0.19.0-rc.1) (2026-03-17)


### Features

* demo03.csv ([04a7fbe](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/04a7fbe3cf52e9e5f13980dbba4717f93f2b96fb))

## [0.18.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.18.0...v0.18.1) (2026-03-17)


### Bug Fixes

* contributing.md ([220274c](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/220274caa80353ab7b1ace2d9918596ae027b524))


### Documentation

* contributing updated and cleanup ([339e6dc](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/339e6dcf0a9af9d6005d65269f22b2d9ce224766))

## [0.18.1-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.18.0...v0.18.1-rc.1) (2026-03-17)


### Bug Fixes

* contributing.md ([220274c](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/220274caa80353ab7b1ace2d9918596ae027b524))


### Documentation

* contributing updated and cleanup ([339e6dc](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/339e6dcf0a9af9d6005d65269f22b2d9ce224766))

## [0.18.0-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.18.0-rc.1...v0.18.0-rc.2) (2026-03-17)


### Bug Fixes

* contributing.md ([220274c](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/220274caa80353ab7b1ace2d9918596ae027b524))


### Documentation

* contributing updated and cleanup ([339e6dc](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/339e6dcf0a9af9d6005d65269f22b2d9ce224766))
* n2n ([ec7dd56](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ec7dd565432d6460de0c6a004a2fc8d18445f44c))
* n2n added ([3db6b84](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/3db6b84918770a9b8ba4c729bb5eb8fdea111708))

## [0.18.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.17.3-rc.1...v0.18.0-rc.1) (2026-03-16)


### Features

* doc completed ([063c9f0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/063c9f05401e1e6190e346b25b862f983e355904))


### Documentation

* tasks ([d8927bb](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/d8927bb7ab1cc472917deec896e5980f6a1fe0d5))

## [0.17.3-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.17.2...v0.17.3-rc.1) (2026-03-16)


### Bug Fixes

* live docs ([a62eeca](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/a62eeca3ff26e03ba6a8c97ead095d8d485acce1))

## [0.17.2-rc.3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.17.2-rc.2...v0.17.2-rc.3) (2026-03-16)


### Bug Fixes

* live docs ([a62eeca](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/a62eeca3ff26e03ba6a8c97ead095d8d485acce1))

## [0.17.2-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.17.2-rc.1...v0.17.2-rc.2) (2026-03-16)


### Bug Fixes

* /var/cache error ([6d267f1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/6d267f1394d835a63f2e4dbd4e120c182dc9e03f))

## [0.17.2-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.17.1...v0.17.2-rc.1) (2026-03-16)


### Bug Fixes

* docs living examples ([3d7f078](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/3d7f07876c14c74758663a6d62827902a9294004))


### Documentation

* imputation explained ([09396dd](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/09396dd8d8da4bd1e04deec1393863a3c3525adf))

## [0.17.1-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.17.1-rc.1...v0.17.1-rc.2) (2026-03-16)


### Bug Fixes

* docs living examples ([3d7f078](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/3d7f07876c14c74758663a6d62827902a9294004))


### Documentation

* imputation explained ([09396dd](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/09396dd8d8da4bd1e04deec1393863a3c3525adf))

## [0.17.1-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.17.0...v0.17.1-rc.1) (2026-03-16)


### Bug Fixes

* doc generation update ([7bdc42e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/7bdc42e978614686f191acf4aa0ef24d9ed9aec3))


### Documentation

* outliers ([a4c9336](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/a4c933619e092d07748ea8353bf9e4c69b86bcf8))

## [0.17.0-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.17.0-rc.1...v0.17.0-rc.2) (2026-03-16)


### Bug Fixes

* doc generation update ([7bdc42e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/7bdc42e978614686f191acf4aa0ef24d9ed9aec3))


### Documentation

* outliers ([a4c9336](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/a4c933619e092d07748ea8353bf9e4c69b86bcf8))

## [0.17.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.16.3-rc.1...v0.17.0-rc.1) (2026-02-24)


### Features

* new dataset demo02.cvs ([ed40bc6](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ed40bc650813392966db98606d96231ed70c7275))
## [0.16.3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.16.2...v0.16.3) (2026-02-23)


### Bug Fixes

* weight_series ([3e8aaf0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/3e8aaf0e0fe27953456c89cfb8bf3f1f2015bb20))

## [0.16.3-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.16.2...v0.16.3-rc.1) (2026-02-23)


### Bug Fixes

* weight_series ([3e8aaf0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/3e8aaf0e0fe27953456c89cfb8bf3f1f2015bb20))

## [0.16.2-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.16.2-rc.1...v0.16.2-rc.2) (2026-02-23)
## [0.16.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.16.1...v0.16.2) (2026-02-23)


### Bug Fixes

* weight_series ([3e8aaf0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/3e8aaf0e0fe27953456c89cfb8bf3f1f2015bb20))
*  Fixed get_missing_weights to return numeric 0/1 weights instead of booleans. ([1b44261](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/1b4426158d84e49d75a1de463b21333f9d9ee58d))


### Documentation

* api completion ([9ec7551](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/9ec75516ca9230724a2086659e9b2608e0e20251))
* fixed reuse ([fbd9c24](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/fbd9c248f5bc4c3df14427ae579f05a947dafd27))
* links fixed for types ([6cffe8c](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/6cffe8c84d4efb0a733c09f78739230476b04939))

## [0.16.2-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.16.1...v0.16.2-rc.1) (2026-02-23)


### Bug Fixes

*  Fixed get_missing_weights to return numeric 0/1 weights instead of booleans. ([1b44261](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/1b4426158d84e49d75a1de463b21333f9d9ee58d))


### Documentation

* api completion ([9ec7551](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/9ec75516ca9230724a2086659e9b2608e0e20251))
* fixed reuse ([fbd9c24](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/fbd9c248f5bc4c3df14427ae579f05a947dafd27))
* links fixed for types ([6cffe8c](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/6cffe8c84d4efb0a733c09f78739230476b04939))

## [0.16.1-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.16.1-rc.1...v0.16.1-rc.2) (2026-02-23)
## [0.16.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.16.0...v0.16.1) (2026-02-22)


### Bug Fixes

*  Fixed get_missing_weights to return numeric 0/1 weights instead of booleans. ([1b44261](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/1b4426158d84e49d75a1de463b21333f9d9ee58d))
* unpinned ([ce6fa79](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ce6fa79698da67601bd4714105f35dc422b0444a))
* workflow identation error ([2d04d4f](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/2d04d4fbbd758f68cc30208c3291262437d1c906))


### Documentation

* links fixed for types ([6cffe8c](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/6cffe8c84d4efb0a733c09f78739230476b04939))
* migrate from MkDocs to Quarto with quartodoc API generation ([8942820](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/8942820273681166c0b88b5fa29beb0bddaa91de))

## [0.16.1-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.16.0...v0.16.1-rc.1) (2026-02-22)


### Bug Fixes

* unpinned ([ce6fa79](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ce6fa79698da67601bd4714105f35dc422b0444a))
* workflow identation error ([2d04d4f](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/2d04d4fbbd758f68cc30208c3291262437d1c906))


### Documentation

* migrate from MkDocs to Quarto with quartodoc API generation ([8942820](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/8942820273681166c0b88b5fa29beb0bddaa91de))

## [0.16.0-rc.4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.16.0-rc.3...v0.16.0-rc.4) (2026-02-22)


### Bug Fixes

* workflow identation error ([2d04d4f](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/2d04d4fbbd758f68cc30208c3291262437d1c906))

## [0.16.0-rc.3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.16.0-rc.2...v0.16.0-rc.3) (2026-02-22)
## [0.16.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.15.3...v0.16.0) (2026-02-20)


### Features

* hardened trainer ([42cd768](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/42cd768b195467f8db32fdd8e63423b18e0357d2))


### Bug Fixes

* unpinned ([ce6fa79](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ce6fa79698da67601bd4714105f35dc422b0444a))


### Documentation

* migrate from MkDocs to Quarto with quartodoc API generation ([8942820](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/8942820273681166c0b88b5fa29beb0bddaa91de))
* version update ([9e34c87](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/9e34c878778242769cc2568bf6c9febc33563413))

## [0.16.0-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.16.0-rc.1...v0.16.0-rc.2) (2026-02-20)


### Bug Fixes

* version update ([9e34c87](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/9e34c878778242769cc2568bf6c9febc33563413))

## [0.16.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.15.3-rc.1...v0.16.0-rc.1) (2026-02-20)


### Features

* hardened trainer ([42cd768](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/42cd768b195467f8db32fdd8e63423b18e0357d2))
## [0.15.3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.15.2...v0.15.3) (2026-02-20)


### Bug Fixes

* cleaup trainer ([f192a22](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/f192a222ccc382e0702e2306f238b8c4c01cddb1))

## [0.15.3-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.15.2...v0.15.3-rc.1) (2026-02-20)


### Bug Fixes

* cleaup trainer ([f192a22](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/f192a222ccc382e0702e2306f238b8c4c01cddb1))

## [0.15.2-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.15.2-rc.1...v0.15.2-rc.2) (2026-02-20)
## [0.15.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.15.1...v0.15.2) (2026-02-20)


### Bug Fixes

* cleaup trainer ([f192a22](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/f192a222ccc382e0702e2306f238b8c4c01cddb1))
* default values ([8c7d15e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/8c7d15ec90ab5de6dbd41a25cbd3c6e46b87ad30))
* forecaster_recursive_model does not use a default end_dev ([f1ddde2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/f1ddde2a9558967f0f945787b0a3507d884ebf11))

## [0.15.2-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.15.1...v0.15.2-rc.1) (2026-02-20)


### Bug Fixes

* default values ([8c7d15e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/8c7d15ec90ab5de6dbd41a25cbd3c6e46b87ad30))
* forecaster_recursive_model does not use a default end_dev ([f1ddde2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/f1ddde2a9558967f0f945787b0a3507d884ebf11))

## [0.15.1-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.15.1-rc.1...v0.15.1-rc.2) (2026-02-20)
## [0.15.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.15.0...v0.15.1) (2026-02-18)


### Bug Fixes

* default values ([8c7d15e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/8c7d15ec90ab5de6dbd41a25cbd3c6e46b87ad30))
* forecaster_recursive_model does not use a default end_dev ([f1ddde2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/f1ddde2a9558967f0f945787b0a3507d884ebf11))
* shape error (_rolling.py) ([e088c8b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/e088c8b78fc8812c80d1ad04524ef7e085a01e23))

## [0.15.1-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.15.0...v0.15.1-rc.1) (2026-02-18)


### Bug Fixes

* shape error (_rolling.py) ([e088c8b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/e088c8b78fc8812c80d1ad04524ef7e085a01e23))

## [0.15.0-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.15.0-rc.1...v0.15.0-rc.2) (2026-02-18)


### Bug Fixes

* shape error (_rolling.py) ([e088c8b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/e088c8b78fc8812c80d1ad04524ef7e085a01e23))
## [0.15.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.14.0...v0.15.0) (2026-02-18)


### Features

* ForecasterRecursiveModel fully functional ([17f39da](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/17f39dabbdb81cd7602be99af2b481cd0c34a754))

## [0.15.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.14.0...v0.15.0-rc.1) (2026-02-18)


### Features

* ForecasterRecursiveModel fully functional ([17f39da](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/17f39dabbdb81cd7602be99af2b481cd0c34a754))

## [0.14.0-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.14.0-rc.1...v0.14.0-rc.2) (2026-02-18)
## [0.14.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.13.2...v0.14.0) (2026-02-18)


### Features

* ForecasterRecursiveModel fully functional ([17f39da](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/17f39dabbdb81cd7602be99af2b481cd0c34a754))
* get and set_params for Forecaster Models ([ba5d85e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ba5d85e16f78470b045c95a30c6e499108e4b8f5))

## [0.14.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.13.2-rc.1...v0.14.0-rc.1) (2026-02-18)


### Features

* get and set_params for Forecaster Models ([ba5d85e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ba5d85e16f78470b045c95a30c6e499108e4b8f5))
## [0.13.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.13.1...v0.13.2) (2026-02-17)


### Bug Fixes

* colnames in datasets demo01.csv ([cedbd8d](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/cedbd8d6fbf9562b4906b7ba8ceaa1fbf9a2b352))
* demo01.csv corrected header ([1d51ec2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/1d51ec234e98d60625074dd80ca511623b1e59d6))

## [0.13.2-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.13.1...v0.13.2-rc.1) (2026-02-17)


### Bug Fixes

* colnames in datasets demo01.csv ([cedbd8d](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/cedbd8d6fbf9562b4906b7ba8ceaa1fbf9a2b352))
* demo01.csv corrected header ([1d51ec2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/1d51ec234e98d60625074dd80ca511623b1e59d6))

## [0.13.1-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.13.1-rc.1...v0.13.1-rc.2) (2026-02-17)
## [0.13.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.13.0...v0.13.1) (2026-02-17)


### Bug Fixes

* colnames in datasets demo01.csv ([cedbd8d](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/cedbd8d6fbf9562b4906b7ba8ceaa1fbf9a2b352))
* demo01.csv corrected header ([1d51ec2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/1d51ec234e98d60625074dd80ca511623b1e59d6))
* tests with Actual ([7c532c2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/7c532c26071174ce98ab2fdee82a95b0754e0c7d))

## [0.13.1-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.13.0...v0.13.1-rc.1) (2026-02-17)


### Bug Fixes

* tests with Actual ([7c532c2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/7c532c26071174ce98ab2fdee82a95b0754e0c7d))

## [0.13.0-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.13.0-rc.1...v0.13.0-rc.2) (2026-02-17)


### Bug Fixes

* tests with Actual ([7c532c2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/7c532c26071174ce98ab2fdee82a95b0754e0c7d))
## [0.13.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.12.2...v0.13.0) (2026-02-16)


### Features

* get_package_data_home (demo01.csv) ([2ef69d7](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/2ef69d76e23c125473587c24ee750ad9dec94307))

## [0.13.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.12.2-rc.1...v0.13.0-rc.1) (2026-02-16)


### Features

* get_package_data_home (demo01.csv) ([2ef69d7](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/2ef69d76e23c125473587c24ee750ad9dec94307))
## [0.12.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.12.1...v0.12.2) (2026-02-15)


### Bug Fixes

* remove unused import ([f46f664](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/f46f66425369635d45e9a56a5797f826455c8fcf))


### Documentation

* Period Configuration Rationale ([449c84b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/449c84b2c9cbbc04cbad81325d8e8cc1fc87867d))

## [0.12.2-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.12.1...v0.12.2-rc.1) (2026-02-15)


### Bug Fixes

* remove unused import ([f46f664](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/f46f66425369635d45e9a56a5797f826455c8fcf))


### Documentation

* Period Configuration Rationale ([449c84b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/449c84b2c9cbbc04cbad81325d8e8cc1fc87867d))

## [0.12.1-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.12.1-rc.1...v0.12.1-rc.2) (2026-02-15)
## [0.12.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.12.0...v0.12.1) (2026-02-15)


### Bug Fixes

* remove unused import ([f46f664](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/f46f66425369635d45e9a56a5797f826455c8fcf))


### Documentation

* Period Configuration Rationale ([449c84b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/449c84b2c9cbbc04cbad81325d8e8cc1fc87867d))
* Period dataclass frozen ([06d2c9e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/06d2c9e5eff745cf461fee3fbb9ffe949ea353fa))

## [0.12.1-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.12.0...v0.12.1-rc.1) (2026-02-15)


### Bug Fixes

* Period dataclass frozen ([06d2c9e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/06d2c9e5eff745cf461fee3fbb9ffe949ea353fa))

## [0.12.0-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.12.0-rc.1...v0.12.0-rc.2) (2026-02-15)


### Bug Fixes

* Period dataclass frozen ([06d2c9e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/06d2c9e5eff745cf461fee3fbb9ffe949ea353fa))
## [0.12.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.11.0...v0.12.0) (2026-02-15)


### Features

* Config (moved from spotforecast2) ([aafaaf3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/aafaaf3eadb958ca8d23e29f023343ae37c71236))

## [0.12.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.11.0...v0.12.0-rc.1) (2026-02-15)


### Features

* Config (moved from spotforecast2) ([aafaaf3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/aafaaf3eadb958ca8d23e29f023343ae37c71236))

## [0.11.0-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.11.0-rc.1...v0.11.0-rc.2) (2026-02-15)
## [0.11.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.10.1...v0.11.0) (2026-02-15)


### Features

* Config (moved from spotforecast2) ([aafaaf3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/aafaaf3eadb958ca8d23e29f023343ae37c71236))
* ForecasterRecursive manager class ([d5ee4dd](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/d5ee4ddd46b9cd92ae62979c63a0151f5eeeaf85))


### Documentation

* entsoe ([107cee3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/107cee3ee28aa37d518f67818528481bd289d8d5))

## [0.11.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.10.1-rc.1...v0.11.0-rc.1) (2026-02-15)


### Features

* ForecasterRecursive manager class ([d5ee4dd](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/d5ee4ddd46b9cd92ae62979c63a0151f5eeeaf85))


### Documentation

* entsoe ([107cee3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/107cee3ee28aa37d518f67818528481bd289d8d5))
## [0.10.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.10.0...v0.10.1) (2026-02-13)


### Bug Fixes

* doctests ([8b67e1a](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/8b67e1a3d029cd92adbe8aa68c417f850986b8af))

## [0.10.1-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.10.0...v0.10.1-rc.1) (2026-02-13)


### Bug Fixes

* doctests ([8b67e1a](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/8b67e1a3d029cd92adbe8aa68c417f850986b8af))

## [0.10.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.9.2...v0.10.0) (2026-02-13)


### Features

* ForecasterRecursiveMultiseries ([e9b3e1e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/e9b3e1e643e797057c4be54a93a1ec168c501535))

## [0.10.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.9.2-rc.1...v0.10.0-rc.1) (2026-02-13)


### Features

* ForecasterRecursiveMultiseries ([e9b3e1e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/e9b3e1e643e797057c4be54a93a1ec168c501535))
## [0.9.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.9.1...v0.9.2) (2026-02-13)


### Bug Fixes

* ForecasterEquivalentDate verified ([69e8f62](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/69e8f62fdd217483a5879ae6fd9e0b6a8071af92))

## [0.9.2-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.9.1...v0.9.2-rc.1) (2026-02-13)


### Bug Fixes

* ForecasterEquivalentDate verified ([69e8f62](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/69e8f62fdd217483a5879ae6fd9e0b6a8071af92))

## [0.9.1-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.9.1-rc.1...v0.9.1-rc.2) (2026-02-13)
## [0.9.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.9.0...v0.9.1) (2026-02-13)


### Bug Fixes

* ForecasterEquivalentDate verified ([69e8f62](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/69e8f62fdd217483a5879ae6fd9e0b6a8071af92))
* predict_bootstrapping ([5fb5ce0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/5fb5ce0fdfb43b0c17833ef56a987ca65505cdda))

## [0.9.1-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.9.0...v0.9.1-rc.1) (2026-02-13)


### Bug Fixes

* predict_bootstrapping ([5fb5ce0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/5fb5ce0fdfb43b0c17833ef56a987ca65505cdda))

## [0.9.0-rc.4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.9.0-rc.3...v0.9.0-rc.4) (2026-02-13)


### Bug Fixes

* predict_bootstrapping ([5fb5ce0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/5fb5ce0fdfb43b0c17833ef56a987ca65505cdda))
* _recursive_predict ([ba54a2e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ba54a2e8973e95b415c1b24aad7880507e60f8b2))
* _recursive_predict_bootstrapping ([59ed543](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/59ed543db4a6489820dba87783f03325585f01f3))
* predict ([26fbdc8](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/26fbdc845d9cb1256649274c0a03be21efe193c9))

## [0.9.0-rc.3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.9.0-rc.2...v0.9.0-rc.3) (2026-02-13)


### Bug Fixes

* ForecasterRecursive ([#29](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/29)) ([179f40b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/179f40b3153260944e331e2970599b4c48bb4476)), closes [#27](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/27) [#28](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/28) [#26](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/26)

## [0.9.0-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.9.0-rc.1...v0.9.0-rc.2) (2026-02-13)


### Bug Fixes

* predict ([26fbdc8](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/26fbdc845d9cb1256649274c0a03be21efe193c9))

## [0.9.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.8.3-rc.4...v0.9.0-rc.1) (2026-02-13)


### Features

* create_predict_X ([85553d6](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/85553d6b26ed27b11b237a6416f3206b2186e35c))

## [0.8.3-rc.4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.8.3-rc.3...v0.8.3-rc.4) (2026-02-13)


### Bug Fixes

* _recursive_predict_bootstrapping ([59ed543](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/59ed543db4a6489820dba87783f03325585f01f3))

## [0.8.3-rc.3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.8.3-rc.2...v0.8.3-rc.3) (2026-02-13)


### Bug Fixes

* _recursive_predict ([ba54a2e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ba54a2e8973e95b415c1b24aad7880507e60f8b2))

## [0.8.3-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.8.3-rc.1...v0.8.3-rc.2) (2026-02-13)


### Bug Fixes

* Develop ([#27](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/27)) ([7c86921](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/7c869217b49f422fff382489014fcb9e87e130e0))
* Develop ([#28](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/28)) ([cf63a97](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/cf63a9710f14ebbd02f7bb6e4a782756b0f9f912))
* ForecasterRecursive ([#26](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/26)) ([d5c4d36](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/d5c4d369cfe649f476e6ffc36c6aeefacd3f05e5))

## [0.8.3-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.8.2...v0.8.3-rc.1) (2026-02-13)


### Bug Fixes

* ForecasterRecursive ([4f524b7](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/4f524b715f5916b529028b7b8729908ac19f1fb5))

## [0.8.2-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.8.2-rc.1...v0.8.2-rc.2) (2026-02-13)
## [0.8.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.8.1...v0.8.2) (2026-02-13)


### Bug Fixes

* ForecasterRecursive ([4f524b7](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/4f524b715f5916b529028b7b8729908ac19f1fb5))
* _rolling.py ([4ab63ed](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/4ab63ed149ccd788bd80d761cc63d546d3bbbfa5))
* rolling ([#25](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/25)) ([c90705c](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/c90705cb75f897c92c981ec44853660c0cccd10b)), closes [#11](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/11)
* rolling features ([#24](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/24)) ([39afa62](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/39afa62885e2ac8c1a4a2c77d491c60040b5d02e)), closes [#11](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/11)

## [0.8.2-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.8.1...v0.8.2-rc.1) (2026-02-13)


### Bug Fixes

* _rolling.py ([4ab63ed](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/4ab63ed149ccd788bd80d761cc63d546d3bbbfa5))

## [0.8.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.8.0...v0.8.1) (2026-02-13)


### Bug Fixes

* unpinned tag ([#23](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/23)) ([3dabf19](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/3dabf19724970fb0360a443d0c41cb85e7e65191))

## [0.8.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.7.0...v0.8.0) (2026-02-13)


### Features

* cpe ([#11](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/11)) ([26a9bf6](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/26a9bf66054beff8622c8a3fdd5298d97621d32d))
* cpe code ([6b0844d](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/6b0844df003e86bfa7026376af7073ef30856680))
* split_ts ([189d5ef](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/189d5ef9d54dadeaa8765127af124cab1bf4f420))
* val and test ([#21](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/21)) ([b4a1075](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/b4a1075aaa842230e960391602ee9ca2a27bf11d)), closes [#11](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/11)
* validates ([#20](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/20)) ([f828d73](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/f828d73aa0ce9a58e432b218cf17007457f58898)), closes [#11](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/11)
* validation ([6255de1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/6255de173d7ec8c94d5e840a404a4e8be1e967de))
* warnings ([7feeeb4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/7feeeb4662ecd410afa6c7eee978398a8252f7c6))


### Documentation

* badges ([d4db607](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/d4db607747dfc8685d09ad8653e5c809432dcc1d))
* contribute ([de263f3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/de263f3a012aec6c3d2d9fba886f24fe664d8289))
* features ([#17](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/17)) ([4ff00c8](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/4ff00c826f19a138bb5dc9503714f7a67fee326d)), closes [#11](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/11)
* README restructured ([791e68e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/791e68e31053f9c4c9dc1a1226924d6a1d8a4ff3))

## [0.8.0-rc.4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.8.0-rc.3...v0.8.0-rc.4) (2026-02-13)


### Features

* validation ([6255de1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/6255de173d7ec8c94d5e840a404a4e8be1e967de))


### Documentation

* badges ([d4db607](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/d4db607747dfc8685d09ad8653e5c809432dcc1d))

## [0.8.0-rc.3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.8.0-rc.2...v0.8.0-rc.3) (2026-02-12)


### Features

* cpe ([#11](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/11)) ([26a9bf6](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/26a9bf66054beff8622c8a3fdd5298d97621d32d))


### Documentation

* contribute ([de263f3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/de263f3a012aec6c3d2d9fba886f24fe664d8289))

## [0.8.0-rc.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.8.0-rc.1...v0.8.0-rc.2) (2026-02-12)


### Features

* cpe code ([6b0844d](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/6b0844df003e86bfa7026376af7073ef30856680))

## [0.8.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.7.0...v0.8.0-rc.1) (2026-02-12)


### Features

* split_ts ([189d5ef](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/189d5ef9d54dadeaa8765127af124cab1bf4f420))
* warnings ([7feeeb4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/7feeeb4662ecd410afa6c7eee978398a8252f7c6))


### Documentation

* README restructured ([791e68e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/791e68e31053f9c4c9dc1a1226924d6a1d8a4ff3))

## [0.7.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.6.0...v0.7.0-rc.1) (2026-02-11)


### Features

* split_ts ([189d5ef](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/189d5ef9d54dadeaa8765127af124cab1bf4f420))
* warnings ([7feeeb4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/7feeeb4662ecd410afa6c7eee978398a8252f7c6))

## [0.6.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.5.0...v0.6.0) (2026-02-11)


### Features

* split_base ([4d9b494](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/4d9b494d28666ddfb1070509ee7f381cbc290caa))

## [0.5.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.4.15...v0.5.0) (2026-02-11)


### Features

* promote pre-release to production ([2540fba](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/2540fba994f6ac2f5e501f624c7ac24718ff61c1))
* warnings ([#5](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/5)) ([954e7ff](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/954e7ffa3ca9cc0e2e0a943f4fe3f82aec440087))

## [0.5.0-rc.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.4.15...v0.5.0-rc.1) (2026-02-11)


### Features

* warnings ([7feeeb4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/7feeeb4662ecd410afa6c7eee978398a8252f7c6))

## [0.4.15](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.4.14...v0.4.15) (2026-02-11)


### Bug Fixes

* Refactor Data Fetching Defaults, no default filename in fetch_data() ([a36d232](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/a36d232726f46653b26d1c4cd26d570772af07b3))

## [0.4.14](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.4.13...v0.4.14) (2026-02-11)


### Bug Fixes

* correct codecov action parameter from file to files ([df427e7](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/df427e71b6ddb4a58631eb9b9dd695bc28e9c7ca))

## [0.4.13](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.4.12...v0.4.13) (2026-02-11)


### Bug Fixes

* copilot ai  improvements ([d875319](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/d875319c6250af7cd7bd849ed6ea942b6603ea97))

## [0.4.12](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.4.11...v0.4.12) (2026-02-11)


### Bug Fixes

* code ([c33d528](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/c33d5284594c613840b7aacda45b19e3da40329c))

## [0.4.11](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.4.10...v0.4.11) (2026-02-11)


### Bug Fixes

* task_safe ([df8a316](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/df8a316f88ea27a46b975bf7633846e4b3f33012))

## [0.4.10](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.4.9...v0.4.10) (2026-02-11)


### Bug Fixes

* reuse ([b7cf00f](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/b7cf00f8d01c419070bada53ef642d29162d6047))

## [0.4.9](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.4.8...v0.4.9) (2026-02-11)


### Bug Fixes

* test with CodeQL ([1ce036e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/1ce036e0c70097e31d4d29cf0712368d6f080155))

## [0.4.8](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.4.7...v0.4.8) (2026-02-11)


### Bug Fixes

* tests (CodeQL) ([28ad2be](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/28ad2be703960f97776c2cb81bb2e0c52b173d3c))

## [0.4.7](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.4.6...v0.4.7) (2026-02-11)


### Bug Fixes

* CodeQL errors ([077e9d2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/077e9d2fb3ce8f8259d88d5192c55c0d3bd3e155))

## [0.4.6](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.4.5...v0.4.6) (2026-02-11)


### Bug Fixes

* reuse compliance ([ee91478](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ee9147811ab20671bd4be6dae165f4fadcada227))

## [0.4.5](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.4.4...v0.4.5) (2026-02-11)


### Bug Fixes

* codeql error (w/o tests) ([fd83b22](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/fd83b222767e8a3fa4c248f1eea37ce104794046))

## [0.4.4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.4.3...v0.4.4) (2026-02-11)


### Bug Fixes

* test_ ([55a2ce3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/55a2ce340b90e400dfb6087d0e91b833cc2cc226))


### Documentation

* Add local CodeQL analysis guide for VS Code ([3cb4c7b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/3cb4c7b6cec326481f8063d8bdc5498af69699fb))
* cleanup ([9b00e9e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/9b00e9e40dfb2337a2968b7dd5c1e00d5888fd09))

## [0.4.3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.4.2...v0.4.3) (2026-02-11)


### Bug Fixes

* initialize output variable with type annotation (CodeQL py/uninitialized-local-variable) ([22245cc](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/22245cce02111e01b1ff3f73b322f44dfd92eece))

## [0.4.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.4.1...v0.4.2) (2026-02-11)


### Bug Fixes

* explicitly convert DataFrames to numpy arrays in fit_predict calls (CodeQL py/hash-unhashable-value) ([a9dc588](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/a9dc588ce62684aaf8aa2aa11a563b0d8c7a74f2))

## [0.4.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.4.0...v0.4.1) (2026-02-11)


### Bug Fixes

* mask sensitive data in logging (CodeQL CWE-312, CWE-532, CWE-359) ([bc8238d](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/bc8238df5f2105fefee7c865a676c8b4f006a07d))

## [0.4.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.3.9...v0.4.0) (2026-02-11)


### Features

* **security:** Improve OpenSSF Scorecard compliance to 8-9/10 ([#1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/issues/1)) ([5c028e2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/5c028e203c1936f7c03ee9a6092c4a171306feb8))


### Documentation

* badges updated ([a180f3a](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/a180f3a89ee07447171642d9967f8ca75411f320))

## [0.3.9](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.3.8...v0.3.9) (2026-02-09)


### Bug Fixes

* data path for entsoe ([4dc3a1d](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/4dc3a1df4fb223e16a087f187635fa5811c96d2a))

## [0.3.8](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.3.7...v0.3.8) (2026-02-09)


### Bug Fixes

* trainer accepts country info ([1c5aef4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/1c5aef4f626c082e1f1090507da65fba297231ed))
* trainer for entsoe ([029f8d6](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/029f8d6388b8c795e112f5770b2e0fb6f78223c0))

## [0.3.7](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.3.6...v0.3.7) (2026-02-09)


### Bug Fixes

* predictor.py ([001c953](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/001c95352fcfcad06ce571789185bfda5f1d9d3d))

## [0.3.6](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.3.5...v0.3.6) (2026-02-09)


### Bug Fixes

* tests trainer ([7cf52ff](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/7cf52ff57f42dfa588038a9c93871601feade361))
* trainer accepts end_dev arg ([cb8099e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/cb8099e707c5e73fa4f1977896f1c23aef342d71))

## [0.3.5](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.3.4...v0.3.5) (2026-02-09)


### Code Refactoring

* exog in _safe completed ([a724fa9](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/a724fa9fb154b3645a7f970652d51fd5acfcc8e7))

## [0.3.4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.3.3...v0.3.4) (2026-02-08)


### Code Refactoring

* task_safe_demo ([a26dbdd](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/a26dbdd124b87f90d98893216efac1fe5e9bf7ac))

## [0.3.3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.3.2...v0.3.3) (2026-02-08)


### Code Refactoring

* new persistence module ([c51e266](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/c51e26614440f445b49fcc0af266ff810b0eaa17))

## [0.3.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.3.1...v0.3.2) (2026-02-08)


### Bug Fixes

* add mkdocs-macros-plugin to optional-dependencies for CI/CD ([9c9dee9](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/9c9dee9c57661c8caa1558e9c45fa1c08f35bd7c))


### Documentation

* downloads ([4e1d90c](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/4e1d90cea1b18120a4de328035130ced9c6aa12d))
* logo ([d5254a4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/d5254a4dc5aaca94c759d91e1dc19720d0ce3d67))

## [0.3.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.3.0...v0.3.1) (2026-02-08)


### Bug Fixes

* reuse ([089b51a](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/089b51a307f051ed15f19d0764aa4c6311ec0c8f))


### Code Refactoring

* tasks mv to src ([6219e67](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/6219e67e9215acbc3626d9c9ea452d7417f0ecd6))

## [0.3.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.2.4...v0.3.0) (2026-02-08)


### Features

* (manager) ([66d4b20](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/66d4b205d85accfd534c213992688e597d571140))

## [0.2.4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.2.3...v0.2.4) (2026-02-07)


### Code Refactoring

* cleanup 0.1.0 ([0460f0a](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/0460f0a15452665b87b6bce20a4a417901ccf578))

## [0.2.3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.2.2...v0.2.3) (2026-02-07)


### Bug Fixes

* formatting (PEP) ([8333269](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/83332690cca1788306f0cf9f35e1a58b2b49a140))
* validation ([52d901c](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/52d901c3bd626de5e271197b07ce7b544d0641c4))

## [0.2.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.2.1...v0.2.2) (2026-02-07)


### Bug Fixes

* init.py ([4a9548b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/4a9548b700bd9e08c20ac8dec415454b534d9ecd))

## [0.2.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.2.0...v0.2.1) (2026-02-07)


### Bug Fixes

* __init__.py ([0278b35](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/0278b35510183d7099c2157e5fdb1cd5b3fb7577))

## [0.2.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.1.0...v0.2.0) (2026-02-07)


### Features

* add model_selection module with backtesting_forecaster and TimeSeriesFold ([8756d01](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/8756d0193b7143654540796518ecc46eed580e5a))
* prediction intervals implemented. extensive refactoring. ([db89da1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/db89da10db786a3b83f25cfe31445e3e98f07513))

## [0.1.0](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.0.7...v0.1.0) (2026-02-07)


### Features

* predict_interval ([d15fb25](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/d15fb2540e542195e2966848cb2ca6389f189ee8))


### Documentation

* (recursive) ([79a5d4d](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/79a5d4da2b931a89b46e8e9204c918cf6c3027c1))
* intro to ForecasterEquivalentDate and ForecasterRecursive ([94313af](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/94313afbe227b5bed76c35045f91e0a5461a402b))

## [0.0.7](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.0.6...v0.0.7) (2026-02-07)


### Documentation

* (MODEL_CARD, VERSION) ([7129013](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/71290137916af665d2c442e06be8237e3b1e7c8c))
* (MODEL_CARD) ([8da8569](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/8da856938ae9fd8f6396f3bf59952e845815b038))
* upddate safe positive list ([fe0d138](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/fe0d1387c6a12f6a97590c2de455a41b6af2e67b))


### Code Refactoring

* version management ([13d2bbe](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/13d2bbef70e00b98ba1f5bfddfa7eb92d137c346))

## [0.0.6](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.0.5...v0.0.6) (2026-02-07)


### Bug Fixes

* stub replaced (check_preprocess_series) ([56c471e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/56c471e0c695b109b65db63126d19631bae4650e))


### Documentation

* minor fix ([1b796ed](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/1b796ed2b7276625d2709d4326bb26e44a533669))
* path corrections (from spotforecast2. -> from spotforecast2_safe.) ([d3fbd68](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/d3fbd682c330e0d7ca1c45f35184ad22301668dc))

## [0.0.5](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.0.4...v0.0.5) (2026-02-06)


### Bug Fixes

* release scripts on github ([939fc38](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/939fc3802548b58757b1663d56906d0ab15004b2))


### Documentation

* README ([114ec88](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/114ec8830e79054a7f2f7d831b07939ebb2d6493))

## [0.0.4](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.0.3...v0.0.4) (2026-02-06)


### Code Refactoring

* tasks and convert_to_utc ([a48f852](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/a48f8527415a1ee9fb6572bc23fbe693fafbd84f))

## [0.0.3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.0.2...v0.0.3) (2026-02-06)


### Bug Fixes

* correct import (outlier.py) ([b4c0e89](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/b4c0e893220dca1890946d2e9e889b81098292cc))

## [0.0.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v0.0.1...v0.0.2) (2026-02-06)


### Bug Fixes

* compliance ([ab2b4e3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/ab2b4e36aa684570723548173e5f1ff3938935d5))
* plot removed (outlier.py) and mv to spotforecast2 (outlier_plot.py) ([7d4520c](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/7d4520c55a2c6426026b021134e7d3ea53c91236))

## 1.0.0 (2026-02-06)


### Features

* first release (not fully safe) ([dbfaa53](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/dbfaa5380b60f8a397c4253795317e74adeafa26))


### Bug Fixes

* **docs:** repair gh-deploy and streamline safe package ([d13a256](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/d13a256f1d2c738120270d6537d83d5eddba72d5))
* first cleanup at *.py level ([6507cb3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/6507cb3ec8f18f5184131c8d3ab4c4da771a41f8))
* fixes for documentation and links ([00af598](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/00af598a296be2ec15595d3bb8d74e17c87cac65))
* Model card ([d31832e](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/d31832ec2980c0a9933af6f5aa291bdd24c95353))
* version ([d893622](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/d893622ad3b7e5d041048b91871577761d375165))


### Documentation

* finalize landing page with safety features and valid links ([9b44ede](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/9b44edec18d2b138bf3b88fef03ddd67ec2f6c69))
* formalize safety-critical identity and add EU AI Act Model Card ([cc29201](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/cc29201ef767477e1af348549c6c505de2a1def8))
* mkdocs complete ([7f2bbf6](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/7f2bbf6c488728860783467ed5bcf0ab9aff30ce))
* safety doc ([fdcbe4b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/fdcbe4b6220cd57a2b3fc492d6b5b5dad52524e8))

## [1.0.3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v1.0.2...v1.0.3) (2026-02-06)


### Bug Fixes

* fixes for documentation and links ([00af598](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/00af598a296be2ec15595d3bb8d74e17c87cac65))


### Documentation

* finalize landing page with safety features and valid links ([9b44ede](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/9b44edec18d2b138bf3b88fef03ddd67ec2f6c69))
* formalize safety-critical identity and add EU AI Act Model Card ([cc29201](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/cc29201ef767477e1af348549c6c505de2a1def8))

## [1.0.2](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v1.0.1...v1.0.2) (2026-02-06)


### Bug Fixes

* **docs:** repair gh-deploy and streamline safe package ([d13a256](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/d13a256f1d2c738120270d6537d83d5eddba72d5))


### Documentation

* safety doc ([fdcbe4b](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/fdcbe4b6220cd57a2b3fc492d6b5b5dad52524e8))

## [1.0.1](https://github.com/sequential-parameter-optimization/spotforecast2-safe/compare/v1.0.0...v1.0.1) (2026-02-06)


### Bug Fixes

* first cleanup at *.py level ([6507cb3](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/6507cb3ec8f18f5184131c8d3ab4c4da771a41f8))


### Documentation

* mkdocs complete ([7f2bbf6](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/7f2bbf6c488728860783467ed5bcf0ab9aff30ce))

## 1.0.0 (2026-02-06)


### Features

* first release (not fully safe) ([dbfaa53](https://github.com/sequential-parameter-optimization/spotforecast2-safe/commit/dbfaa5380b60f8a397c4253795317e74adeafa26))

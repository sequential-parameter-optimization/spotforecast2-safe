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

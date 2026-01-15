# Changelog

## [Unreleased]

### Bug Fixes

- *(agent)* Return training loss and resolve type errors ([3e93034](3e93034f46607db06509e05e9ee2d040997636fe))

### Documentation

- *(readme)* Update file tree with changelog tooling ([d3a39d4](d3a39d421328f44c0ed276c9dcc0f0fa6db1cbd7))
- *(readme)* Update project structure tree ([91d0f47](91d0f474ae98c58f9d647839ed103089b9ab39dd))
- *(readme)* Reorganize structure and expand technical roadmap ([5682652](56826525713d6db7d2f11a471d418ad39eefd941))
- *(readme)* Add install and hook setup instructions ([ff3d233](ff3d2334329d3ab3ce6920ce3f07376856e7b614))
- *(README)* Update configuration table in README ([78a9ec4](78a9ec424bac9f5f426acdb1ab82a4bac07fcc86))
- *(readme)* Add definition table for key execution metrics ([6a48639](6a486395e9eb53b88b7c48cb862c9688a9c58acf))
- *(readme)* Add instructions for optuna hyperparameter tuning ([a30b9d7](a30b9d7119878c92e76d16ec1c73299ffb48e9e9))
- *(readme)* Format make commands as code blocks in table ([d12e63f](d12e63f8ee1c79476f965f8d94d55db39774c43f))
- *(config)* Add custom settings template and organize config files ([cf7f525](cf7f525d945a7d643a8e7ba401209f3b90a30c18))
- *(readme)* Overhaul documentation with setup, docker, and config guides ([903f074](903f074c916cea48275d73403ae2a33d5d4ae27b))

### Features

- *(eval)* Overhaul evaluation with detailed IS metrics and plots ([eddc599](eddc599e9a7c8e5cc35811c44c608dc3c3ffdf00))
- *(optimize)* Implement optuna hyperparameter tuning script ([a0ca836](a0ca8360372fc7c3da6485ed2a73676583e2bde0))
- *(training)* Integrate optuna pruning and reporting into training loop ([c484fd3](c484fd3ef739d23fdcf1442d92cbe05cf1142fc6))
- *(settings)* Add optimization schema and prioritize best_params.yaml ([fa5fcb6](fa5fcb67c3a1eae335901c38386a6d05ac00605d))
- *(config)* Add hyperparameter optimization settings ([7cec0d2](7cec0d295732b627b1fa5a27294580bed9d73dd1))
- *(training)* Implement DQN training loop with TensorBoard logging ([016d390](016d390bb58b497cf325e411e4221205db2f2eb5))
- *(evaluation)* Implement agent evaluation and visualization tools ([4138be1](4138be1f43493c6a2e8ecdb2c8a37c1f3e234b76))
- *(environment)* Implement OrderExecutionEnv gymnasium environment ([f0347a7](f0347a7082978b8daa756e5c6b812335e1194a6f))
- *(agent)* Implement DQN agent with experience replay ([0aadfc2](0aadfc2ae07540c8ffcbee2add655b63becc0c31))
- *(settings)* Implement typed configuration management with Pydantic ([22bd4d6](22bd4d68be08dd94ebf73ca0b12f333f20f1c2c4))
- *(config)* Add initial configuration for simulation and RL parameters ([e197324](e197324a749f265cf76d93f0a5e4c328bad812d5))
- *(main)* Implement training and evaluation entry point ([ce674e0](ce674e04c2489c87e5d6c3671d8c82ac80bd24be))
- *(core)* Add initial main.py entry point ([65804aa](65804aa764d81234aad155331f12ec4eab9092fc))

### Miscellaneous Tasks

- *(cliff)* Restore version and type grouping in changelog ([fb660a2](fb660a282102beed5fe9a0712e157236c73a4560))
- *(docs)* Update changelog ([9d40217](9d40217eddcb2b8bb8b83b8af0c3d9fafa46cffc))
- *(cliff)* Switch to date-based changelog grouping ([e8c4af0](e8c4af0142067487b2f5b7020d85a83cef3f5b99))
- *(docs)* Update changelog ([e11be01](e11be01afcfd6ad7234c9e30615bdd0a4dd98fae))
- *(changelog)* Bump checkout to v6 ([0c7fe6b](0c7fe6b145a134968a734db2e33c24c05061ae53))
- *(changelog)* Bump git-cliff-action to v4 ([cd4b8f5](cd4b8f58d90adea72cc152e92ad1992e5dfde3f8))
- *(changelog)* Add workflow to auto-generate changelog ([7adb9a9](7adb9a9dfde3d967f1a68128b76ef121fc06ce99))
- *(changelog)* Add git-cliff configuration ([4ef3114](4ef3114bc9725a84438217554e376e96de5b3ca6))
- *(config)* Update RL best parameters for gamma and lr ([98f19fc](98f19fc384b260c5de0e88d64b0eb37a0497fceb))
- *(gitignore)* Update project output ignore rules ([d669911](d669911e8b33cbd78b83ce0b6300749a598e6319))
- *(config)* Add best_params.yaml to track optimized hyperparameters ([cc7ba54](cc7ba544b47bf02cf3ad6a9a9edc641eaaa4bab8))
- *(deps)* Add optuna dependency for hyperparameter tuning ([012c811](012c811e14a0209b4a1fb183a3646baec8c08677))
- *(deps)* Add optuna dependency for hyperparameter tuning ([af32b47](af32b47dcd8bf30873d63178a9f92de7efa2a2e9))
- *(makefile)* Add optimize target for hyperparameter tuning ([68173f8](68173f801e49fb389da4866af62f54e5dac7d396))
- *(gitignore)* Add database artifacts and mypy cache to ignored files ([38fb384](38fb3846915a5a15ff5b9275038cc861fd354a26))
- *(workflows)* Add documentation auto-update workflow ([300860b](300860b74098e477ab435447648ab91a915a76e9))
- *(deps)* Add settings-doc to dev dependencies ([78aa82a](78aa82a28a9ad55aede27a75a0247ed2934b8551))
- *(workflows)* Add CI pipeline for quality checks ([e1d27c7](e1d27c75cec70e75ecc92daa5ac2f2ad2ecc108b))
- *(.gitattributes)* Treat uv.lock as binary ([401bde7](401bde7b74af0cf7b38ba5259171ea2bc4043b7e))
- *(.gitignore)* Ignore TensorBoard runs directory ([e8c2687](e8c2687760a63100a09b6060a6e144a114d916cb))
- *(git)* Add .gitignore to exclude artifacts and venv ([2743b3c](2743b3cb7bc890392475d47f7c824da57fa1d21d))

### Refactor

- *(settings)* Update config path and enrich logging schema ([4027e9d](4027e9ddb8cdaf7543f9aceb3fded22589a830b9))

### Styling

- *(eval)* Auto-format code with black/linter ([9e5ad0a](9e5ad0a2148d5f3754d513a9871c56f74d912f74))
- *(readme)* Align comments in project structure tree ([804efd3](804efd3de9241e6d329902dc73dfd74b5a427e55))
- *(training)* Auto-format code with ruff ([88cc8ba](88cc8ba9c685cda51c0a47bf5ca8584ffa113ff7))
- *(settings)* Auto-format code with ruff ([af055d4](af055d40e51d13b781c198ee05fdbb23f856a8ea))

### Testing

- *(tests)* Add initial simulation test suite ([bc3b3d7](bc3b3d7df345a4324c67c12fb3327fd2641d0c71))

### Build

- *(pre-commit)* Add configuration for local hooks ([d55d458](d55d4583d5458ab9d59050b5d479748b79dfa652))
- *(deps)* Add pre-commit to dev dependencies ([82408de](82408deea28343289ed766d9dab93768e4cb44a6))
- *(makefile)* Add setup-hooks target ([d83bf9a](d83bf9a9d6d9c7ff2d4de1d86939f0c1b4c99751))
- *(deps)* Add seaborn dependency ([c5ff8f5](c5ff8f5c6aee27c50a1b0ccb2212364e9dd5691b))
- *(makefile)* Add docker commands and docs automation ([99e300f](99e300f73b8af0f5bb4058762d7856b2d55e2d39))
- *(docker)* Add containerization support ([9b54169](9b54169310eb76c5aa1924e01175e2a9109f513a))
- *(package)* Initialize source package structure ([c6f583b](c6f583b02eae39b4a68ffc8e275ea2e0f9ca6400))
- *(uv.lock)* Add initial dependency lockfile ([8def62c](8def62c7605a8d2dd514234bd838634418564ead))
- *(pyproject.toml)* Define dependencies and tool configuration ([0d44d7f](0d44d7f51dd43a3f6d3afda4a08517a1ec279228))
- *(Makefile)* Add runs directory to clean target ([c984a54](c984a54d87da9b3c9b590ee8442498a61e66dd5f))
- *(Makefile)* Add Makefile for project automation and quality checks ([6edf2c7](6edf2c736adb291313d55d53d2e3d7f51ed6e4dd))
- *(uv)* Initialize project configuration and dependencies ([d98f60f](d98f60fe2c44380a6794d72aac13faaac515a223))

<!-- generated by git-cliff -->

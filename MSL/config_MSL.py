CONFIG_TRAIN = dict(DEFAULT_NET_PARAM={'cpu_only': False, 'regularizer': None, "n_classes_localization": 504},
                    DEFAULT_COST_PARAM={"multi_source_localization": True},
                    DEFAULT_RUN_PARAM={'learning_rate': 1e-3,
                                       'batch_size': 16,
                                       'testing': False,
                                       'model_version': ['100000'],  # orig model weights
                                       "display_step": 25,
                                       "total_steps": 10000,
                                       "checkpoint_step": 335},
                    DEFAULT_DATA_PARAM={"augment": True}
                    )

CONFIG_TEST = dict(DEFAULT_NET_PARAM={'cpu_only': False, 'regularizer': None, "n_classes_localization": 504,
                                      "decision_criterion": 0.009},
                   DEFAULT_COST_PARAM={"multi_source_localization": True},
                   DEFAULT_RUN_PARAM={'learning_rate': 1e-3,
                                      'batch_size': 16,
                                      'testing': True,
                                      'model_version':
                                          [str(x) for x in range(CONFIG_TRAIN["DEFAULT_RUN_PARAM"]["checkpoint_step"],
                                                                 CONFIG_TRAIN["DEFAULT_RUN_PARAM"]["total_steps"],
                                                                 CONFIG_TRAIN["DEFAULT_RUN_PARAM"]["checkpoint_step"])]},
                   DEFAULT_DATA_PARAM={}
                   )
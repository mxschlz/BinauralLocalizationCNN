CONFIG_TRAIN = dict(DEFAULT_NET_PARAM={'cpu_only': False, 'regularizer': None, "n_classes_localization": 504},
                    DEFAULT_COST_PARAM={"multi_source_localization": True},
                    DEFAULT_RUN_PARAM={'learning_rate': 1e-3,
                                       'batch_size': 16,
                                       'testing': False,
                                       'model_version': ['100000'],  # orig model weights
                                       "display_step": 1,
                                       "total_steps": 10000,
                                       "checkpoint_step": 10},
                    DEFAULT_DATA_PARAM={}
                    )

CONFIG_TEST = dict(DEFAULT_NET_PARAM={'cpu_only': False, 'regularizer': None, "n_classes_localization": 504,
                                      "decision_criterion": 0.009},
                   DEFAULT_COST_PARAM={"multi_source_localization": True},
                   DEFAULT_RUN_PARAM={'learning_rate': 1e-3,
                                      'batch_size': 16,
                                      'testing': True,
                                      'model_version': ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100",
                                                        "110", "120", "130", "140", "150", "160", "170"]},
                   DEFAULT_DATA_PARAM={}
                   )
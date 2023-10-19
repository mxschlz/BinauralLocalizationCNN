CONFIG_TRAIN = dict(DEFAULT_NET_PARAM={'cpu_only': True, 'regularizer': None, "n_classes_localization": 504},
                    DEFAULT_COST_PARAM={"multi_source_localization": False},
                    DEFAULT_RUN_PARAM={'learning_rate': 1e-3,
                                       'batch_size': 16,
                                       'testing': False,
                                       'model_version': ['100000'],  # orig model weights
                                       "display_step": 1,
                                       "total_steps": 10000,
                                       "checkpoint_step": 10}
                    )

CONFIG_TEST = dict(DEFAULT_NET_PARAM={'cpu_only': True, 'regularizer': None, "n_classes_localization": 504},
                   DEFAULT_COST_PARAM={"multi_source_localization": False},
                   DEFAULT_RUN_PARAM={'learning_rate': 1e-3,
                                      'batch_size': 16,
                                      'testing': True,
                                      'model_version': ['100000']}  # orig model weights
                   )
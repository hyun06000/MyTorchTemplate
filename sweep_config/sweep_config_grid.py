sweep_config_grid = {
    "method" : "grid",
    "parameters" : {
        "EPOCHS" : {
            "value" : 5
        },
        "BATCH_SIZE" : {
            "values" : [32, 64, 128]
        },
        "LEARNING_RATE" :{
            "values" : [0.001, 0.01, 0.1]
        },
    }
}
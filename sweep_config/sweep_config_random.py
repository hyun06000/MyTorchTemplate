sweep_config_random = {
    "method" : "random", 
    "parameters" : {
        "EPOCHS" : {
            "value" : 10
        },
        "BATCH_SIZE" : {
            "values" : [16, 32, 64, 128]
        },
        "LEARNING_RATE" :{
            "min": 0.0001, "max": 0.1
        },
    }
}

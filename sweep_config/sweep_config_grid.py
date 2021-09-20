sweep_config_grid = {
    "method" : "grid",
    "parameters" : {
        "EPOCHS" : {
            "value" : 5
        },
        "BATCH_SIZE" : {
            "values" : [16, 32, 64, 128]
        },
        "LEARNING_RATE" :{
            "min": 0.0001, "max": 0.1
        },
        "SAVEFIG_NAME"  : {
            "value":"result.png"
        },
        "NOW" : {
            "value" : "None"
        },
        "NUM_WORKERS":{
            "value" : 1
        },
        "DATASET":{
            "value": "MNIST-handwrite"
        },
        "DATA_PATH":{
            "value" : '../dataset/'
        },
        "ARCHITECTURE":{
            "value":"single-layer-perceptron",
        }
    }
}
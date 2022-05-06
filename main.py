import tensorflow as tf
from Model import build_model
from get_data import get_data

DIM_IMG = [256,256]
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100

def main():
    train_data,test_data = get_data()
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss=tf.keras.losses.MeanSquaredError())

    model.fit(train_data,
              epochs=NUM_EPOCHS,
              validation_data=train_data,
              batch_size = 32
            )
    outputs = model.predict(test_data)
    
if __name__ == "__main__":
    main()
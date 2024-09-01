from src.model_training import build_model, train_model
from src.model_evaluation import plot_metrics, evaluate_model

# first name paths to data dir
train_data_dir = 'data/train'
val_data_dir = 'data/val'
test_data_dir = 'data/test'

# then building the model
model = build_model()

# then to train the model now
history = train_model(model, train_data_dir, val_data_dir, batch_size=32, epochs=10)

# now time to plot training metrics
plot_metrics(history)

# then i will evaluate the model
eval
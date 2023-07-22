from models.s9 import Model

model = Model(skip=True, dropout=0.01)
print(model.summary(input_size=(32, 3, 32, 32)))

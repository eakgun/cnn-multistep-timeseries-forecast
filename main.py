import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import TSMethods


#DO A MODEL SELECTION WITH TEST DATA !!!!
# INCLUDE OTHER HYPERPARAMETERS TO THE MODEL SELECTION !!!!

df = pd.read_csv('Datasets\date_time_temp_filtered_combined.csv',
                 parse_dates=[['Date', 'Time']], 
                 dayfirst=True, 
                 na_filter=True)


df.index = pd.to_datetime(df['Date_Time'])
df= df[df['temp'] != 0]

tempData = df['temp']
tempData = tempData.dropna()
tempData = tempData['2018':'2019']

# tempData = (tempData-tempData.min())/(tempData.max()-tempData.min())
#standardize

# tempData = (tempData - tempData.mean()) / (tempData.std())
# tempData = (tempData-tempData.min())/(tempData.max()-tempData.min())





#Gotta include these in hyperparam optimization 
WINDOW_WIDTH = 156
FORECAST_LENGTH = 15
# batchSize = 1024
predictHorizon = 72
no_epochs = 60
predThreshold = WINDOW_WIDTH + FORECAST_LENGTH + 1
temp = tempData['2018':'2019'][:-predThreshold]
to_pred = tempData['2018':'2019'][-predThreshold:]
# predThreshold = WINDOW_WIDTH + FORECAST_LENGTH + 1


#----------------ADF TEST  ----------------------------------------------------------------

# if TSMethods.stationarity_test(temp, return_p=True, print_res = False) > 0.05:
#     print("P Value is high. Consider Differencing: " + str(TSMethods.stationarity_test(temp, 
#                                                                                        return_p = True,
#                                                                                        print_res = False)))
# else:
#     TSMethods.stationarity_test(temp)


#------------------------------------------------------------------------------------------------

#----------------DATA RESHAPE FOR WINDOW ----------------

def df_to_X_y(df, window_size, forecast_length):
    df_as_np = df.to_numpy()
    X = []
    y = []
    print(len(df_as_np),' - ', window_size,' - ', forecast_length, ' = ',
          len(df_as_np) - window_size - forecast_length)
    for i in range(len(df_as_np) - window_size - forecast_length):
        row_x = [[a] for a in df_as_np[i:i+window_size]]
        
        X.append(row_x)
        # row_y = df_as_np[i+window_size]
        row_y = [b for b in df_as_np[i+window_size:i+window_size+forecast_length]]
        
        y.append(row_y)
        
    return np.array(X), np.array(y)

X, y = df_to_X_y(temp, WINDOW_WIDTH, FORECAST_LENGTH)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

#------------------------------------------------------------------------------------------------
# def dataPrep(WINDOW_WIDTH, FORECAST_LENGTH):
    
#     predThreshold = WINDOW_WIDTH + FORECAST_LENGTH + 1
   
#     temp = tempData['2018':'2019'][:-predThreshold]
#     to_pred = tempData['2018':'2019'][-predThreshold:]
    
#     X, y = df_to_X_y(temp, WINDOW_WIDTH, FORECAST_LENGTH)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    
#     return X_train, X_test, y_train, y_test, X_val, y_val, temp, to_pred, X, y


# filePath = './saved_models/CNN_cp3.h5'

# X, y = df_to_X_y(temp, WINDOW_WIDTH, FORECAST_LENGTH)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# X_train[:,:,0] = normalize(X_train[:,:,0])

#---------------- THE MODEL ARCHITECTURE  ----------------

from keras.models import Sequential, save_model, load_model
from keras.layers import *
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
import keras.backend as K
import keras_tuner as kt

class MyHyperModel(kt.HyperModel):
     
     

        
     
     
            
    def build(self, hp):
        
        model = Sequential()
        model.add(Conv1D(filters = 125, kernel_size=5, activation='relu', 
                         input_shape = X_train.shape[1:]))
        model.add(MaxPooling1D(pool_size=5))
        model.add(Flatten())
        for i in range(hp.Int("layers", min_value=1, max_value=5, step=1)):
            model.add(
                Dense(
                    units=hp.Int("units_" + str(i), min_value=32, max_value=1024, step=32),
                    activation=hp.Choice('act_'+ str(i), ['relu','linear']),
                )
            )
        model.add(Dense(FORECAST_LENGTH, activation="linear"))
        model.compile(
            optimizer=Adam(learning_rate=hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")), 
            loss="mse",
            metrics=["accuracy"],
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Int("batch_size", min_value=32, max_value=1024, step=32),
            shuffle = hp.Boolean("shuffle"),
            **kwargs,
            )



#
# tuner = kt.RandomSearch(
#     hypermodel=build_model,
#     objective="val_accuracy",
#     max_trials=10,
#     executions_per_trial=2,
#     overwrite=True,
#     directory="hyperParams",
#     project_name="HPS",
# )    



# tuner = kt.BayesianOptimization(
#     hypermodel=MyHyperModel(),
#     objective="val_accuracy",
#     max_trials=20,
#     executions_per_trial=2,
#     overwrite=True,
#     directory="hyperParams",
#     project_name="HPS"
# )    

tuner = kt.Hyperband(
    MyHyperModel(),
    objective = 'val_accuracy',
    max_epochs = 7,
    factor=3,
    hyperband_iterations=2,
    
    directory='HPS3',
    project_name="WORKTHISTIME2",
    overwrite=True
)




    
tuner.search_space_summary()



tuner.search(X_train, y_train, validation_data=(X_val,y_val) , epochs=5, verbose=2)
models = tuner.get_best_models(num_models=1)
best_model = models[0]
# Build the model.
# Needed for `Sequential` without specified `input_shape`.

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                             patience=5, min_lr=0.001)

hypermodel = MyHyperModel()
best_hp = tuner.get_best_hyperparameters()[0]
model = hypermodel.build(best_hp)
history = hypermodel.fit(best_hp, model, 
                         X_train, y_train,
                         validation_data=(X_val, y_val),
                         epochs=no_epochs)


# history = model.fit(X_train, 
#                     y_train, 
#                     validation_data=(X_val,y_val), 
#                     epochs=no_epochs,
#                     # batch_size=batchSize,
#                     # callbacks=[reduce_lr],
#                     verbose=1)


sns.set_theme(style="darkgrid")
sns.set(rc = {'figure.figsize':(20,10)})




plt.figure(2)
plt.plot(history.history['loss'], color = 'red', label='Loss')
plt.plot(history.history['val_loss'], color = 'green', label = 'Validation Loss')
plt.legend(['Loss', 'Val_loss'], loc='upper right')



#modeli  


# save_model(model, filePath)
# print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


#load ze modél 







    

train_pred = model.predict(X_train).flatten()

# test_pred = model.predict(X_test).flatten()




#X[len(X)//2:].flatten()

train_results = pd.DataFrame(data={'Actual':y_train.flatten() , 'Prediction':train_pred})
# test_results = pd.DataFrame(data={'Actual':y_test.flatten() , 'Prediction':test_pred})
sns.set(rc = {'figure.figsize':(20,10)})
plt.figure(3)
sns.lineplot(data=train_results[0:100])


plt.figure(1)
train_results[:168].plot()
# test_results[:168].plot()
plt.show()
#Future Forecast

X_forecast, y_forecast= df_to_X_y(to_pred, WINDOW_WIDTH, FORECAST_LENGTH)


def forecast(X, predictHorizon,  forecast_length):
    future = []
    currentStep =  np.array([X[0]]) #last step from the previous prediction
    realPredictHorizon = predictHorizon//forecast_length

    if realPredictHorizon < 1:
        realPredictHorizon = 1

    print(currentStep.shape)    
    
    print('Calculated predict horizon: ', realPredictHorizon)
    for i in range(realPredictHorizon):
        prediction = model.predict(currentStep) 
        # print(i)
        print(currentStep.shape)    

        future.append(prediction) 
        prediction = prediction.reshape(1, FORECAST_LENGTH, 1)
        currentStep = np.delete(currentStep,np.arange(FORECAST_LENGTH),axis=1) 
        currentStep = np.append(currentStep,prediction,1)
    return future, currentStep


future, currentStep = forecast(X_forecast, predictHorizon, FORECAST_LENGTH)        


future = np.array(future)
future = future.flatten()

forecast_results =  pd.DataFrame(data={'Actual':X_forecast.flatten()[:len(future)], 'Forecast':future})
forecast_error = (forecast_results['Actual']-forecast_results['Forecast'])
forecast_error = forecast_error.abs()
forecast_results['Error'] = forecast_error
# fig, axes = plt.plot(1, 1, sharex=True, figsize=(15,5))
maxIndex = forecast_results['Error'].idxmax()
minIndex = forecast_results['Error'].idxmin()


sns.set(rc = {'figure.figsize':(20,10)})
fig2 = plt.figure(5)
fig2.suptitle('FORECAST')
sns.lineplot(data=forecast_results)
# plt.axvline(maxIndex, color='red', linestyle='dashed')
plt.vlines(x=maxIndex, ymin=forecast_results['Actual'][maxIndex], 
           ymax=forecast_results['Forecast'][maxIndex], colors='red', ls='--', lw=2, label='vline_multiple')


MSE = mean_squared_error(forecast_results['Actual'], forecast_results['Forecast'])



model.summary()


print('MSE OF THE FORECAST: ',MSE, 
      'MAX ERROR: ', forecast_error[maxIndex], '°C',
      'WINDOW WITDH: ', WINDOW_WIDTH,
      'FORWARD WITDH: ', FORECAST_LENGTH, 
      'Batch Size: ', 'tbd')

K.clear_session()

# future = np.array(future)
# plt.figure(2)
# plt.plot(future.flatten())
# plt.show()


#after processing a sequence, reset the states for safety
# model.reset_states()





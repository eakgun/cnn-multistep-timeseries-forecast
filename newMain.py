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



WINDOW_WIDTH = 168
FORECAST_LENGTH = 30
batchSize = 1000
predictHorizon = 60
no_epochs = 50


predThreshold = WINDOW_WIDTH + FORECAST_LENGTH + 1
predThreshold = 400


temp = tempData['2018':'2019'][:-predThreshold]
to_pred = tempData['2018':'2019'][-predThreshold:]

temp = temp.dropna()
to_pred = to_pred.dropna()
# temp = temp[::6]

#----------------ADF TEST  ----------------------------------------------------------------

if TSMethods.stationarity_test(temp, return_p=True, print_res = False) > 0.05:
    print("P Value is high. Consider Differencing: " + str(TSMethods.stationarity_test(temp, 
                                                                                       return_p = True,
                                                                                       print_res = False)))
else:
    TSMethods.stationarity_test(temp)


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


#------------------------------------------------------------------------------------------------




# filePath = './saved_models/CNN_cp3.h5'

X, y = df_to_X_y(temp, WINDOW_WIDTH, FORECAST_LENGTH)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# X_train[:,:,0] = normalize(X_train[:,:,0])

#---------------- THE MODEL ARCHITECTURE  ----------------

from keras.models import Sequential, save_model, load_model
from keras.layers import *
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError



def trainModel(neuron,path):

    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.001, patience=100)
    model = Sequential()
    

    # model.add(LSTM(input_shape = X_train.shape[1:], units=125, return_sequences=True))
    model.add(Conv1D(filters = neuron, kernel_size=5, activation='relu', input_shape = X_train.shape[1:]))
    model.add(MaxPooling1D(pool_size=5))
    
    model.add(Flatten())
    # model.add(Conv1D(filters = 250, kernel_size=5, activation='relu', input_shape = X_train.shape[1:]))
    # model.add(MaxPooling1D(pool_size=5))
    # model.add(Flatten())

    # model.add(LeakyReLU(alpha=0.3))


    # model.add(LSTM(units=64, return_sequences=True))
    # model.add(LSTM(units=100))
    # model.add(Dropout(rate=0.2))
    model.add(Dense(1000, activation='relu'))


 
    # model.add(Dense(FORECAST_LENGTH, activation='relu'))

    model.add(Dense(units=FORECAST_LENGTH, activation='linear'))

    # model.summary()
    # Adam(learning_rate=0.001)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                metrics=['accuracy'] , 
                loss=MeanSquaredError())

    save_weights_at = path
    save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min')


    history = model.fit(X_train, 
                        y_train, 
                        validation_data=(X_val,y_val), 
                        epochs=no_epochs,
                        batch_size=batchSize,
                        callbacks=[save_best, rlrop],
                        shuffle=True,
                        verbose=0)
    return history, model, save_best


#----------- TRAINING and MODEL SELECTION -----------------

history_list = []
bestModel = []
scores = []
pathList = []
cnt = 0
patCount = 0
patience = 4
for i in range(100,2000,1):
    
    filePathTemp = f'./saved_models/CNN_save{i}.h5'
    history, model, best = trainModel(i, filePathTemp)
    model.reset_states()
    pathList.append(filePathTemp)
    save_model(model, filePathTemp)
    history_list.append(history)
    bestModel.append(best)
    score = model.evaluate(X_test, y_test, verbose = 1)
    scores.append(score)
    if cnt > patience:
        if score[0] > scores[-(patience+patCount)][0]:
            patCount = patCount + 1  
            if patCount >= patience:
                history = history_list[-(patience+1)]
                score = scores[-patience-1]
                modelIndex = i
                model = load_model(pathList[-patience-1], compile=True) 
                
                break
        else:
            patCount = 0 
    cnt = cnt + 1 
        
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}',
         f'For {i} no of neurons')
   
#--------------------------------------------



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

test_pred = model.predict(X_test).flatten()




#X[len(X)//2:].flatten()

train_results = pd.DataFrame(data={'Actual':y_train.flatten() , 'Prediction':train_pred})
test_results = pd.DataFrame(data={'Actual':y_test.flatten() , 'Prediction':test_pred})
sns.set(rc = {'figure.figsize':(20,10)})
plt.figure(3)
sns.lineplot(data=train_results[0:100])


plt.figure(1)
train_results[:168].plot()
test_results[:168].plot()
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
      'Batch Size: ', batchSize)



# future = np.array(future)
# plt.figure(2)
# plt.plot(future.flatten())
# plt.show()


#after processing a sequence, reset the states for safety
# model.reset_states()
# import required packages
import utils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras



# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__":
    verbose=True
    algorithm = "cbow"

    # 1. Load your saved model
    try:
        model = keras.models.load_model('models/')

        # 2. Load your testing data
        try:
            print("attempting to find previously preproceed test set")
            test_data = pd.read_pickle(os.path.join(os.getcwd(), 'data/NLP_test_Preproc.pickle'))
            print("DATA FOUND YEEEE BOII")
            print(train_data)
        except IOError:
            raw_test_data = utils.load_NLP_test_data('data/aclImdb/test/', verbose=False)

            # Preprocess data - I gotchu boo
            test_data = utils.preprocess_NLP_data(raw_test_data, verbose=False)

            #Save preprocessed data to save time between runs/tuning (~2 min per run)
            test_data.to_pickle(os.path.join(os.getcwd(), 'data/NLP_test_Preproc.pickle'))
            del raw_test_data

       # 3 Genterate proper embedded dataset using the new model prior to output training
        #  This allows us to reuse previous iterations
        try:
            print("Attempting to find previous trained "+ algorithm +" model")
            model = KeyedVectors.load(os.path.join(os.getcwd(), 'models/' + algorithm + '_model.blob')) #uncomment for production


            test_embedded_df = pd.read_pickle(os.path.join(os.getcwd(), 'data/NLP_test_'+algorithm+'.pickle'))
            print("Test Data. FOUND YEEEE BOII")
            print(test_embedded_df)

            del test_data

        except IOError:
            print("Generating vectorized training from dataframe")
            print(test_data)
            test_embedded_df = utils.embedd_dataset(test_data, model)

            if test_embedded_df is not None:
                test_embedded_df.to_pickle(os.path.join(os.getcwd(), 'data/NLP_test_'+algorithm+'.pickle'), compression='gzip')
                print("Reesult stored:" + str(os.path.join(os.getcwd(), 'data/NLP_test_'+algorithm+'.pickle')))
                del test_data




        # 3. Run prediction on the test data and print the test accuracy
        score = model.evaluate(test_data, test_labels, batch_size=1, verbose=True)

        scores = []
        predictions = []
        # cycle through each test point to get loss at each point
        for ii in range(0, test_data.shape[0]):
            # reshape to maintain dimensionality
            dat = np.expand_dims(test_data[ii], 0)
            lab = np.expand_dims(np.asarray(test_labels[ii]), 0)
            # scores.append(model.evaluate(dat, lab, verbose=verbose))
            predictions.append(model.predict(dat, verbose=verbose))
            scores.append(predictions[ii]-lab)
        plt.figure()
        plt.subplot(211)
        plt.title('NLP Test Predictions')
        plt.plot(np.squeeze(predictions), label='predicitons')
        plt.plot(test_labels, label='ground truth')
        plt.legend()
        plt.subplot(212)
        plt.title('NLP Prediction Difference')
        plt.plot(np.squeeze(scores))
        plt.legend()
        plt.show()

    except IOError:
        print("Loading model failed for testing")
        print("run train_NLP.py to generate, data/models")


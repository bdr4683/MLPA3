################################################################
# a3_q3.py
#
# (Small MNIST) A3 Question 3
################################################################

from debug import *
timer = DebugTimer("Initialization")

from data_sets_a3 import *
from classifiers import *
# import matplotlib.pyplot as plt
# from results_visualization import draw_results, draw_contours

RANDOM_SEED = 42

def show_digits( digits, clabel, y_in ):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10,3))
    images = digits.reshape( digits.shape[0], 8, 8 )
    for ax, image, label in zip(axes, images, y_in ):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title( clabel + ": %i" % label)

def show_cmatrix( title, y_test, y_predicted ):
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_predicted)
    disp.figure_.suptitle(title + " Confusion Matrix")
    print(f"{title} Confusion matrix:\n{disp.confusion_matrix}")

def main():
    timer.qcheck("Imports complete")

    # Load data
    ( digits, mnist_data_tuple ) = load_small_mnist(timer)
    ( X_train, X_test, y_train, y_test ) = mnist_data_tuple


    # Sanity checks
    #show_digits( X_train, 'Train', y_train )
    #show_cmatrix( "Test", y_test, y_test )
    #plt.show()


    # Grid search (SVM)
    C_values = [1.0,1.1,1.2]
    kernels = ['linear', 'rbf']
    gamma_values = [1.0, 1.1, 1.2]

    for kernel_value in kernels:
        for C_value in C_values:
            if kernel_value == 'linear':
                clf = svm.SVC(kernel='linear', C=C_value)
                train_test_clf(clf, "MNIST data", mnist_data_tuple)

                print("---- SUPPORT VECTORS:" + str(clf.n_support_))
                report_metrics( clf, f"Linear SVM, C={C_value}", y_test, clf.predict(X_test) )
            else:
                for gamma_value in gamma_values:
                    clf = svm.SVC(kernel=kernel_value, C=C_value, gamma=gamma_value)
                    train_test_clf(clf, "MNIST data", mnist_data_tuple)
                    print("----- SUPPORT VECTORS:" + str(clf.n_support_))
                    report_metrics( clf, "RBF SVM, gamma={gamma_value}, C={C_value}", y_test, clf.predict(X_test) )


    # Grid search (RF)
    nTrees = [ 1,2,3 ]
    maxDepth = [ 1,2,3 ]
    nFeatures = [ 1 ]
    parameter_values = [
            {"n_estimators": [1], "max_depth": [None], "max_features": [None] },
            {"n_estimators": nTrees, "max_depth": maxDepth, "max_features": nFeatures } ]
    clf = RandomForestClassifier( random_state=RANDOM_SEED )
    grid_out = grid_search(clf, "Metal Parts", parameter_values, mnist_data_tuple) 

    # oops, not performance metrics? how could we get performance from the models used in the RF grid search?

    timer.qcheck("Q3 program complete")
    print(timer)

if __name__ == "__main__":
    main()

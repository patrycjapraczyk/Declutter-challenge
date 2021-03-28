from models.model_exec import ModelExec


def main():
    # Runs k-fold validation for the original data set
    # Running this file generates a few warnings- SettingWithCopyWarning and ConvergenceWarning.
    # They do not affect the correctness of the program and can be ignored.
    # The resulting scores from k-fold validation procedure will be printed in the console.
    # Please comment-out lines 12 and 13 if executing code from lines 19 and 20
    # comment_vectoriser can be changed to 'BOW', 'B-NGRAM', 'TFIDF', 'W2V'
    # imbalance_sampling can be changed to 'SMOTE', 'ADASYN', 'RANDOM_UNDERSAMPLE', RANDOM_OVERSAMPLE'
    exec = ModelExec(comment_vectoriser='BOW', imbalance_sampling='ADASYN')
    exec.kfold_split(10, 0.5454545454545454, 0.45454545454545453, exec.data)

    # Uncomment line  to test models against a Python data set. Comment-out code on lines 12 and 13 if running this.
    # comment_vectoriser can be changed to 'BOW', 'B-NGRAM', 'TFIDF', 'W2V'
    # imbalance_sampling can be changed to 'SMOTE', 'ADASYN', 'RANDOM_UNDERSAMPLE', RANDOM_OVERSAMPLE'

    # exec = ModelExec(comment_vectoriser='BOW', imbalance_sampling='ADASYN')
    # exec.test_python_data()


if __name__ == "__main__":
    main()